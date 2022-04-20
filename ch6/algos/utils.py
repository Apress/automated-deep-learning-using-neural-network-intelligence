import copy
import os
import numpy as np
import torch
from nni.compression.pytorch import ModelSpeedup
from torchsummary import summary
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def oneshot_prune(
        model_original,
        pruner_cls,
        pruner_config,
        train_ds,
        epochs = 10,
        model_input_shape = (1, 1, 28, 28)
):
    """
    model_original: Pretrained original model
    pruner_cls: Pruner Class
    pruner_config: Pruner Configuration Dict
    train_ds: Train Dataset
    epochs: Number of Training Epochs
    model_input_shape: Shape of Model input
    """
    pruner_name = pruner_cls.__name__
    model = copy.deepcopy(model_original)

    # Step 2: Initializing pruner
    pruner = pruner_cls(model, pruner_config)

    # Step 3: Compressing the model
    model_pruned, mask = pruner.compress()

    # Step 4: Retraining compressed model
    for epoch in range(1, epochs + 1):
        model_pruned.train_model(
            epochs = 1,
            train_dataset = train_ds
        )

    # Step 5: Saving model
    model_path = f'{CUR_DIR}/../data/{pruner_name}_pruned.pth'
    mask_path = f'{CUR_DIR}/../data/{pruner_name}_mask.pth'
    pruner.export_model(
        model_path = model_path,
        mask_path = mask_path
    )

    # Step 6: Speeding Up Model
    dummy_input = torch.randn(model_input_shape)
    model_pruned = model_original.__class__()
    model_pruned.load_state_dict(torch.load(model_path))
    speedup = ModelSpeedup(model_pruned, dummy_input, mask_path)
    speedup.speedup_model()
    model_pruned.eval()

    return model_pruned, mask


def visualize_mask(masks):
    active_w = {}
    for i, layer in enumerate(masks):
        mask = masks[layer]['weight'].detach().numpy()
        active = np.sum(mask)
        total = mask.size
        active_w[layer] = active / total
    plt.title('Masked Weights')
    layer_range = range(len(active_w))
    plt.barh(layer_range, active_w.values(), label = 'Active Weights', color = 'green')
    plt.ylabel('Layer')
    plt.yticks(layer_range, list(active_w.keys()))
    plt.legend()
    plt.xlim(0, 1)
    plt.show()


def model_comparison(original, compressed, test_ds, input_shape):
    print(f'Original accuracy: {original.test_model(test_ds)}')
    print(f'Compressed accuracy: {compressed.test_model(test_ds)}')

    print('==== ORIGINAL MODEL =====')
    summary(original, input_shape)
    print('=========================')

    print('====  PRUNED MODEL  =====')
    summary(compressed, input_shape)
    print('=========================')
