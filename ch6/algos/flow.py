import copy
import os
import torch
from nni.compression.pytorch import ModelSpeedup
from torchsummary import summary

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def prune_flow(
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
    model_pruned = pruner.compress()

    # Step 4: Retraining compressed model
    for epoch in range(1, epochs + 1):
        pruner.update_epoch(epoch)
        model_pruned.train_model(epochs = 1, train_dataset = train_ds)

    # Step 5: Saving model
    model_path = f'{CUR_DIR}/../data/{pruner_name}_pruned.pth'
    mask_path = f'{CUR_DIR}/../data/{pruner_name}_mask.pth'
    pruner.export_model(model_path = model_path, mask_path = mask_path)

    # Step 6: Speeding Up Model
    dummy_input = torch.randn(model_input_shape)
    model_pruned = model_original.__class__()
    model_pruned.load_state_dict(torch.load(model_path))
    speedup = ModelSpeedup(model_pruned, dummy_input, mask_path)
    speedup.speedup_model()
    model_pruned.eval()

    return model_pruned


def model_comparison(original, compressed, test_ds, input_shape):
    print(f'Original accuracy: {original.test_model(test_ds)}')
    print(f'Compressed accuracy: {compressed.test_model(test_ds)}')

    print('==== ORIGINAL MODEL =====')
    summary(original, input_shape)
    print('=========================')

    print('====  PRUNED MODEL  =====')
    summary(compressed, input_shape)
    print('=========================')
