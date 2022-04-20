import os
import random
import matplotlib.pyplot as plt
import torch
from nni.algorithms.compression.v2.pytorch.pruning import LevelPruner
from ch6.datasets import mnist_dataset
from ch6.model.pt_lenet import PtLeNetModel

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Making script reproducible
random.seed(1)
torch.manual_seed(1)

# Loading trained model
model = PtLeNetModel()
path = f'{CUR_DIR}/../data/lenet.pth'
model.load_state_dict(torch.load(path))

# Loading Dataset
train_ds, test_ds = mnist_dataset()

# Accuracy of original model
original_acc = model.test_model(test_ds)

# Non-zero weights of original model
original_nzw = model.count_nonzero_weights()

print(f'Original Model. '
      f'Non zero weight: {original_nzw}. '
      f'Accuracy: {original_acc}.')
#

# Pruning Config
prune_config = [{
    'sparsity': .8,
    'op_types': ['default'],
}]
# LevelPruner
pruner = LevelPruner(model, prune_config)
# Compressing the model
model_pruned, _ = pruner.compress()

# Retraining (fine-tuning) compressed model
epochs = 10
acc_list = []
for epoch in range(1, epochs + 1):
    model_pruned.train_model(epochs = 1, train_dataset = train_ds)
    acc = model_pruned.test_model(test_dataset = test_ds)
    acc_list.append(acc)
    print(f'Pruned: Epoch {epoch}. Accuracy: {acc}')

# Displaying Results
pruned_nzw = model_pruned.count_nonzero_weights()
plt.title(
    'Fine-tuning\n'
    f'Original Non-zero weights number: {original_nzw}\n'
    f'Pruned Non-zero weights number: {pruned_nzw}')
plt.axhline(y = original_acc, c = "red",
            label = 'Original model accuracy')
plt.plot(acc_list, label = 'Pruned model accuracy')
plt.xlabel('Retraining Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Saving Pruned Model
model_path = f'{CUR_DIR}/../data/lenet_pruned.pth'
mask_path = f'{CUR_DIR}/../data/mask.pth'
pruner.export_model(
    model_path = model_path,
    mask_path = mask_path
)
