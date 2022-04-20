import os
import random
import matplotlib.pyplot as plt
import torch
from ch6.model.pt_lenet import PtLeNetModel
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from ch8.utils import mnist_dataset

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
model_pruned = pruner.compress()

# Retraining compressed model
epochs = 10
acc_list = []
size_list = []
for epoch in range(1, epochs + 1):
    pruner.update_epoch(epoch)
    model_pruned.train_model(epochs = 1, train_dataset = train_ds)
    nzw = model_pruned.count_nonzero_weights()
    acc = model_pruned.test_model(test_dataset = test_ds)
    acc_list.append(acc)
    size_list.append(nzw)
    print(f'Pruned: Epoch {epoch}. Non zero weight: {nzw}. Accuracy: {acc}.')

# Displaying Results
fig, axs = plt.subplots(2)

axs[0].set_title('Accuracy')
axs[0].axhline(y = original_acc, c = "red")
axs[0].plot(acc_list)
axs[0].set_xticks([])

axs[1].set_title('Size')
axs[1].axhline(y = original_nzw, c = "red")
axs[1].plot(size_list)
axs[1].set_xticks([])

plt.show()

# Saving Pruned Model
model_path = f'{CUR_DIR}/../data/lenet_pruned.pth'
mask_path = f'{CUR_DIR}/../data/mask.pth'
pruner.export_model(model_path = model_path, mask_path = mask_path)
