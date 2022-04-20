import torch
import numpy as np
import matplotlib.pyplot as plt
from ch2.lenet_to_alexnet.pt_lenet_evolution import PtLeNetEvolution
from ch2.utils.datasets import hoh_dataset

# Best Hyperparameters
hparams = {
    "fe_slot_1":     {
        "_name":     "with_pool",
        "filters":   32,
        "kernel":    7,
        "pool_size": 5
    },
    "fe_slot_2":     {
        "_name":     "with_pool",
        "filters":   8,
        "kernel":    11,
        "pool_size": 5
    },
    "fe_slot_3":     {
        "_name":   "simple",
        "filters": 8,
        "kernel":  7
    },
    "l1_size":       1024,
    "l2_size":       512,
    "dropout_rate":  0.3,
    "learning_rate": 0.0001
}

feat_ext_sequences = []
for k, v in hparams.items():
    if k.startswith('fe_slot_'):
        v['type'] = v['_name']
        feat_ext_sequences.append(v)

model = PtLeNetEvolution(
    feat_ext_sequences = feat_ext_sequences,
    l1_size = hparams['l1_size'],
    l2_size = hparams['l2_size'],
    dropout_rate = hparams['dropout_rate'],
    learning_rate = hparams['learning_rate']
)

accuracy = model.train_and_test_model(
    batch_size = 16,
    epochs = 50
)

# HoH Dataset
(_, _), (x_test, y_test) = hoh_dataset()
x_test = torch.from_numpy(x_test).float()
x_test = torch.permute(x_test, (0, 3, 1, 2))

# Predictions
model.eval()
output = model(x_test)
y_pred = output.argmax(dim = 1, keepdim = True)

# Collecting Failed Predictions
number_of_fails_left = 3
fails = []
for i in range(len(x_test)):
    if number_of_fails_left == 0:
        break
    if y_pred[i] != y_test[i]:
        fails.append((x_test[i], (y_pred[i], y_test[i])))
        number_of_fails_left -= 1

# Displaying Failed Predictions
fig, axs = plt.subplots(1, 3)
for i in range(len(fails)):
    sample, (pred, actual) = fails[i]
    img = np.array(sample, dtype = 'float')
    img = img.transpose((1, 2, 0))
    ax = axs[i % 3]
    ax.set_title(f'#{i+1}')
    # ax.set_title(f'#{i+1}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img, interpolation = 'nearest')
plt.show()
