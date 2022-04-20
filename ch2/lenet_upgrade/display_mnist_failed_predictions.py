from math import floor
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ch2.lenet_upgrade.tf_lenet_upgrade_model import TfLeNetUpgradeModel
from ch2.utils.datasets import mnist_dataset

# Best Hyperparameters
hparams = {
    "activation":    "relu",
    "use_dropout":   {
        "_name": 1,
        "rate":  0.5
    },
    "filter_size":   32,
    "kernel_size":   5,
    "l1_size":       256,
    "batch_size":    512,
    "learning_rate": 0.001
}

# Making this script Reproducible
tf.random.set_seed(1)

use_dropout = bool(hparams['use_dropout']['_name'])
model_params = {
    "filter_size": hparams['filter_size'],
    "kernel_size": hparams['kernel_size'],
    "l1_size":     hparams['l1_size'],
    "activation":  hparams['activation'],
    "use_dropout": use_dropout
}
if use_dropout:
    model_params['dropout_rate'] = hparams['use_dropout']['rate']

# Initializing LeNet Model
model = TfLeNetUpgradeModel(**model_params)

# Model Training
model.train(
    batch_size = hparams['batch_size'],
    learning_rate = hparams['learning_rate']
)

# MNIST Dataset
(_, _), (x_test, y_test) = mnist_dataset()

# Predictions
output = model(x_test)
y_pred = tf.argmax(output, 1)

# Collecting Failed Predictions
number_of_fails_left = 9
fails = []
for i in range(len(x_test)):
    if number_of_fails_left == 0:
        break
    if y_pred[i] != y_test[i]:
        fails.append((x_test[i], (y_pred[i], y_test[i])))
        number_of_fails_left -= 1

# Displaying Failed Predictions
fig, axs = plt.subplots(3, 3)
for i in range(len(fails)):
    sample, (pred, actual) = fails[i]
    img = np.array(sample, dtype = 'float')
    img = img * 255
    pixels = img.reshape((28, 28))
    ax = axs[floor(i / 3), i % 3]
    # ax.set_title(f'#{i+1}: {actual} ({pred})')
    ax.set_title(f'#{i+1}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(pixels, cmap = 'gray')
plt.show()
