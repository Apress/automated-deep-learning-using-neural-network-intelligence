import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def mnist_dataset():
    ds = tfds.as_numpy(tfds.load(
        'mnist',
        batch_size = -1,
        as_supervised = True,
    ))
    (x_train, y_train) = ds['train']
    (x_test, y_test) = ds['test']
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def hoh_dataset():
    """Download Horses and Humans Dataset"""
    x, y = tfds.as_numpy(tfds.load(
        'horses_or_humans',
        batch_size = -1,
        as_supervised = True,
        split = 'train'
    ))
    x = x / 255.0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':

    (_, _), (x_test, y_test) = hoh_dataset()

    for i in range(1):
        img = np.array(x_test[i], dtype = 'float')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, interpolation = 'nearest')
        plt.show()
