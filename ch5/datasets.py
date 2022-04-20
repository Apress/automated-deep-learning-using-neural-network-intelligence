import tensorflow_datasets as tfds
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
import tensorflow as tf


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


def cifar10_dataset():
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    train_set = (x_train, y_train)
    valid_set = (x_valid, y_valid)
    return train_set, valid_set


def get_dataset(cls):
    MEAN = [0.49139968]
    STD = [0.24703233]
    transf = [
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root = "./data", train = True, download = True, transform = train_transform)
        dataset_valid = CIFAR10(root = "./data", train = False, download = True, transform = valid_transform)
    elif cls == "mnist":
        dataset_train = MNIST(root = "./data", train = True, download = True, transform = train_transform)
        dataset_valid = MNIST(root = "./data", train = False, download = True, transform = valid_transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid
