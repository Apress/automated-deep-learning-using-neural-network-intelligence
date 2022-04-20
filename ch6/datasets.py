import tensorflow_datasets as tfds


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
