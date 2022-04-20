import tensorflow_datasets as tfds


def cifar10_dataset():
    ds = tfds.as_numpy(tfds.load(
        'cifar10',
        batch_size = -1,
        as_supervised = True,
    ))
    (x_train, y_train) = ds['train']
    (x_test, y_test) = ds['test']
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    ds, info = tfds.load('cifar10', split = 'train', with_info = True)
    fig = tfds.show_examples(ds, info)
    fig.show()
