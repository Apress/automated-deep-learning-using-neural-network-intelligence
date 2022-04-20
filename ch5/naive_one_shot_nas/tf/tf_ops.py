from tensorflow.keras.layers import Conv2D


def create_conv(kernel, filter):
    return Conv2D(
        filters = filter,
        kernel_size = kernel,
        activation = 'relu',
        padding = 'same'
    )
