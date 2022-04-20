import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    ReLU,
    SeparableConv2D,
)


def build_conv(filters, kernel_size, name = None):
    return Sequential([
        Conv2D(filters, kernel_size = 1, use_bias = False),
        BatchNormalization(trainable = False),
        ReLU(),
        Conv2D(filters, kernel_size, padding = 'same'),
        BatchNormalization(trainable = False),
        ReLU(),
    ], name)


def build_separable_conv(filters, kernel_size, name = None):
    return Sequential([
        Conv2D(filters, kernel_size = 1, use_bias = False),
        BatchNormalization(trainable = False),
        ReLU(),
        SeparableConv2D(filters, kernel_size, padding = 'same', use_bias = False),
        Conv2D(filters, kernel_size = 1, use_bias = False),
        BatchNormalization(trainable = False),
        ReLU(),
    ], name)


def build_avg_pool(filters, name = None):
    return Sequential([
        Conv2D(filters, kernel_size = 1, use_bias = False),
        BatchNormalization(trainable = False),
        ReLU(),
        AveragePooling2D(pool_size = 3, strides = 1, padding = 'same'),
        BatchNormalization(trainable = False),
    ], name)


def build_max_pool(filters, name = None):
    return Sequential([
        Conv2D(filters, kernel_size = 1, use_bias = False),
        BatchNormalization(trainable = False),
        ReLU(),
        MaxPool2D(pool_size = 3, strides = 1, padding = 'same'),
        BatchNormalization(trainable = False),
    ], name)


class FactorizedReduce(Model):

    def __init__(self, filters):
        super().__init__()
        self.conv1 = Conv2D(filters // 2, kernel_size = 1, strides = 2, use_bias = False)
        self.conv2 = Conv2D(filters // 2, kernel_size = 1, strides = 2, use_bias = False)
        self.bn = BatchNormalization(trainable = False)

    def call(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x[:, 1:, 1:, :])
        out = tf.concat([out1, out2], axis = 3)
        out = self.bn(out)
        return out
