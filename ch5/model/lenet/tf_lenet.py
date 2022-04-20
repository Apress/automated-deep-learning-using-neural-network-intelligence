from nni.nas.tensorflow.mutables import InputChoice, LayerChoice

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D


# Creates Convolution Layer
def create_conv(kernel, filters):
    return Conv2D(
        filters = filters,
        kernel_size = kernel,
        activation = 'relu',
        padding = 'same'
    )


class TfLeNetSupernet(Model):

    def __init__(self):
        super().__init__()
        self.conv1 = LayerChoice([
            create_conv(kernel = 1, filters = 16),  # 0
            create_conv(kernel = 3, filters = 16),  # 1
            create_conv(kernel = 5, filters = 16)  # 2
        ], key = 'conv1')

        self.conv2 = LayerChoice([
            create_conv(kernel = 1, filters = 32),  # 0
            create_conv(kernel = 3, filters = 32),  # 1
            create_conv(kernel = 5, filters = 32)  # 2
        ], key = 'conv2')

        self.pool = MaxPool2D(2)
        self.flat = Flatten()

        # Choosing Decision Making Layers
        self.dm = InputChoice(n_candidates = 2, n_chosen = 1, key = 'dm')

        self.fc11 = Dense(256, activation = 'relu')
        self.fc12 = Dense(10, activation = 'softmax')
        self.fc2 = Dense(10, activation = 'softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flat(x)

        # branch 1
        x1 = self.fc12(self.fc11(x))
        # branch 2
        x2 = self.fc2(x)

        # Choosing one of the branches
        x = self.dm([
            x1,  # 0
            x2  # 1
        ])

        return x
