import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Dense, MaxPool2D, Layer
from nni.nas.tensorflow.mutables import LayerChoice, InputChoice


class LeNetModelSpace(Model):

    def __init__(self):
        super().__init__()
        self.conv1 = LayerChoice([
            Conv2D(6, 3, padding = 'same', activation = 'relu'),
            Conv2D(6, 5, padding = 'same', activation = 'relu'),
            Conv2D(6, 7, padding = 'same', activation = 'relu'),
        ], key = 'conv1')
        self.pool = LayerChoice([
            MaxPool2D(2),
            MaxPool2D(3)],
            key = 'pool'
        )
        self.conv2 = LayerChoice([
            Conv2D(16, 3, padding = 'same', activation = 'relu'),
            Conv2D(16, 5, padding = 'same', activation = 'relu'),
            Conv2D(16, 7, padding = 'same', activation = 'relu'),
        ], key = 'conv2')
        self.conv3 = Conv2D(16, 1)

        self.skip_connect = InputChoice(
            n_candidates = 2,
            n_chosen = 1,
            key = 'skip_connect'
        )
        self.bn = BatchNormalization()

        self.gap = AveragePooling2D(2)
        self.fc1 = Dense(120, activation = 'relu')
        self.fc2 = LayerChoice([
            Dense(84, activation = 'relu'),
            Layer()
        ], key = 'fc2')
        self.fc3 = Dense(10)

    def call(self, x):

        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.pool(x)

        x0 = self.conv2(x)
        x1 = self.conv3(x0)

        x0 = self.skip_connect([x0, None])
        if x0 is not None:
            x1 += x0

        x = self.pool(self.bn(x1))

        x = self.gap(x)
        x = tf.reshape(x, [batch_size, -1])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
