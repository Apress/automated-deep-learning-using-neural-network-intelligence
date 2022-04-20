import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import\
    (
    AveragePooling2D, BatchNormalization, Conv2D, Dense, MaxPool2D,
)


class LeNetModel(Model):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(6, 3, padding = 'same', activation = 'relu')
        self.pool = MaxPool2D(2)
        self.conv2 = Conv2D(16, 3, padding = 'same', activation = 'relu')

        self.bn = BatchNormalization()

        self.gap = AveragePooling2D(2)
        self.fc1 = Dense(120, activation = 'relu')
        self.fc2 = Dense(84, activation = 'relu')
        self.fc3 = Dense(10)

    def call(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)

        x = self.pool(self.bn(x))

        x = self.gap(x)
        x = tf.reshape(x, [batch_size, -1])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
