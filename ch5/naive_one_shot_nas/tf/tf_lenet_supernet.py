from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D
from ch5.naive_one_shot_nas.tf.tf_ops import create_conv


class TfLeNetNaiveSupernet(Model):

    def __init__(self):
        super().__init__()

        self.conv1_1 = create_conv(1, 16)
        self.conv1_3 = create_conv(3, 16)
        self.conv1_5 = create_conv(5, 16)

        self.conv2_1 = create_conv(1, 32)
        self.conv2_3 = create_conv(3, 32)
        self.conv2_5 = create_conv(5, 32)

        self.pool1 = MaxPool2D(pool_size = 2)
        self.pool2 = MaxPool2D(pool_size = 2)
        self.flatten = Flatten()
        self.fc1 = Dense(128, 'relu')
        self.fc2 = Dense(10, 'softmax')

    def call(self, x, mask = None):

        # Sum all in training mode
        if mask is None:
            mask = [[1, 1, 1], [1, 1, 1]]

        x = mask[0][0] * self.conv1_1(x) +\
            mask[0][1] * self.conv1_3(x) +\
            mask[0][2] * self.conv1_5(x)
        x = self.pool1(x)

        x = mask[1][0] * self.conv2_1(x) +\
            mask[1][1] * self.conv2_3(x) +\
            mask[1][2] * self.conv2_5(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
