from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D
from ch5.naive_one_shot_nas.tf.tf_ops import create_conv


class TfLeNetMultiTrialModel(Model):

    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.conv1 = create_conv(kernel1, filter = 16)
        self.pool1 = MaxPool2D(pool_size = 2)
        self.conv2 = create_conv(kernel2, filter = 32)
        self.pool2 = MaxPool2D(pool_size = 2)
        self.flatten = Flatten()
        self.fc1 = Dense(128, 'relu')
        self.fc2 = Dense(10, 'softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
