from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from ch2.utils.datasets import mnist_dataset
from ch2.utils.tf_utils import TfNniIntermediateResult


class TfLeNetModel(Model):

    def __init__(self, filter_size, kernel_size, l1_size):
        """
        LeNet Model Initialization
        HyperParameters:
        ----------
        filter_size: int
            Filter size of 1st convolutional layer.
        kernel_size : int
            Kernel size of convolutional layers.
        l1_size : int
            Dimensionality of 1st linear layer.
        """
        super().__init__()
        self.conv1 = Conv2D(
            filters = filter_size,
            kernel_size = kernel_size,
            activation = 'sigmoid'
        )
        self.pool1 = MaxPool2D(pool_size = 2)
        self.conv2 = Conv2D(
            filters = filter_size * 2,
            kernel_size = kernel_size,
            activation = 'sigmoid'
        )
        self.pool2 = MaxPool2D(pool_size = 2)
        self.flatten = Flatten()
        self.fc1 = Dense(
            units = l1_size,
            activation = 'sigmoid'
        )
        self.fc2 = Dense(
            units = 10,
            activation = 'softmax'
        )

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

    def train(self, learning_rate, batch_size):
        self.compile(
            optimizer = Adam(learning_rate = learning_rate),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )
        (x_train, y_train), _ = mnist_dataset()

        # Intermediate NNI Result
        intermediate_cb = TfNniIntermediateResult('accuracy')

        self.fit(
            x_train,
            y_train,
            batch_size = batch_size,
            epochs = 10,
            verbose = 0,
            callbacks = [intermediate_cb]
        )

    def test(self):
        """Testing Trained Model Performance"""
        (_, _), (x_test, y_test) = mnist_dataset()
        loss, accuracy = self.evaluate(x_test, y_test, verbose = 0)
        return accuracy
