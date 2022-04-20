import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, Dense,
    Dropout, Flatten, MaxPool2D, ReLU,
)
from tensorflow.keras.optimizers import Adam
from ch2.utils.datasets import hoh_dataset
from ch2.utils.tf_utils import TfNniIntermediateResult


class TfLeNetEvolution(Model):

    def __init__(
            self,
            feat_ext_sequences,
            l1_size,
            l2_size,
            dropout_rate
    ):
        """
        LeNet Evolution Model
        """
        super().__init__()

        layer_stack = []

        # ============
        # Constructing Feature Extraction Stack
        for fe_seq in feat_ext_sequences:
            if fe_seq['type'] in ['simple', 'with_pool']:
                # Constructing Feature Extraction Sequence
                layer_stack.append(
                    Conv2D(
                        filters = fe_seq['filters'],
                        kernel_size = fe_seq['kernel']
                    )
                )
                if fe_seq['type'] == 'with_pool':
                    layer_stack.append(
                        MaxPool2D(
                            pool_size = fe_seq['pool_size']
                        )
                    )
                layer_stack.append(ReLU())

        layer_stack.append(Flatten())

        # ============
        # Decision Maker Stack
        layer_stack.append(
            Dense(
                units = l1_size,
                activation = 'relu'
            )
        )
        layer_stack.append(
            Dropout(rate = dropout_rate)
        )

        # Optional Linear Layer
        if l2_size > 0:
            layer_stack.append(
                Dense(
                    units = l2_size,
                    activation = 'relu'
                )
            )
            layer_stack.append(
                Dropout(rate = dropout_rate)
            )

        layer_stack.append(
            Dense(
                units = 2,
                activation = 'softmax'
            )
        )

        self.seq = tf.keras.Sequential(layer_stack)

    def call(self, x, **kwargs):
        y = self.seq(x)
        return y

    def train(self, learning_rate, batch_size, epochs):
        self.compile(
            optimizer = Adam(learning_rate = learning_rate),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )

        (x_train, y_train), _ = hoh_dataset()

        intermediate_cb = TfNniIntermediateResult('accuracy')
        self.fit(
            x_train,
            y_train,
            batch_size = batch_size,
            epochs = epochs,
            verbose = 0,
            callbacks = [intermediate_cb]
        )

    def test(self):
        (_, _), (x_test, y_test) = hoh_dataset()
        loss, accuracy = self.evaluate(x_test, y_test, verbose = 0)
        return accuracy
