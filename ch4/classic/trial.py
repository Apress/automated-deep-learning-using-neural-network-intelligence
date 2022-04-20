import os
import sys
import numpy as np
import tensorflow as tf
import nni
from nni.algorithms.nas.tensorflow.classic_nas import get_and_apply_next_architecture

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append(SCRIPT_DIR)

from ch4.classic.model import LeNetModelSpace

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)


def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training = training)

    return loss_object(y_true = y, y_pred = y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training = True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train_model(net, train_dataset, optimizer, num_epochs):
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for x, y in train_dataset:
            loss_value, grads = grad(net, x, y)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, net(x, training = True))

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))


def test_model(model, test_dataset):
    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(x, training = False)
        prediction = tf.argmax(logits, axis = 1, output_type = tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    return test_accuracy.result()


if __name__ == '__main__':

    epochs = 10

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis = -1)
    y_train = np.expand_dims(y_train, axis = -1)
    x_test = np.expand_dims(x_test, axis = -1)
    y_test = np.expand_dims(y_test, axis = -1)
    # x_test.shape = (bs, 28, 28, 1)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    split = int(len(x_train) * 0.9)
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train[:split], y_train[:split])).batch(64)
    dataset_valid = tf.data.Dataset.from_tensor_slices((x_train[split:], y_train[split:])).batch(64)
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

    net = LeNetModelSpace()
    get_and_apply_next_architecture(net)

    train_model(net, dataset_train, optimizer, epochs)

    acc = test_model(net, dataset_test)

    nni.report_final_result(acc.numpy())
