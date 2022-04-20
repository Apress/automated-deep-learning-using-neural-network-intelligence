from tensorflow.keras.optimizers import Adam
from ch5.datasets import mnist_dataset


def train(model):
    learning_rate = 0.001
    batch_size = 256
    model.compile(
        optimizer = Adam(learning_rate = learning_rate),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    (x_train, y_train), _ = mnist_dataset()

    model.fit(
        x_train,
        y_train,
        batch_size = batch_size,
        epochs = 10,
        verbose = 1
    )


def test(model):
    """Testing Trained Model Performance"""
    (_, _), (x_test, y_test) = mnist_dataset()
    loss, accuracy = model.evaluate(x_test, y_test, verbose = 1)
    return accuracy
