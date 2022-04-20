import tensorflow as tf
from tensorflow.keras.layers import Dense

# Problem Parameters
inp_dim = 5
out_dim = 1

# Hyperparameters
l1_dim = 8
l2_dim = 4

# Model
model = tf.keras.Sequential(
    [
        Dense(l1_dim, name = 'l1',
              activation = 'sigmoid', input_dim = inp_dim),
        Dense(l2_dim, name = 'l2',
              activation = 'relu'),
        Dense(out_dim, name = 'l3'),
    ]
)
model.build()

# Weights and Biases
print(model.summary())
