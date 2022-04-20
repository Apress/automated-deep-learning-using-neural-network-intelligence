import tensorflow as tf
from sklearn.metrics import accuracy_score
from ch5.datasets import mnist_dataset
from ch5.naive_one_shot_nas.tf.tf_lenet_supernet import TfLeNetNaiveSupernet
from ch5.naive_one_shot_nas.tf.tf_train import train

# Initializing Supernet
model = TfLeNetNaiveSupernet()

# Training Supernet
train(model)

# Loading test dataset
_, (x, y) = mnist_dataset()

kernel1_choices = [1, 3, 5]
kernel2_choices = [1, 3, 5]

# Evaluating Supernet activating each candidate
results = {}
for m1 in range(0, len(kernel1_choices)):
    for m2 in range(0, len(kernel2_choices)):
        # activation mask
        mask = [[0, 0, 0], [0, 0, 0]]
        # activating conv1 and conv2 layers
        mask[0][m1] = 1
        mask[1][m2] = 1

        # calculating accuracy
        output = model(x, mask = mask)
        predict = tf.argmax(output, axis = 1)
        accuracy = round(accuracy_score(predict, y), 4)
        results[(kernel1_choices[m1], kernel2_choices[m2])] = accuracy

# Displaying results
print('=======')
print('Results:')
for k, v in results.items():
    print(f'Conv1 {k[0]}x{k[0]}, Conv2: {k[1]}x{k[1]} : {v}')
