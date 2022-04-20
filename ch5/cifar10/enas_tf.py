from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
import ch5.datasets as datasets
from nni.algorithms.nas.tensorflow import enas
from ch5.model.general.tf_general import GeneralSupernet
from ch5.tf_utils import accuracy, reward_accuracy, get_best_model

# Supernet
model = GeneralSupernet()

# Dataset
dataset_train, dataset_valid = datasets.cifar10_dataset()

# Loss Function
loss = SparseCategoricalCrossentropy(
    from_logits = True,
    reduction = Reduction.NONE
)

# Optimizer
optimizer = SGD(learning_rate = 0.05, momentum = 0.9)

# Trainer params
metrics = accuracy
reward_function = reward_accuracy
batch_size = 256
num_epochs = 100

# ENAS Trainer
trainer = enas.EnasTrainer(
    model,
    loss = loss,
    metrics = metrics,
    reward_function = reward_function,
    optimizer = optimizer,
    batch_size = batch_size,
    num_epochs = num_epochs,
    dataset_train = dataset_train,
    dataset_valid = dataset_valid
)

# Training
trainer.train()

# Best model
best = get_best_model(trainer.mutator)
print(best)