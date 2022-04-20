from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import ch5.datasets as datasets
from nni.algorithms.nas.tensorflow import enas
from ch5.model.lenet.tf_lenet import TfLeNetSupernet
from ch5.tf_utils import accuracy, reward_accuracy, get_best_model

# Supernet
model = TfLeNetSupernet()

# Dataset
dataset_train, dataset_valid = datasets.mnist_dataset()

# Loss Function
loss = SparseCategoricalCrossentropy(
    from_logits = True,
    reduction = Reduction.NONE
)

# Optimizer
optimizer = Adam()

# Training Params
num_epochs = 10
batch_size = 256

# ENAS Trainer
trainer = enas.EnasTrainer(
    model,
    loss = loss,
    metrics = accuracy,
    reward_function = reward_accuracy,
    optimizer = optimizer,
    batch_size = batch_size,
    num_epochs = num_epochs,
    dataset_train = dataset_train,
    dataset_valid = dataset_valid,
    log_frequency = 10,
    child_steps = 10,
    mutator_steps = 30
)

# Training
trainer.train()

# Best model
best = get_best_model(trainer.mutator)
print(best)
