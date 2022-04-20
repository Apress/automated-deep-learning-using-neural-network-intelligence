import torch.nn as nn
from nni.retiarii.oneshot.pytorch.enas import EnasTrainer
from torch.optim.sgd import SGD
import ch5.datasets as datasets
from ch5.model.general.pt_general import GeneralSupernet
from ch5.pt_utils import accuracy, reward_accuracy

# Dataset
dataset_train, dataset_valid = datasets.get_dataset("cifar10")

# Supernet
model = GeneralSupernet()

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = SGD(
    model.parameters(), 0.05,
    momentum = 0.9, weight_decay = 1.0E-4
)

# Trainer parameters
batch_size = 128
log_frequency = 10
num_epochs = 100
metrics = accuracy
reward_function = reward_accuracy

# RL Controller parameters
ctrl_kwargs = {"tanh_constant": 1.1}

# ENAS Trainer
trainer = EnasTrainer(
    model,
    loss = criterion,
    metrics = metrics,
    reward_function = reward_function,
    optimizer = optimizer,
    batch_size = batch_size,
    num_epochs = num_epochs,
    dataset = dataset_train,
    log_frequency = log_frequency,
    ctrl_kwargs = ctrl_kwargs)

# Training
trainer.fit()

# Best Model
best_model = trainer.export()
print(best_model)
