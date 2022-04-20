import torch
import torch.nn as nn
import ch5.datasets as datasets
from nni.retiarii.oneshot.pytorch import DartsTrainer
from ch5.model.general.pt_general import GeneralSupernet
from ch5.pt_utils import accuracy

# Supernet
model = GeneralSupernet()

# Dataset
dataset_train, dataset_valid = datasets.get_dataset("cifar10")

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optim = torch.optim.SGD(
    model.parameters(), 0.025,
    momentum = 0.9, weight_decay = 3.0E-4
)

# Supernet training params:
num_epochs = 100
batch_size = 128
accuracy_metrics = accuracy

# DARTS Trainer
trainer = DartsTrainer(
    model = model,
    loss = criterion,
    metrics = accuracy_metrics,
    optimizer = optim,
    num_epochs = num_epochs,
    dataset = dataset_train,
    batch_size = batch_size,
    log_frequency = 10,
    unrolled = False
)

# Training
trainer.fit()

# Best Subnet
best_architecture = trainer.export()
print('Best architecture:', best_architecture)
