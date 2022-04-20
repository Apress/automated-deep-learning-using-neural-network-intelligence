import torch
import torch.nn as nn
import ch5.datasets as datasets
from nni.retiarii.oneshot.pytorch import DartsTrainer
from ch5.model.lenet.pt_lenet import PtLeNetSupernet
from ch5.pt_utils import accuracy

# Supernet
model = PtLeNetSupernet()

# Dataset
dataset_train, dataset_valid = datasets.get_dataset("mnist")

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optim = torch.optim.SGD(
    model.parameters(), 0.025,
    momentum = 0.9, weight_decay = 3.0E-4
)

# Trainer params
num_epochs = 10
batch_size = 256
metrics = accuracy

# DARTS Trainer
trainer = DartsTrainer(
    model = model,
    loss = criterion,
    metrics = metrics,
    optimizer = optim,
    num_epochs = num_epochs,
    dataset = dataset_train,
    batch_size = batch_size,
    log_frequency = 10,
    unrolled = False
)

# Training
trainer.fit()

# Best architecture
best_architecture = trainer.export()
print('Best architecture:', best_architecture)
