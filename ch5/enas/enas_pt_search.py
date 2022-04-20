import torch.nn as nn
from nni.retiarii.oneshot.pytorch.enas import EnasTrainer
from torch.optim.sgd import SGD
import ch5.datasets as datasets
from ch5.model.lenet.pt_lenet import PtLeNetSupernet
from ch5.pt_utils import accuracy, reward_accuracy

# Supernet
model = PtLeNetSupernet()

# Dataset
dataset_train, dataset_valid = datasets.get_dataset("mnist")

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = SGD(
    model.parameters(), 0.05,
    momentum = 0.9, weight_decay = 1.0E-4
)

# Train params
batch_size = 256
log_frequency = 50
num_epochs = 10

# RL Controller params
ctrl_kwargs = {"tanh_constant": 1.1}

# ENAS Trainer
trainer = EnasTrainer(
    model,
    loss = criterion,
    metrics = accuracy,
    reward_function = reward_accuracy,
    optimizer = optimizer,
    batch_size = batch_size,
    num_epochs = num_epochs,
    dataset = dataset_train,
    log_frequency = log_frequency,
    ctrl_kwargs = ctrl_kwargs,
    ctrl_steps_aggregate = 20
)

# Training
trainer.fit()

# Best Model
best_model = trainer.export()
print(best_model)

# Done
# {'conv1': 1, 'conv2': 1, 'dm': 0}
