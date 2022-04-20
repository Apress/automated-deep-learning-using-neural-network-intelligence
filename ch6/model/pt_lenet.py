import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from ch6.datasets import mnist_dataset

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class PtLeNetModel(nn.Module):

    def __init__(self, fs = 16, ks = 5):
        super(PtLeNetModel, self).__init__()

        self.conv1 = nn.Conv2d(1, fs, ks)
        self.conv2 = nn.Conv2d(fs, fs * 2, ks)
        self.conv3 = nn.Conv2d(fs * 2, fs * 2, ks)

        self.fc1 = nn.Linear(288, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = torch.flatten(x, start_dim = 1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        return F.log_softmax(x, dim = 1)

    def train_model(
            self,
            learning_rate = 0.001,
            batch_size = 256,
            epochs = 20,
            train_dataset = None
    ):

        # Preparing Train Dataset
        if train_dataset:
            x, y = train_dataset
        else:
            (x, y), _ = mnist_dataset()

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        # Permute dimensions for PyTorch Convolutions
        x = torch.permute(x, (0, 3, 1, 2))
        dataset_size = x.shape[0]

        optimizer = optim.Adam(
            self.parameters(),
            lr = learning_rate
        )

        self.train()
        for epoch in range(1, epochs + 1):
            # Random permutations for batch training
            permutation = torch.randperm(dataset_size)
            for bi in range(1, dataset_size, batch_size):
                # Creating New Batch
                indices = permutation[bi:bi + batch_size]
                batch_x, batch_y = x[indices], y[indices]

                # Model Parameters Optimization
                optimizer.zero_grad()
                output = self(batch_x)
                loss = F.cross_entropy(output, batch_y)
                loss.backward()
                optimizer.step()

            # Epoch Intermediate metrics:
            output = self(x)
            predict = output.argmax(dim = 1, keepdim = True)
            accuracy = round(accuracy_score(predict, y), 4)
            print(F'Accuracy: {accuracy}')

    def test_model(self, test_dataset = None):
        self.eval()
        # Preparing Test Dataset
        if test_dataset:
            x, y = test_dataset
        else:
            _, (x, y) = mnist_dataset()
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        x = torch.permute(x, (0, 3, 1, 2))

        with torch.no_grad():
            output = self(x)
            predict = output.argmax(dim = 1, keepdim = True)
            accuracy = round(accuracy_score(predict, y), 4)

        return accuracy

    def count_nonzero_weights(self):
        counter = 0
        for params in list(self.parameters()):
            counter += torch.count_nonzero(params).item()
        return counter

    def count_total_weights(self):
        counter = 0
        for params in list(self.parameters()):
            counter += torch.numel(params)
        return counter

    @classmethod
    def load_model(cls):
        model = PtLeNetModel()
        path = f'{CUR_DIR}/../data/lenet.pth'
        model.load_state_dict(torch.load(path))
        return model
