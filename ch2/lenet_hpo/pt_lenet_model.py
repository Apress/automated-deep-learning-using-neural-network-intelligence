import numpy as np
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from ch2.utils.datasets import mnist_dataset


class PtLeNetModel(nn.Module):

    def __init__(self, filter_size, kernel_size, l1_size):
        super(PtLeNetModel, self).__init__()

        # Saving l1_size HyperParameter
        self.l1_size = l1_size

        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = filter_size,
            kernel_size = kernel_size
        )
        self.conv2 = nn.Conv2d(
            in_channels = filter_size,
            out_channels = filter_size * 2,
            kernel_size = kernel_size
        )

        # Lazy fc1 Layer Initialization
        self.fc1__in_features = 0
        self._fc1 = None
        self.fc2 = nn.Linear(l1_size, 10)

    # Lazy Layer Initialization
    @property
    def fc1(self):
        if self._fc1 is None:
            self._fc1 = nn.Linear(
                self.fc1__in_features,
                self.l1_size
            )
        return self._fc1

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.sigmoid(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        # Flatting all dimensions but batch-dimension
        if not self.fc1__in_features:
            self.fc1__in_features = np.prod(x.shape[1:])

        x = x.view(-1, self.fc1__in_features)

        # FC1 initializes lazy
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

    def train_model(self, learning_rate, batch_size):

        # Preparing Train Dataset
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
        for epoch in range(1, 10 + 1):
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
            print(F'Epoch: {epoch}| Accuracy: {accuracy}')
            # report intermediate result
            nni.report_intermediate_result(accuracy)

    def test_model(self):
        self.eval()
        # Preparing Test Dataset
        _, (x, y) = mnist_dataset()
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        x = torch.permute(x, (0, 3, 1, 2))

        with torch.no_grad():
            output = self(x)
            predict = output.argmax(dim = 1, keepdim = True)
            accuracy = round(accuracy_score(predict, y), 4)

        return accuracy
