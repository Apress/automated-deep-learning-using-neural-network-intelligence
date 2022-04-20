import torch
import torch.nn.functional as F

import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


@model_wrapper
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            nn.Identity
        ], label = 'conv_layer')
        self.dropout1 = nn.Dropout(
            nn.ValueChoice([0.25, 0.5, 0.75]),
            label = 'dropout'
        )
        self.dropout2 = nn.Dropout(0.5)
        feature = nn.ValueChoice(
            [64, 128, 256],
            label = 'hidden_size'
        )
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim = 1)
        return output
