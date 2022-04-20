import torch
import torch.nn as nn
import torch.nn.functional as F
from ch5.naive_one_shot_nas.pt.pt_ops import create_conv


class PtLeNetMultiTrialModel(nn.Module):

    def __init__(self, kernel1, kernel2):
        super(PtLeNetMultiTrialModel, self).__init__()

        self.conv1 = create_conv(kernel1, in_channels = 1, out_channels = 16)
        self.conv2 = create_conv(kernel2, in_channels = 16, out_channels = 32)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.flat(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)
