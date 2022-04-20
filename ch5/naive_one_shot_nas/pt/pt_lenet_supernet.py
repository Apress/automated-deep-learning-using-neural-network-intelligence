import torch
import torch.nn as nn
import torch.nn.functional as F
from ch5.naive_one_shot_nas.pt.pt_ops import create_conv


class PtLeNetNaiveSupernet(nn.Module):

    def __init__(self):
        super(PtLeNetNaiveSupernet, self).__init__()

        self.conv1_1 = create_conv(1, 1, 16)
        self.conv1_3 = create_conv(3, 1, 16)
        self.conv1_5 = create_conv(5, 1, 16)

        self.conv2_1 = create_conv(1, 16, 32)
        self.conv2_3 = create_conv(3, 16, 32)
        self.conv2_5 = create_conv(5, 16, 32)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, mask = None):

        # Sum all in training mode
        if mask is None:
            mask = [[1, 1, 1], [1, 1, 1]]

        x = mask[0][0] * self.conv1_1(x) +\
            mask[0][1] * self.conv1_3(x) +\
            mask[0][2] * self.conv1_5(x)

        x = torch.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = mask[1][0] * self.conv2_1(x) +\
            mask[1][1] * self.conv2_3(x) +\
            mask[1][2] * self.conv2_5(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.flat(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim = 1)
