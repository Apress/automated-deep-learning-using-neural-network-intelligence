from nni.retiarii.nn.pytorch import LayerChoice, InputChoice

from typing import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


# Creates Convolution Layer
def create_conv(kernel, in_ch, out_ch):
    return nn.Conv2d(
        in_channels = in_ch,
        out_channels = out_ch,
        kernel_size = kernel,
        padding = int((kernel - 1) / 2)
    )


class PtLeNetSupernet(nn.Module):

    def __init__(self, input_ts = 32):
        super(PtLeNetSupernet, self).__init__()

        self.conv1 = LayerChoice(OrderedDict(
            [
                ('conv1x1->16', create_conv(1, 1, 16)),  # 0
                ('conv3x3->16', create_conv(3, 1, 16)),  # 1
                ('conv5x5->16', create_conv(5, 1, 16)),  # 2
            ]
        ), label = 'conv1')

        self.conv2 = LayerChoice(OrderedDict(
            [
                ('conv1x1->32', create_conv(1, 16, 32)),  # 0
                ('conv3x3->32', create_conv(3, 16, 32)),  # 1
                ('conv5x5->32', create_conv(5, 16, 32)),  # 2
            ]
        ), label = 'conv2')

        self.act = nn.ReLU()
        self.flat = nn.Flatten()

        # Choosing Decision Making Layers
        self.dm = InputChoice(n_candidates = 2, n_chosen = 1, label = 'dm')

        self.fc11 = nn.Linear(input_ts * 8 * 8, 256)
        self.fc12 = nn.Linear(256, 10)

        self.fc2 = nn.Linear(input_ts * 8 * 8, 10)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # x.shape = (16, 16, 16)
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # x.shape = (32, 8, 8)

        x = self.flat(x)

        # branch 1
        x1 = self.act(self.fc11(x))
        x1 = self.act(self.fc12(x1))

        # branch 2
        x2 = self.act(self.fc2(x))

        # Choosing one of the branches
        x = self.dm([
            x1,  # 0
            x2  # 1
        ])

        return F.log_softmax(x, dim = 1)
