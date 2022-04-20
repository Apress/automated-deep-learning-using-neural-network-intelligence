import torch.nn as nn


# Helper method to create convolution layer
def create_conv(kernel, in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels, kernel,
        padding = int((kernel - 1) / 2)
    )
