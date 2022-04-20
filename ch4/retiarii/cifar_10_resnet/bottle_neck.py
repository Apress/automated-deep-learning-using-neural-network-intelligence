import nni.retiarii.nn.pytorch as nn


class Bottleneck(nn.Module):
    """
    BottleNeck Residual Block
    """

    # BottleNeck expansion ratio
    expansion = 4

    @classmethod
    def result_channels_num(cls, channels):
        return channels * cls.expansion

    def __init__(
            self,
            cell_num,
            in_channels,
            out_channels,
            i_downsample = None,
            stride = 1
    ):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size = 1, stride = 1, padding = 0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size = 3, stride = stride, padding = 1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, self.result_channels_num(out_channels),
            kernel_size = 1, stride = 1, padding = 0
        )
        self.batch_norm3 = nn.BatchNorm2d(self.result_channels_num(out_channels))

        # Skip connection acts the same for all blocks in Residual Cell
        self.skip_connection = nn.InputChoice(
            n_candidates = 2,
            n_chosen = 1,
            label = f'bottle_neck_{cell_num}_skip_connection'
        )

        # identity downsample
        self.i_downsample = i_downsample

        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):

        # x0
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        identity = self.skip_connection([identity, None])

        # skipping connection
        if identity is not None:
            #downsample if needed
            if self.i_downsample is not None:
                identity = self.i_downsample(identity)
            # adding identity
            x += identity

        x = self.relu(x)

        return x
