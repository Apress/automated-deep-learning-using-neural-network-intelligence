from typing import Iterator
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from torch.nn import Parameter
from ch4.retiarii.cifar_10_resnet.bottle_neck import Bottleneck


@model_wrapper
class ResNetModelSpace(nn.Module):
    """
    ResNet NAS Model Space
    """

    # classification classes
    num_classes = 10

    # input channels for RGB image
    in_channels = 3

    # ResNet Channel constant
    channels = 64

    def __init__(self):
        super().__init__()

        # Activation
        self.relu = nn.ReLU()

        # === First Convolution Block ===
        self.conv1 = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.channels,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            bias = False
        )

        self.batch_norm1 = nn.BatchNorm2d(64)

        pool_size = nn.ValueChoice([2, 3], label = 'pool_size')
        self.max_pool = nn.MaxPool2d(kernel_size = pool_size, stride = 2, padding = 1)

        # === Residual Cells Sequence ===
        self.res_cells = nn.Repeat(
            ResNetModelSpace.residual_cell_builder(),
            depth = (2, 5), label = 'res_cells_repeat'
        )

        # === Fully Connected Sequence ===
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1_input_dim = None
        self.fc1_output_dim = nn.ValueChoice(
            [256, 512],
            label = 'fc1_output_dim'
        )
        self._fc1 = None
        self.fc2 = nn.Linear(
            in_features = self.fc1_output_dim,
            out_features = self.num_classes
        )

    # Assuring lazy layers are initialized
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self._fc1 is None:
            raise Exception('fc1 layer is not initialized')
        return super().parameters(recurse)

    # Lazy Layer Initialization
    @property
    def fc1(self):
        if self._fc1 is None:
            self._fc1 = nn.Linear(
                self.fc1_input_dim,
                self.fc1_output_dim
            )
        return self._fc1

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.res_cells(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)

        if not self.fc1_input_dim:
            self.fc1_input_dim = x.shape[1]

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @classmethod
    def residual_cell_builder(cls):

        def create_cell(cell_num):
            # planes sequence: 64, 128, 256, 512,...
            planes = 64 * pow(2, cell_num)
            # stride sequence: 1, 2, 2, 2,...
            stride = max(1 + cell_num, 2)
            # block sequence: 3, 4, 5, 5,...
            blocks = max(3 + cell_num, 5)

            downsample = None
            layers = []

            if stride != 1 or cls.channels != Bottleneck.result_channels_num(planes):
                # downsample expands 'out_channels' to planes * Bottleneck.expansion
                # the same out dimension as Bottleneck produces
                downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels = cls.channels,
                        out_channels = Bottleneck.result_channels_num(planes),
                        kernel_size = 1,
                        stride = stride
                    ),
                    nn.BatchNorm2d(
                        num_features = Bottleneck.result_channels_num(planes)
                    )
                )

            layers.append(
                Bottleneck(
                    cell_num = cell_num,
                    in_channels = cls.channels,
                    out_channels = planes,
                    i_downsample = downsample,
                    stride = stride
                )
            )

            cls.channels = Bottleneck.result_channels_num(planes)

            for i in range(blocks - 1):
                layers.append(
                    Bottleneck(
                        cell_num = cell_num,
                        in_channels = cls.channels,
                        out_channels = planes
                    )
                )

            return nn.Sequential(*layers)

        return create_cell
