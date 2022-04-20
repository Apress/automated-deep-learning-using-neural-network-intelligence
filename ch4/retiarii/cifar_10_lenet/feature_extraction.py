from typing import Tuple
import nni.retiarii.nn.pytorch as nn


class FeatureExtractionBlock(nn.Module):

    def __init__(
            self,
            dim: Tuple[int, int],
            kernel_size,
            activation,
            block_num = 0
    ) -> None:
        """
        Parameters:
             dim[0] - input dimension
             dim[1] - output dimension
             kernel_size - convolution kernel ValueChoice
             activation - activation layer
        """

        super().__init__()

        self.input_dim = dim[0]
        self.output_dim = dim[1]

        self.conv = nn.Conv2d(
            in_channels = self.input_dim,
            out_channels = self.output_dim,
            kernel_size = kernel_size
        )
        self.max_pool = nn.MaxPool2d(2, 2)
        self.activation = activation

        self.switch = nn.InputChoice(
            n_candidates = 2,
            n_chosen = 1,
            label = f'fe_switch_{block_num}'
        )

    def forward(self, x):

        x = self.conv(x)

        # Branch A
        a = self.max_pool(x)
        a = self.activation(a)

        # Branch B
        b = self.activation(x)

        return self.switch([a, b])

    @classmethod
    def create(cls, activation, in_dimension):
        """
        Constructs sequence of blocks with nn.Repeat
        """

        def create_block(i):
            params = {
                'kernel_size': nn.ValueChoice(
                    [3, 5],
                    label = f'fe_kernel_size_{i}'
                )
            }

            # Feature Block Dimensions
            if i == 0:
                dim = (in_dimension, 8)
            elif i == 1:
                dim = (8, 16)
            else:
                dim = (16, 16)

            params['dim'] = dim
            params['activation'] = activation

            return FeatureExtractionBlock(**params)

        return create_block
