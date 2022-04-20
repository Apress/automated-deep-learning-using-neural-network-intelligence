from typing import Tuple, Iterator
from torch.nn import Parameter
import nni.retiarii.nn.pytorch as nn


class FullyConnectedBlock(nn.Module):

    def __init__(
            self,
            dim: Tuple[int, int],
            dropout_rate,
            activation,
            block_num
    ) -> None:

        """
        Params:
            dim[0] - linear layer input
            dim[1] - linear layer output
            activation - activation layer
            block_num - ordinal number in fully connected sequence
        """
        super().__init__()

        self.input_dim = dim[0]
        self.output_dim = dim[1]

        self._linear = None
        self.dropout = nn.Dropout(p = dropout_rate)
        self.activation = activation

        self.switch = nn.InputChoice(
            n_candidates = 2,
            n_chosen = 1,
            label = f'fc_switch_{block_num}'
        )

    # Assuring lazy layers are initialized
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self._linear is None:
            raise Exception('linear layer is not initialized')
        return super().parameters(recurse)

    # Lazy Layer Initialization
    @property
    def linear(self):
        if self._linear is None:
            self._linear = nn.Linear(
                self.input_dim,
                self.output_dim
            )
        return self._linear

    def forward(self, x):

        if not self.input_dim:
            self.input_dim = x.shape[1]

        # Branch A
        a = self.linear(x)
        a = self.dropout(a)
        a = self.activation(a)

        # Branch B
        b = self.linear(x)
        b = self.activation(b)

        return self.switch([a, b])

    @classmethod
    def create(cls, activation, units, dropout_rate):

        def create_block(i):
            return FullyConnectedBlock(
                dim = (units[i], units[i + 1]),
                dropout_rate = dropout_rate,
                activation = activation,
                block_num = i
            )

        return create_block
