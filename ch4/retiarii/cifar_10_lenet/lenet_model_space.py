from typing import Iterator
from torch.nn import Parameter
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from ch4.retiarii.cifar_10_lenet.feature_extraction import FeatureExtractionBlock
from ch4.retiarii.cifar_10_lenet.fully_connected import FullyConnectedBlock


@model_wrapper
class Cifar10LeNetModelSpace(nn.Module):

    def __init__(self):

        super().__init__()
        # number of classes for CIFAR-10 dataset
        self.class_num = 10

        # RGB input channels
        self.input_channels = 3

        # === Feature Extraction ===

        # feature extraction activation function
        fe_activation = nn.LayerChoice(
            [nn.Sigmoid(), nn.ReLU()],
            label = f'fe_activation'
        )

        # Feature Extraction Block Stack
        self.fe = nn.Repeat(
            FeatureExtractionBlock.create(fe_activation, self.input_channels),
            depth = (2, 3), label = 'fe_repeat'
        )

        # === Decision Maker ===
        self.flat = nn.Flatten()

        # fully
        dm_activation = nn.LayerChoice(
            [nn.Sigmoid(), nn.ReLU()],
            label = f'fc_activation'
        )
        l1_size = nn.ValueChoice([256, 128], label = 'l1_size')
        l2_size = nn.ValueChoice([128, 64], label = 'l2_size')
        l3_size = nn.ValueChoice([64, 32], label = 'l3_size')
        dropout_rate = nn.ValueChoice([.3, .5], label = 'fc_dropout_rate')

        # Decision Maker Block Stack
        self.dm = nn.Repeat(
            FullyConnectedBlock.create(
                dm_activation,
                [None, l1_size, l2_size, l3_size],
                dropout_rate
            ),
            depth = (1, 3), label = 'fc_repeat'
        )

        self.linear_final_input_dim = None
        self._linear_final = None
        self.log_max = nn.LogSoftmax(dim = 1)

    # Assuring lazy layers are initialized
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self._linear_final is None:
            raise Exception('linear_final layer is not initialized')
        return super().parameters(recurse)

    # Lazy Layer Initialization
    @property
    def linear_final(self):
        if self._linear_final is None:
            self._linear_final = nn.Linear(
                self.linear_final_input_dim,
                self.class_num
            )
        return self._linear_final

    def forward(self, x):
        x = self.fe(x)
        x = self.flat(x)
        x = self.dm(x)

        if not self.linear_final_input_dim:
            self.linear_final_input_dim = x.shape[1]

        x = self.linear_final(x)
        return self.log_max(x)
