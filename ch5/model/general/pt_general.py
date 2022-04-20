from typing import OrderedDict
import torch.nn as nn
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from ch5.model.general.pt_ops import ConvBranch, PoolBranch, FactorizedReduce


class Cell(nn.Module):
    """
    General Cell
    """

    def __init__(self, cell_ord, input_num, in_f, out_f):
        super().__init__()

        # Block Operation
        self.block_op = LayerChoice(OrderedDict([
            ('SepConvBranch(3)', ConvBranch(in_f, out_f, 3, 1, 1, False)),
            ('NonSepConvBranch(3)', ConvBranch(in_f, out_f, 3, 1, 1, True)),
            ('SepConvBranch(5)', ConvBranch(in_f, out_f, 5, 1, 2, False)),
            ('NonSepConvBranch(3)', ConvBranch(in_f, out_f, 5, 1, 2, True)),
            ('AvgPoolBranch', PoolBranch('avg', in_f, out_f, 3, 1, 1)),
            ('MaxPoolBranch', PoolBranch('max', in_f, out_f, 3, 1, 1))
        ]), label = f'op_{cell_ord}')

        # Input connections
        if input_num > 0:
            self.connections = InputChoice(
                n_candidates = input_num, n_chosen = None,
                label = f'con_{cell_ord}'
            )
        else:
            self.connections = None

        # Batch Normalization
        self.batch_norm = nn.BatchNorm2d(out_f, affine = False)

    def forward(self, inputs):
        out = self.block_op(inputs[-1])

        # Input connections
        if self.connections is not None:
            connection = self.connections(inputs[:-1])
            if connection is not None:
                out = out + connection

        return self.batch_norm(out)


class GeneralSupernet(nn.Module):

    def __init__(
            self,
            num_cells = 6,
            out_f = 24,
            in_channels = 3,
            num_classes = 10
    ):
        super().__init__()
        self.num_cells = num_cells

        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_f, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_f)
        )

        # num_cells = 6 -> pool_layers_idx = [3, 6]
        self.pool_layers_idx = [
            cell_id
            for cell_id in range(1, num_cells + 1) if cell_id % 3 == 0
        ]

        # Cells
        self.cells = nn.ModuleList()

        # Pool Layers
        self.pool_layers = nn.ModuleList()

        # Initializing Cells and Pool Layers
        for cell_ord in range(num_cells):
            if cell_ord in self.pool_layers_idx:
                pool_layer = FactorizedReduce(out_f, out_f)
                self.pool_layers.append(pool_layer)
            cell = Cell(cell_ord, cell_ord, out_f, out_f)
            self.cells.append(cell)

        # Final Layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(out_f, num_classes)

    def forward(self, x):
        bs = x.size(0)
        cur = self.stem(x)

        # Constructing Calculation Graph
        cells = [cur]
        for cell_id in range(self.num_cells):
            cur = self.cells[cell_id](cells)
            cells.append(cur)
            # If pool layer is added
            if cell_id in self.pool_layers_idx:
                # Number of Pool Layer
                # 0, 1, 2, ...
                pool_ord = self.pool_layers_idx.index(cell_id)
                # Adding Pool Layer to all input cells
                for i, cell in enumerate(cells):
                    cells[i] = self.pool_layers[pool_ord](cell)
                cur = cells[-1]

        cur = self.gap(cur).view(bs, -1)
        logits = self.dense(cur)
        return logits
