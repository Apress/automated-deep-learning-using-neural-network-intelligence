from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D
from nni.nas.tensorflow.mutables import InputChoice, LayerChoice, MutableScope
from ch5.model.general.tf_ops import build_conv, build_separable_conv, build_avg_pool, build_max_pool, FactorizedReduce


class Cell(MutableScope):
    """
    General Cell
    """

    def __init__(self, cell_ord, input_num, filters):
        super().__init__(f'cell_{cell_ord}')

        # Block Operation
        self.block_op = LayerChoice([
            build_conv(filters, 3, 'conv3'),
            build_separable_conv(filters, 3, 'sepconv3'),
            build_conv(filters, 5, 'conv5'),
            build_separable_conv(filters, 5, 'sepconv5'),
            build_avg_pool(filters, 'avgpool'),
            build_max_pool(filters, 'maxpool'),
        ], key = f'op_{cell_ord}')

        # Input connections
        if input_num > 0:
            self.connections = InputChoice(
                n_candidates = input_num,
                n_chosen = None,
                key = f'con_{cell_ord}'
            )
        else:
            self.connections = None

        # Batch Normalization
        self.batch_norm = BatchNormalization(trainable = False)

    def call(self, inputs):
        out = self.block_op(inputs[-1])

        # Input connections
        if self.connections is not None:
            connection = self.connections(inputs[:-1])
            if connection is not None:
                out += connection

        return self.batch_norm(out)


class GeneralSupernet(Model):

    def __init__(
            self,
            num_cells = 6,
            filters = 24,
            num_classes = 10
    ):
        super().__init__()
        self.num_cells = num_cells

        self.stem = Sequential([
            Conv2D(filters, kernel_size = 3, padding = 'same', use_bias = False),
            BatchNormalization()
        ])

        # num_cells = 6 -> pool_layers_idx = [3, 6]
        self.pool_layers_idx = [
            cell_id
            for cell_id in range(1, num_cells + 1) if cell_id % 3 == 0
        ]

        # Cells
        self.cells = []

        # Pool Layers
        self.pool_layers = []

        # Initializing Cells and Pool Layers
        for cell_ord in range(num_cells):
            if cell_ord in self.pool_layers_idx:
                pool_layer = FactorizedReduce(filters)
                self.pool_layers.append(pool_layer)
            cell = Cell(cell_ord, cell_ord, filters)
            self.cells.append(cell)

        # Final Layers
        self.gap = GlobalAveragePooling2D()
        self.dense = Dense(num_classes)

    def call(self, x):
        cur = self.stem(x)
        prev_outputs = [cur]

        for cell_id, cell in enumerate(self.cells):
            if cell_id in self.pool_layers_idx:
                # Number of Pool Layer
                # 0, 1, 2, ....
                pool_ord = self.pool_layers_idx.index(cell_id)
                pool = self.pool_layers[pool_ord]
                prev_outputs = [pool(tensor) for tensor in prev_outputs]
                cur = prev_outputs[-1]

            cur = cell(prev_outputs)
            prev_outputs.append(cur)

        cur = self.gap(cur)
        logits = self.dense(cur)
        return logits
