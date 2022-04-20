import os
import torch

import nni
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


class ProdBlock(nn.Module):

    def __init__(self, multiplier = 0):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x):
        x = x * self.multiplier
        return x


@model_wrapper
class InputChoiceModelSpace(nn.Module):

    def __init__(self):
        super().__init__()

        # x 2 multiplier
        self.x2 = ProdBlock(2)
        # x 3 multiplier
        self.x3 = ProdBlock(3)
        # x 4 multiplier
        self.x4 = ProdBlock(4)

        self.mix = nn.InputChoice(
            n_candidates = 3,
            n_chosen = 2,
            reduction = 'sum'
        )

    def forward(self, x):
        # Branch A
        a = self.x2(x)

        # Branch B
        b = self.x2(x)
        b = self.x3(b)

        # Branch C
        c = self.x2(x)
        c = self.x3(c)
        c = self.x4(c)

        return self.mix([a, b, c])


def evaluate(model_cls):
    model = model_cls()
    x = 1
    out = model(x)

    # visualizing
    onnx_dir = os.path.abspath(os.environ.get('NNI_OUTPUT_DIR', '.'))
    os.makedirs(onnx_dir, exist_ok = True)
    torch.onnx.export(model, x, onnx_dir + '/model.onnx')

    nni.report_final_result(out)
