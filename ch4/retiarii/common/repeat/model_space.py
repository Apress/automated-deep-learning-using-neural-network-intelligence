import os
import torch

import nni
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


class AddBlock(nn.Module):

    def __init__(self, add = 0):
        super().__init__()
        self.add = add

    def forward(self, x):
        x = x + self.add
        return x

    @classmethod
    def create(cls, block_num):
        return AddBlock(block_num)


@model_wrapper
class RepeatModelSpace(nn.Module):

    def __init__(self):
        super().__init__()
        self.repeat = nn.Repeat(
            AddBlock.create,
            depth = (1, 5)
        )

    def forward(self, x):
        return self.repeat(x)


def evaluate(model_cls):
    model = model_cls()
    x = 1
    out = model(x)

    # visualizing
    onnx_dir = os.path.abspath(os.environ.get('NNI_OUTPUT_DIR', '.'))
    os.makedirs(onnx_dir, exist_ok = True)
    torch.onnx.export(model, x, onnx_dir + '/model.onnx')

    nni.report_final_result(out)
