import os
import torch
import nni
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


@model_wrapper
class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

        # operator 1
        self.op1 = nn.LayerChoice([
            nn.Tanh(),
            nn.Sigmoid(),
            nn.ReLU()
        ])

        # addition
        self.add = nn.ValueChoice([0, 1, 2])

        # operator 2
        self.op2 = nn.LayerChoice([
            nn.Tanh(),
            nn.Sigmoid(),
            nn.ReLU()
        ])

    def forward(self, x):
        x = self.op1(x)
        x = x * 2
        x += self.add
        x = x * 4
        x = self.op2(x)
        return x


def evaluate(model_cls):
    model = model_cls()
    x = torch.Tensor([1])
    y = model(x)

    # visualizing
    onnx_dir = os.path.abspath(os.environ.get('NNI_OUTPUT_DIR', '.'))
    os.makedirs(onnx_dir, exist_ok = True)
    torch.onnx.export(model, x, onnx_dir + '/model.onnx')

    nni.report_final_result(y.item())
