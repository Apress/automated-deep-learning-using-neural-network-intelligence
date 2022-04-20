# pip install torchsummary
import os
import torch
from nni.compression.pytorch import ModelSpeedup
from torchsummary import summary
from ch6.model.pt_lenet import PtLeNetModel

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Loading Pruned Model
dummy_input = torch.randn((500, 1, 28, 28))
model_path = f'{CUR_DIR}/../data/lenet_pruned.pth'
mask_path = f'{CUR_DIR}/../data/mask.pth'
model_pruned = PtLeNetModel()
model_pruned.load_state_dict(torch.load(model_path))
speedup = ModelSpeedup(model_pruned, dummy_input, mask_path)
speedup.speedup_model()
model_pruned.eval()

# Testing Accuracy
acc = model_pruned.test_model()
print(acc)

# Loading Original Model
model_original = PtLeNetModel()
model_path = f'{CUR_DIR}/../data/lenet.pth'
model_original.load_state_dict(torch.load(model_path))

# Displaying summary of Original and Pruned models
print('==== ORIGINAL MODEL =====')
summary(model_original, (1, 28, 28))
print('=========================')

print('====  PRUNED MODEL  =====')
summary(model_pruned, (1, 28, 28))
print('=========================')
