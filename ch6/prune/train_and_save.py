import os
import torch
from ch6.model.pt_lenet import PtLeNetModel

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

model = PtLeNetModel()
model.train_model(epochs = 20)
accuracy = model.test_model()
print(f'Model accuracy: {accuracy}')
torch.save(model.state_dict(), f'{CUR_DIR}/../data/lenet.pth')
