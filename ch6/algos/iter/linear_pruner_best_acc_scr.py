import os
import torch
from nni.algorithms.compression.v2.pytorch.pruning import LinearPruner
from ch6.algos.utils import model_comparison
from ch6.datasets import mnist_dataset
from ch6.model.pt_lenet import PtLeNetModel

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

original = PtLeNetModel.load_model()

config_list = [
    {'sparsity': 0.85, 'op_types': ['Conv2d']},
    {'sparsity': 0.4, 'op_types': ['Linear']},
    {'op_names': ['fc3'], 'exclude': True}  # excluding final layer
]


def evaluator(m: PtLeNetModel):
    if m.count_total_weights() > 30_000:
        return 0
    return m.test_model()


pruner = LinearPruner(
    model = original,
    config_list = config_list,
    pruning_algorithm = 'l1',
    total_iteration = 10,
    finetuner = lambda m: m.train_model(epochs = 1),
    evaluator = evaluator,
    speedup = True,
    dummy_input = torch.rand(10, 1, 28, 28),
    log_dir = CUR_DIR  # logging results (model.pth and mask.pth is there)
)

# Compressing loop
pruner.compress()

_, compressed, masks, best_acc, best_sparsity =\
    pruner.get_best_result()


# Best model saved in CUR_DIR/<date/best_result
print(f'Best Model is saved in: {CUR_DIR}')

print('===========')
print(f'Best accuracy: {best_acc}')
print('Best Sparsity:')
for layer in best_sparsity:
    print(f'{layer}')

# Displaying comparison
train_ds, test_ds = mnist_dataset()
model_comparison(original, compressed, test_ds, (1, 28, 28))
