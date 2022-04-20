from ch6.algos.flow import prune_flow, model_comparison
from ch6.model.pt_lenet import PtLeNetModel
from nni.algorithms.compression.pytorch.pruning import L2FilterPruner
from ch8.utils import mnist_dataset

# Loading trained model
original = PtLeNetModel.load_model()
train_ds, test_ds = mnist_dataset()

# Compressing
compressed = prune_flow(
    original,
    L2FilterPruner,
    [{
        'sparsity': .8,
        'op_types': ['Conv2d'],
    }],
    train_ds
)

# Displaying comparison
model_comparison(original, compressed, test_ds, (1, 28, 28))
