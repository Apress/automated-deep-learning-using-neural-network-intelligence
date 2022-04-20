from nni.algorithms.compression.v2.pytorch.pruning import FPGMPruner
from ch6.algos.utils import oneshot_prune, model_comparison, visualize_mask
from ch6.datasets import mnist_dataset
from ch6.model.pt_lenet import PtLeNetModel

# Loading trained model and datasets
original = PtLeNetModel.load_model()
train_ds, test_ds = mnist_dataset()

# Compressing
compressed, mask = oneshot_prune(
    original,
    FPGMPruner,
    [{
        'sparsity': .5,
        'op_types': ['Conv2d'],
    }],
    train_ds
)

# Visualizing mask
visualize_mask(mask)

# Displaying comparison
model_comparison(original, compressed, test_ds, (1, 28, 28))
