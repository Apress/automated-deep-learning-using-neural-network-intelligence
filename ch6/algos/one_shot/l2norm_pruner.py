from nni.algorithms.compression.v2.pytorch.pruning import L2NormPruner
from ch6.algos.utils import oneshot_prune, model_comparison, visualize_mask
from ch6.datasets import mnist_dataset
from ch6.model.pt_lenet import PtLeNetModel

# Loading trained model
original = PtLeNetModel.load_model()
train_ds, test_ds = mnist_dataset()

# Compressing
compressed, masks = oneshot_prune(
    original,
    L2NormPruner,
    [{
        'sparsity': .7,
        'op_types': ['Conv2d'],
    }],
    train_ds
)

# Visualizing mask
visualize_mask(masks)

# Displaying comparison
model_comparison(original, compressed, test_ds, (1, 28, 28))
