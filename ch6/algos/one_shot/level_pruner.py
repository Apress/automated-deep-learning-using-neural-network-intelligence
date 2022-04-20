from nni.algorithms.compression.v2.pytorch.pruning import LevelPruner
from ch6.algos.utils import model_comparison, oneshot_prune, visualize_mask
from ch6.datasets import mnist_dataset
from ch6.model.pt_lenet import PtLeNetModel

# Loading trained model
original = PtLeNetModel.load_model()
train_ds, test_ds = mnist_dataset()

# Pruner
prune_config = [
    {
        'sparsity': .8,
        'op_types': ['Conv2d'],
    },
    {
        'sparsity': .6,
        'op_types': ['Linear'],
    },
    {
        'op_names': ['fc3'],
        'exclude':  True
    }
]
pruner_cls = LevelPruner

# Compressing
compressed, mask = oneshot_prune(
    original,
    pruner_cls,
    prune_config,
    train_ds,
    epochs = 10
)

# Visualizing mask
visualize_mask(mask)

# Displaying comparison
model_comparison(original, compressed, test_ds, (1, 28, 28))
