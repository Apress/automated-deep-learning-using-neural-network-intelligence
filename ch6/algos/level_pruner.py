from ch6.algos.flow import prune_flow, model_comparison
from ch6.model.pt_lenet import PtLeNetModel
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from ch8.utils import mnist_dataset

# Loading trained model
original = PtLeNetModel.load_model()
train_ds, test_ds = mnist_dataset()

# Pruner
prune_config = [{
    'sparsity': .8,
    'op_types': ['default'],
}]
pruner_cls = LevelPruner

# Compressing
compressed = prune_flow(original, pruner_cls, prune_config, train_ds, epochs = 1)

# Displaying comparison
model_comparison(original, compressed, test_ds, (1, 28, 28))
