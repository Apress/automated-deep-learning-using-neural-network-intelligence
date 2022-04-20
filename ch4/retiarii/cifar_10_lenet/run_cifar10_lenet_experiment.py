from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
from ch4.retiarii.cifar_10_lenet.eval import evaluate
from ch4.retiarii.cifar_10_lenet.lenet_model_space import Cifar10LeNetModelSpace

# Model Space
model_space = Cifar10LeNetModelSpace()

# Evaluator
evaluator = FunctionalEvaluator(evaluate)

# Search Strategy
search_strategy = strategy.PolicyBasedRL(
    trial_per_collect = 10,
    max_collect = 200
)

# Experiment
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)

# Experiment Configuration
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'CIFAR10_LeNet_NAS'
exp_config.trial_concurrency = 1
exp_config.max_trial_number = 500
exp_config.training_service.use_active_gpu = False
export_formatter = 'dict'

# Launching Experiment
exp.run(exp_config, 8080)

# Returning Results
print('Final model:')
for model_code in exp.export_top_models(formatter = export_formatter):
    print(model_code)
