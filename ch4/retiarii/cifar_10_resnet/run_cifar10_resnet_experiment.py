from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
from ch4.retiarii.cifar_10_resnet.eval import evaluate
from ch4.retiarii.cifar_10_resnet.res_net_model_space import ResNetModelSpace

# Model Space
model_space = ResNetModelSpace()

# Evaluator
evaluator = FunctionalEvaluator(evaluate)

# Search Strategy
search_strategy = strategy.PolicyBasedRL(
    trial_per_collect = 10,
    max_collect = 20
)

# Experiment
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)

# Experiment Configuration
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'CIFAR10_ResNet_NAS'
exp_config.trial_concurrency = 1
exp_config.max_trial_number = 100
exp_config.training_service.use_active_gpu = False
export_formatter = 'code'

# Launching Experiment
exp.run(exp_config, 8080)

# Returning Results
print('Final model:')
for model_code in exp.export_top_models(formatter = export_formatter):
    print(model_code)
