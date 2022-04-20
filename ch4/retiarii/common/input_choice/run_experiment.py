from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
from ch4.retiarii.common.input_choice.model_space import InputChoiceModelSpace, evaluate

# Model Space
model_space = InputChoiceModelSpace()

# Evaluator
evaluator = FunctionalEvaluator(evaluate)

# Search Strategy
search_strategy = strategy.GridSearch()

# Experiment
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)

# Experiment Configuration
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'dummy_search'
exp_config.trial_concurrency = 1
exp_config.max_trial_number = 10
exp_config.training_service.use_active_gpu = False
export_formatter = 'dict'

# Launching Experiment
exp.run(exp_config, 8080)

# Returning Results
print('Final model:')
for model_code in exp.export_top_models(formatter = export_formatter):
    print(model_code)
