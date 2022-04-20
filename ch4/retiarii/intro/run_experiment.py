from time import sleep
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
from ch4.retiarii.intro.dummy_model import DummyModel, evaluate

# Model Space
model_space = DummyModel()

# Evaluator
evaluator = FunctionalEvaluator(evaluate)

# Search Strategy
search_strategy = strategy.Random(dedup = True)

# Experiment
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)

# Experiment Configuration
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'dummy_search'
exp_config.trial_concurrency = 1
exp_config.max_trial_number = 100
exp_config.training_service.use_active_gpu = False
export_formatter = 'dict'

# Launching Experiment
exp.run(exp_config, 8080)

# Returning Results
while True:
    sleep(1)
    input("Experiment is finished. Press any key to exit...")
    print('Final model:')
    for model_code in exp.export_top_models(formatter = export_formatter):
        print(model_code)
    break
