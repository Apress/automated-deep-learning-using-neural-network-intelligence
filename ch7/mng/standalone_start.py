# Loading Packages
from nni.experiment import Experiment
from pathlib import Path

# Defining Search Space
search_space = {
    "x": {"_type": "quniform", "_value": [1, 100, .1]}
}

# Experiment Configuration
experiment = Experiment('local')
experiment.config.experiment_name = 'Experiment'
experiment.config.trial_concurrency = 4
experiment.config.trial_command = 'python3 trial.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.max_trial_number = 100
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'

# Starting NNI
http_port = 8080
experiment.start(http_port)

# Event Loop
while True:
    if experiment.get_status() == 'DONE':
        break
