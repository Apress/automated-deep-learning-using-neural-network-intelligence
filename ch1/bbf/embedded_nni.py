# Loading Packages
from pathlib import Path
from nni.experiment import Experiment

# Defining Search Space
search_space = {
    "x": {"_type": "quniform", "_value": [1, 100, 1]},
    "y": {"_type": "quniform", "_value": [1, 10, 1]},
    "z": {"_type": "quniform", "_value": [1, 10000, 0.01]}
}

# Experiment Configuration
experiment = Experiment('local')
experiment.config.experiment_name = 'Black Box Function Optimization'
experiment.config.trial_concurrency = 4
experiment.config.max_trial_number = 1000
experiment.config.search_space = search_space
experiment.config.trial_command = 'python3 trial.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

# Starting NNI
http_port = 8080
experiment.start(http_port)

# Event Loop
while True:
    if experiment.get_status() == 'DONE':
        search_data = experiment.export_data()
        search_metrics = experiment.get_job_metrics()
        input("Experiment is finished. Press any key to exit...")
        break
