# Loading Packages
from pathlib import Path
from nni.experiment import Experiment
from ch3.bbf.f1 import black_box_f1
from ch3.bbf.utils import scatter_plot

# Defining Search Space
search_space = {
    "x": {"_type": "quniform", "_value": [-10, 10, 1]},
    "y": {"_type": "quniform", "_value": [-10, 10, 1]}
}

# Experiment Configuration
experiment = Experiment('local')
experiment.config.experiment_name = 'Random Tuner'
experiment.config.trial_concurrency = 4
experiment.config.max_trial_number = 100
experiment.config.search_space = search_space
experiment.config.trial_command = 'python3 trial.py'
experiment.config.trial_code_directory = Path(__file__).parent

experiment.config.tuner.name = 'Random'

# Starting NNI
http_port = 8080
experiment.start(http_port)

# Event Loop
while True:
    if experiment.get_status() == 'DONE':
        search_data = experiment.export_data()
        trial_params = [trial.parameter for trial in search_data]

        # Visualizing Trial Parameters
        scatter_plot(
            black_box_f1, [-10, 10], [-10, 10],
            trial_params, title = 'Random Search'
        )

        search_metrics = experiment.get_job_metrics()
        input("Experiment is finished. Press any key to exit...")
        break
