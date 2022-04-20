# Loading Packages
from pathlib import Path
from nni.experiment import Experiment
from ch3.bbf.holder import holder_function
from ch3.bbf.utils import scatter_plot

# Defining Search Space
search_space = {
    "x": {"_type": "uniform", "_value": [-10, 10]},
    "y": {"_type": "uniform", "_value": [-10, 10]}
}

# Experiment Configuration
experiment = Experiment('local')
experiment.config.experiment_name = 'GP Tuner'
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 100

experiment.config.search_space = search_space
experiment.config.trial_command = 'python3 trial.py'

experiment.config.trial_code_directory = Path(__file__).parent

experiment.config.tuner.name = 'GPTuner'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

# Starting NNI
http_port = 8080
experiment.start(http_port)

# Event Loop
while True:
    if experiment.get_status() == 'DONE':
        search_data = experiment.export_data()

        # Experiment Trial Parameters
        trial_params = [trial.parameter for trial in search_data]

        # Visualizing Trial Parameters
        scatter_plot(
            holder_function, [-10, 10], [-10, 10],
            trial_params, title = f'GP. Holder\'s Black-Box Function.'
        )

        search_metrics = experiment.get_job_metrics()
        input("Experiment is finished. Press any key to exit...")
        break
