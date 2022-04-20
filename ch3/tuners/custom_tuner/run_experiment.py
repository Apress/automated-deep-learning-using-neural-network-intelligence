# Loading Packages
from pathlib import Path
from nni.experiment import Experiment, CustomAlgorithmConfig
from ch3.bbf.ackley import ackley_function
from ch3.bbf.utils import scatter_plot

# Defining Search Space
search_space = {
    "x": {"_type": "quniform", "_value": [-10, 10, .01]},
    "y": {"_type": "quniform", "_value": [-10, 10, .01]}
}

# Experiment Configuration
experiment = Experiment('local')
experiment.config.experiment_name = 'New Evolution Tuner'
experiment.config.trial_concurrency = 4

# Evolution for 8 generations
experiment.config.max_trial_number = 100

experiment.config.search_space = search_space
experiment.config.trial_command = 'python3 trial.py'
experiment.config.trial_code_directory = Path(__file__).parent

experiment.config.tuner = CustomAlgorithmConfig()
experiment.config.tuner.code_directory = Path(__file__).parent
experiment.config.tuner.class_name = 'evolution_tuner.NewEvolutionTuner'
experiment.config.tuner.class_args = {'population_size': 8}

# Starting NNI
http_port = 8080
experiment.start(http_port)

# Event Loop
while True:
    if experiment.get_status() == 'DONE':
        search_data = experiment.export_data()

        # Experiment Trial Parameters
        trial_params = [trial.parameter for trial in search_data]
        scatter_plot(
            ackley_function, [-10, 10], [-10, 10],
            trial_params, title = 'New Evolution Tuner'
        )

        search_metrics = experiment.get_job_metrics()
        input("Experiment is finished. Press any key to exit...")
        break
