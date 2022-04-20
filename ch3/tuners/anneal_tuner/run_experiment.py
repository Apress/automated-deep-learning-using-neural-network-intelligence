# Loading Packages
from pathlib import Path
from nni.experiment import Experiment
from ch3.bbf.holder import holder_function
from ch3.bbf.utils import scatter_plot

# Defining Search Space
search_space = {
    "x": {"_type": "quniform", "_value": [-10, 10, .01]},
    "y": {"_type": "quniform", "_value": [-10, 10, .01]}
}

# Experiment Configuration
experiment = Experiment('local')
experiment.config.experiment_name = 'Anneal Tuner'
experiment.config.trial_concurrency = 8
experiment.config.max_trial_number = 100

experiment.config.search_space = search_space
experiment.config.trial_command = 'python3 trial.py'

experiment.config.trial_code_directory = Path(__file__).parent

experiment.config.tuner.name = 'Anneal'
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

        # Trial Parameters split by chunks
        trial_params_chunks = [
            trial_params[i:i + 25]
            for i in range(0, len(trial_params), 25)
        ]

        for i, population in dict(enumerate(trial_params_chunks)).items():
            scatter_plot(
                holder_function, [-10, 10], [-10, 10],
                population, title = f'Anneal. Holder\'s Black-Box Function. Generation: {i+1}'
            )

        search_metrics = experiment.get_job_metrics()
        input("Experiment is finished. Press any key to exit...")
        break
