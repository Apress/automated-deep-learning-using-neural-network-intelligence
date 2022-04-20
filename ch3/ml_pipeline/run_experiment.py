from pathlib import Path
from nni.experiment import Experiment, CustomAlgorithmConfig
from ch3.ml_pipeline.search_space import SearchSpace

# Experiment Configuration
experiment = Experiment('local')
experiment.config.experiment_name = 'AutoML Pipeline'
experiment.config.trial_concurrency = 4
experiment.config.max_trial_number = 500

# Generating Search Space
experiment.config.search_space = SearchSpace.build()

# Trial Configuration
experiment.config.trial_command = 'python3 trial.py'
experiment.config.trial_code_directory = Path(__file__).parent

# Tuner Configuration
experiment.config.tuner = CustomAlgorithmConfig()
experiment.config.tuner.code_directory = Path(__file__).parent
experiment.config.tuner.class_name = 'evolution_shrink_tuner.EvolutionShrinkTuner'
experiment.config.tuner.class_args = {
    'optimize_mode':   'maximize',
    'population_size': 64
}

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
