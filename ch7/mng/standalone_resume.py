# Loading Packages
from time import sleep
from nni.experiment import Experiment

# Experiment Configuration
experiment = Experiment('local')
experiment.resume('experiment_id', port)

while True:
    sleep(1)
    if experiment.get_status() == 'DONE':
        break
