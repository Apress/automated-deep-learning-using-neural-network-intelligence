# Loading Packages
from nni.experiment import Experiment, RemoteConfig, RemoteMachineConfig
from pathlib import Path

# Remote Training Service Params:
nni_host_ip = '10.10.120.20'
remote_ip = '10.10.120.21'
remote_ssh_user = 'nni_user'
remote_ssh_pass = 'nni_pass'
remote_python_path = '/opt/python3/bin'

# Defining Search Space
search_space = {
    "x": {"_type": "quniform", "_value": [1, 100, .1]}
}

# Experiment Configuration
experiment = Experiment('remote')
experiment.config.experiment_name = 'Remote Experiment'
experiment.config.trial_concurrency = 4
experiment.config.trial_command = 'python3 trial.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.max_trial_number = 1000
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'

experiment.config.nni_manager_ip = nni_host_ip
remote_service = RemoteConfig()
remote_machine = RemoteMachineConfig()
remote_machine.host = remote_ip
remote_machine.user = remote_ssh_user
remote_machine.password = remote_ssh_pass
remote_machine.python_path = remote_python_path
remote_service.machine_list = [remote_machine]
experiment.config.training_service = remote_service

# Starting NNI
http_port = 8080
experiment.start(http_port)

# Event Loop
while True:
    if experiment.get_status() == 'DONE':
        break
