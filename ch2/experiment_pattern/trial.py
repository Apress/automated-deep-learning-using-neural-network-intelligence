import os
import sys
import nni

# For NNI use relative import for user-defined modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append(SCRIPT_DIR)

from ch2.experiment_pattern.model import DummyModel


def trial(hparams):
    """
    Trial Script:
      - Initiate Model
      - Train
      - Test
      - Report
    """
    model = DummyModel(**hparams)
    model.train()
    accuracy = model.test()

    # send final accuracy to NNI
    nni.report_final_result(accuracy)


if __name__ == '__main__':

    # Manual HyperParameters
    hparams = {
        'x': 1,
        'y': 1,
    }

    # NNI HyperParameters
    # Run safely without NNI Experiment Context
    nni_hparams = nni.get_next_parameter()
    hparams.update(nni_hparams)

    trial(hparams)
