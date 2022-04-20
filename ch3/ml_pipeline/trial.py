import os
import sys
import nni

# We use relative import for user-defined modules
# For NNI use relative import for user-defined modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append(SCRIPT_DIR)

from ch3.ml_pipeline.model import MlPipelineClassifier
from ch3.ml_pipeline.utils import telescope_dataset


def trial(hparams):
    #Initializing model
    model = MlPipelineClassifier(hparams)

    # Preparing dataset
    X_train, y_train, X_test, y_test = telescope_dataset()

    model.train(X_train, y_train)

    # Calculating `score` on test dataset
    score = model.score(X_test, y_test)

    # Send final score to NNI
    nni.report_final_result(score)


if __name__ == '__main__':

    # Manual HyperParameters
    hparams = {
        'op_1': {
            '_name':      'SelectPercentile',
            'percentile': 2
        },
        'op_2': {
            '_name': 'none'
        },
        'op_3': {
            '_name': 'Normalizer',
            'norm':  'l1'
        },
        'op_4': {
            '_name':          'PCA',
            'svd_solver':     'randomized',
            'iterated_power': 3
        },
        'op_5': {
            '_name':     'DecisionTreeClassifier',
            'criterion': "entropy",
            'max_depth': 8
        }
    }

    # NNI HyperParameters
    # Run safely without NNI Experiment Context
    nni_hparams = nni.get_next_parameter()
    hparams.update(nni_hparams)

    trial(hparams)
