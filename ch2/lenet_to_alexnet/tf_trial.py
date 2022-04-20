import os
import sys
import nni

# We use relative import for user-defined modules
# For NNI use relative import for user-defined modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append(SCRIPT_DIR)

from ch2.lenet_to_alexnet.tf_lenet_evolution import TfLeNetEvolution


def trial(hparams):
    """
    Trial Script:
      - Initiate Model
      - Train
      - Test
      - Report
    """

    # Converting Feature Extraction Sequences
    feat_ext_sequences = []
    for k, v in hparams.items():
        if k.startswith('fe_slot_'):
            v['type'] = v['_name']
            feat_ext_sequences.append(v)

    model = TfLeNetEvolution(
        feat_ext_sequences = feat_ext_sequences,
        l1_size = hparams['l1_size'],
        l2_size = hparams['l2_size'],
        dropout_rate = hparams['dropout_rate']
    )
    model.train(
        batch_size = 16,
        learning_rate = hparams['learning_rate'],
        epochs = 50
    )
    accuracy = model.test()

    # send final accuracy to NNI
    nni.report_final_result(accuracy)


if __name__ == '__main__':

    # Manual HyperParameters
    hparams = {
        'fe_slot_1':     {
            '_name':   'simple',
            'filters': 16,
            'kernel':  7
        },
        'fe_slot_2':     {
            '_name':     'with_pool',
            'filters':   8,
            'kernel':    5,
            'pool_size': 5
        },
        'fe_slot_3':     {
            '_name':     'with_pool',
            'filters':   8,
            'kernel':    5,
            'pool_size': 3
        },
        'l1_size':       1024,
        'l2_size':       512,
        'dropout_rate':  .3,
        'learning_rate': 0.001
    }

    # NNI HyperParameters
    # Run safely without NNI Experiment Context
    nni_hparams = nni.get_next_parameter()
    hparams.update(nni_hparams)

    trial(hparams)
