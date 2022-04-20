import os
import sys
from time import sleep
import nni

# For NNI use relative import for user-defined modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../..'
sys.path.append(SCRIPT_DIR)

from ch3.early_stop.medianstop.model import identity_with_parabolic_training

if __name__ == '__main__':
    params = nni.get_next_parameter()
    x = params['x']

    final, history = identity_with_parabolic_training(x)
    for h in history:
        sleep(.1)
        nni.report_intermediate_result(h)

    nni.report_final_result(final)
