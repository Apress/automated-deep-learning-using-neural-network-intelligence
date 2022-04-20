import os
import sys
import nni

# For NNI use relative import for user-defined modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../..'
sys.path.append(SCRIPT_DIR)

from ch3.bbf.f1 import black_box_f1

if __name__ == '__main__':
    # parameter from the search space selected by tuner
    p = nni.get_next_parameter()
    x, y = p['x'], p['y']
    r = black_box_f1(x, y)
    # returning result to NNI
    nni.report_final_result(r)
