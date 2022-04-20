import os
from random import random
import nni

log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')

if __name__ == '__main__':
    p = nni.get_next_parameter()

    for i in range(100):
        acc = min((i + random() * 10) / 100, 1)
        loss = max((100 - i + random() * 10) / 100, 0)
        nni.report_intermediate_result(acc)

    nni.report_final_result(acc)
