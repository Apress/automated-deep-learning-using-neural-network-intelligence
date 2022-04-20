import os
from random import random
import nni

from torch.utils.tensorboard import SummaryWriter

log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
writer = SummaryWriter(log_dir)

if __name__ == '__main__':
    p = nni.get_next_parameter()

    for i in range(100):
        # Dummy metrics
        acc = min((i + random() * 10) / 100, 1)
        loss = max((100 - i + random() * 10) / 100, 0)

        # Writing metrics to TensorBoard
        writer.add_scalar('Accuracy', acc, i)
        writer.add_scalar('Loss', loss, i)

        nni.report_intermediate_result(acc)

    nni.report_final_result(acc)
