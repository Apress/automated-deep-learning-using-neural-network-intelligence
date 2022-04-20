import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Initializing SummaryWriter
cd = os.path.dirname(os.path.abspath(__file__))
dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
tb_summary = SummaryWriter(f'{cd}/runs/{dt}')
iter_counter = 0


def accuracy(output, target, topk = (1,)):
    """
    Method calculating Darts Supernet Accuracy
    """
    global iter_counter

    # Computes accuracy for PyTorch Supernet
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accuracy = correct_k.mul_(1.0 / batch_size).item()

        # Passing accuracy to TensorBoard
        tb_summary.add_scalar('darts_lenet', accuracy, iter_counter)
        iter_counter += 1

        res["acc{}".format(k)] = accuracy

    return res
