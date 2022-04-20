import os
import nni
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from ch4.utils.datasets import cifar10_dataset


def early_stop_plateau(epoch_accuracy):
    """
    Early Stopping Algorithm
    """
    if len(epoch_accuracy) < 10:
        return False

    max_prev = max(epoch_accuracy[:-5])
    max_curr = max(epoch_accuracy)

    if max_prev * 1.01 > max_curr:
        return True

    return False


def evaluate(model_cls):
    learning_rate = 0.001
    batch_size = 16
    train_epochs = 50

    model = model_cls()
    # Preparing Train Dataset
    (x, y), (x_test, y_test) = cifar10_dataset()
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()

    # Permute dimensions for PyTorch Convolutions
    x = torch.permute(x, (0, 3, 1, 2))
    dataset_size = x.shape[0]

    #Initializing model
    model(x[0:1])

    # visualizing
    onnx_dir = os.path.abspath(os.environ.get('NNI_OUTPUT_DIR', '.'))
    os.makedirs(onnx_dir, exist_ok = True)
    torch.onnx.export(model, x[0:1], onnx_dir + '/model.onnx')

    optimizer = optim.Adam(
        model.parameters(),
        lr = learning_rate
    )

    epoch_accuracy = []

    model.train()
    for epoch in range(1, train_epochs + 1):
        # Random permutations for batch training
        permutation = torch.randperm(dataset_size)
        for bi in range(1, dataset_size, batch_size):
            # Creating New Batch
            indices = permutation[bi:bi + batch_size]
            batch_x, batch_y = x[indices], y[indices]

            # Model Parameters Optimization
            optimizer.zero_grad()
            output = model(batch_x)
            error = F.cross_entropy(output, batch_y)
            error.backward()
            optimizer.step()

        # Epoch Intermediate metrics:
        output = model(x)
        predict = output.argmax(dim = 1, keepdim = True)
        accuracy = round(accuracy_score(predict, y), 4)
        epoch_accuracy.append(accuracy)
        print(F'Epoch: {epoch}| Accuracy: {accuracy}')
        # report intermediate result
        nni.report_intermediate_result(accuracy)

        if early_stop_plateau(epoch_accuracy):
            print(f'Early Stopping Training')
            break

    model.eval()
    # Preparing Test Dataset
    x = torch.from_numpy(x_test).float()
    y = torch.from_numpy(y_test).long()
    x = torch.permute(x, (0, 3, 1, 2))

    with torch.no_grad():
        output = model(x)
        predict = output.argmax(dim = 1, keepdim = True)
        accuracy = round(accuracy_score(predict, y), 4)

    nni.report_final_result(accuracy)
