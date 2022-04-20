import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from ch5.datasets import mnist_dataset


def train_model(model):
    learning_rate = 0.001
    batch_size = 256
    # Preparing Train Dataset
    (x, y), _ = mnist_dataset()
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()

    # Permute dimensions for PyTorch Convolutions
    x = torch.permute(x, (0, 3, 1, 2))
    dataset_size = x.shape[0]

    optimizer = optim.Adam(
        model.parameters(),
        lr = learning_rate
    )

    model.train()
    for epoch in range(1, 10 + 1):
        # Random permutations for batch training
        permutation = torch.randperm(dataset_size)
        for bi in range(1, dataset_size, batch_size):
            # Creating New Batch
            indices = permutation[bi:bi + batch_size]
            batch_x, batch_y = x[indices], y[indices]

            # Model Parameters Optimization
            optimizer.zero_grad()
            output = model(batch_x)
            loss = F.cross_entropy(output, batch_y)
            loss.backward()
            optimizer.step()

        # Epoch Intermediate metrics:
        output = model(x)
        predict = output.argmax(dim = 1, keepdim = True)
        accuracy = round(accuracy_score(predict, y), 4)
        print(F'Epoch: {epoch}| Accuracy: {accuracy}')

    return accuracy


def test_model(model):
    model.eval()
    # Preparing Test Dataset
    _, (x, y) = mnist_dataset()
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x = torch.permute(x, (0, 3, 1, 2))

    with torch.no_grad():
        output = model(x)
        predict = output.argmax(dim = 1, keepdim = True)
        accuracy = round(accuracy_score(predict, y), 4)

    return accuracy
