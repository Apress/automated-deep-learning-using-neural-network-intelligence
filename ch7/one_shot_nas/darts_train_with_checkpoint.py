import pickle
import os
from os.path import exists
import torch
import torch.nn as nn
import ch7.datasets as datasets
from nni.retiarii.oneshot.pytorch import DartsTrainer
from ch7.one_shot_nas.pt_lenet import PtLeNetSupernet
from ch7.one_shot_nas.pt_utils import accuracy

cd = os.path.dirname(os.path.abspath(__file__))
trainer_checkpoint_path = f'{cd}/darts_trainer_checkpoint.bin'


def get_darts_trainer():
    # Supernet
    model = PtLeNetSupernet()

    # Dataset
    dataset_train, dataset_valid = datasets.get_dataset("mnist")

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optim = torch.optim.SGD(
        model.parameters(), 0.025,
        momentum = 0.9, weight_decay = 3.0E-4
    )

    # Trainer params
    num_epochs = 0
    batch_size = 256
    metrics = accuracy

    # DARTS Trainer
    darts_trainer = DartsTrainer(
        model = model,
        loss = criterion,
        metrics = metrics,
        optimizer = optim,
        num_epochs = num_epochs,
        dataset = dataset_train,
        batch_size = batch_size,
        log_frequency = 10,
        unrolled = False
    )

    return darts_trainer


def train_and_dump(darts_trainer, epochs):
    """
    Trains Supernet according to DARTS algorithm
    """
    darts_trainer.num_epochs = epochs
    darts_trainer.fit()

    with open(trainer_checkpoint_path, 'wb') as f:
        pickle.dump(darts_trainer, f)

    return darts_trainer


if __name__ == '__main__':

    if exists(trainer_checkpoint_path):
        with open(trainer_checkpoint_path, 'rb') as f:
            trainer = pickle.load(f)
    else:
        trainer = get_darts_trainer()

    for _ in range(10):
        trainer = train_and_dump(trainer, epochs = 5)

    print(f'Best model: {trainer.export()}')
