import nni
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from ch2.utils.datasets import hoh_dataset
from ch2.utils.pt_utils import SimpleDataset


class PtLeNetEvolution(pl.LightningModule):

    def __init__(
            self,
            feat_ext_sequences,
            l1_size,
            l2_size,
            dropout_rate,
            learning_rate
    ) -> None:
        """
         LeNet Evolution Model
        """

        super().__init__()

        self.lr = learning_rate
        self.dropout_rate = dropout_rate
        self.save_hyperparameters()

        fe_stack = []

        # ============
        # Constructing Feature Extraction Stack

        # Input size of next conv layer is out_channels of previous one
        in_dim = 3

        for fe_seq in feat_ext_sequences:
            if fe_seq['type'] in ['simple', 'with_pool']:
                fe_stack.append(
                    nn.Conv2d(
                        in_dim,
                        out_channels = fe_seq['filters'],
                        kernel_size = fe_seq['kernel'],
                        bias = False
                    )
                )
                if fe_seq['type'] == 'with_pool':
                    fe_stack.append(
                        nn.MaxPool2d(
                            kernel_size = fe_seq['pool_size']
                        )
                    )
                fe_stack.append(nn.ReLU())
                in_dim = fe_seq['filters']

        self.fe_stack = nn.Sequential(*fe_stack)

        # ============
        # Decision Maker Stack

        # Lazy fc1 Layer Initialization
        self.fc1__in_features = 0
        self._fc1 = None

        if l2_size > 0:
            self.fc2 = nn.Sequential(
                nn.Linear(l1_size, l2_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.fc3 = nn.Linear(l2_size, 2)
        else:
            self.fc2 = nn.Identity()
            self.fc3 = nn.Linear(l1_size, 2)

    # Lazy Layer Initialization
    @property
    def fc1(self):
        if self._fc1 is None:
            self._fc1 = nn.Sequential(
                nn.Linear(
                    self.fc1__in_features,
                    self.hparams['l1_size']
                ),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )
        return self._fc1

    def forward(self, x):
        # calling feature extraction layer sequence
        x = self.fe_stack(x)
        # Flatting all dimensions but batch-dimension
        self.fc1__in_features = np.prod(x.shape[1:])
        x = x.view(-1, self.fc1__in_features)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr = self.lr
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        loss = F.cross_entropy(p, y)
        self.log("train_loss", loss, prog_bar = True)
        nni.report_intermediate_result(loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        loss = F.cross_entropy(p, y)
        self.log('test_loss', loss, prog_bar = True)
        return loss

    def train_and_test_model(self, batch_size, epochs):

        # ============
        # Preparing Raw Training and Testing Datasets
        (x_train, y_train), (x_test, y_test) = hoh_dataset()
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).long()
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).long()

        x_train = torch.permute(x_train, (0, 3, 1, 2))
        x_test = torch.permute(x_test, (0, 3, 1, 2))

        # Dataset to DataLoader
        train_ds = SimpleDataset(x_train, y_train)
        test_ds = SimpleDataset(x_test, y_test)

        train_loader = DataLoader(train_ds, batch_size)
        test_loader = DataLoader(test_ds, batch_size)

        # PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs = epochs,
            checkpoint_callback = False
        )

        # Model Training
        trainer.fit(self, train_loader)

        # Testing
        test_loss = trainer.test(self, test_loader)

        output = self(x_test)
        predict = output.argmax(dim = 1, keepdim = True)
        accuracy = round(accuracy_score(predict, y_test), 4)

        return accuracy
