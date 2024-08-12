import os, sys
os.environ["OMP_NUM_THREADS"] = '6'

qJetClassifier_path = os.path.abspath("/data/tdesrous/qJetClassifier")
sys.path.append(qJetClassifier_path)

from qJetClassifier import HLS4MLData150, qJetClassifier, JetDataset, accuracy
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import pennylane as qml
from torchmetrics.classification import MultilabelAccuracy, MultilabelROC
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-hl', '--hidden_layer_size', type=int, default=12)
parser.add_argument('-l', '--latent_space_size', type=int, default=6)
parser.add_argument('-r', '--nb_reuploading', type=int, default=3)
parser.add_argument('-a', '--nb_ansatz_layers', type=int, default=3)
parser.add_argument('--nb_node_features', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('-e', '--nb_epochs', type=int, default=100)
parser.add_argument('-tr', '--nb_train_samples', type=int, default=1000)
parser.add_argument('-te', '--nb_test_samples', type=int, default=100)
parser.add_argument('-s', '--sum_observable', action='store_true', default=False)

args = parser.parse_args()

model_params = {
        'hidden_layer_size': args.hidden_layer_size,
        'latent_space_size': args.latent_space_size,
        'nb_reuploading': args.nb_reuploading,
        'nb_ansatz_layers': args.nb_ansatz_layers,
        'diff_method': 'adjoint',
        'nb_node_features': args.nb_node_features,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'nb_epochs': args.nb_epochs,
        'nb_train_samples': args.nb_train_samples,
        'nb_test_samples': args.nb_test_samples,
        'sum_observable': args.sum_observable
    }

# average_roc = MultilabelROC(num_labels=5)

class LitQJetClassifier(L.LightningModule):
    def __init__(self, loss_fn):
        super().__init__()
        self.model = qJetClassifier(model_params)
        self.loss_fn = loss_fn
        self.model_params = model_params
        self.roc = MultilabelROC(num_labels=5)

        # for name, param in self.model.named_parameters():
        #     print(name, param)

    def training_step(self, batch, batch_idx):
        x, y = batch, batch.y[:batch.batch_size]
        y = torch.tensor(y)
        y_hat = self.model(x)
        # print(y_hat)
        loss = self.loss_fn(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_accuracy', acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch, batch.y[:batch.batch_size]
        y = torch.tensor(y)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_accuracy', acc, prog_bar=False, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch, batch.y[:batch.batch_size]
        y = torch.tensor(y)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc = accuracy(y_hat, y)
        y = y.type(torch.LongTensor)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_accuracy', acc, on_step=True, on_epoch=True)
        self.roc.update(y_hat, y)
      
    def on_test_epoch_end(self) -> None:
        labels = ['g', 'q', 'W', 'Z', 't']
        fig_, ax_ = self.roc.plot(score=True, labels=labels)

        log_dir = self.logger.log_dir
        path = os.path.join(log_dir, "roc_curve.png")
        fig_.savefig(path)
        self.roc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model_params['lr'])
        return optimizer

    def on_fit_start(self):
        self.logger.log_hyperparams(self.model_params)


def main():

    #Build the training_dataset
    dataset = HLS4MLData150(
        root="/data/tdesrous/qJetClassifier/data",
        nconst=6,
        feats="ptetaphi",
        norm="standard",
        train=True,
        kfolds=0,
        seed=123
    )

    #Build the test_dataset
    test_dataset = HLS4MLData150(
        root="/data/tdesrous/qJetClassifier/data_test",
        nconst=6,
        feats="ptetaphi",
        norm="standard",
        train=False,
        kfolds=0,
        seed=123
    )

    x_file_path = "/data/tdesrous/qJetClassifier/data/processed/x_train_standard_6const_ptetaphi.npy"
    y_file_path = "/data/tdesrous/qJetClassifier/data/processed/y_train_standard_6const_ptetaphi.npy"

    train_dataset = JetDataset(x_file_path, y_file_path, model_params['nb_train_samples'])

    x_test_file_path = "/data/tdesrous/qJetClassifier/data_test/processed/x_val_standard_6const_ptetaphi.npy"
    y_test_file_path = "/data/tdesrous/qJetClassifier/data_test/processed/y_val_standard_6const_ptetaphi.npy"

    test_dataset = JetDataset(x_test_file_path, y_test_file_path, model_params['nb_test_samples'])

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")


    #Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    #Split the training dataset into training and validation datasets
    # use 20% of training data for validation
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
    train_dataloader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset)

    qClassifier = LitQJetClassifier(loss_fn)

    #Train 
    trainer = L.Trainer(max_epochs=model_params['nb_epochs'], callbacks=[EarlyStopping(monitor='train_loss', mode = 'min', patience=20)], default_root_dir="/data/tdesrous/qJetClassifier/logs")
    trainer.fit(qClassifier, train_dataloader, valid_dataloader)
    trainer.test(qClassifier, dataloaders=DataLoader(test_dataset))

main()