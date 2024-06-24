from qJetClassifier import HLS4MLData150, qJetClassifier, JetDataset
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import lightning as L

class LitQJetClassifier(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch, batch.y[:batch.batch_size]
        y = torch.tensor(y)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch, batch.y[:batch.batch_size]
        y = torch.tensor(y)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch, batch.y[:batch.batch_size]
        y = torch.tensor(y)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer

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

    train_dataset = JetDataset(x_file_path, y_file_path, 1000)

    x_test_file_path = "/data/tdesrous/qJetClassifier/data_test/processed/x_val_standard_6const_ptetaphi.npy"
    y_test_file_path = "/data/tdesrous/qJetClassifier/data_test/processed/y_val_standard_6const_ptetaphi.npy"

    test_dataset = JetDataset(x_test_file_path, y_test_file_path, 100)


    model_params = {
        'hidden_layer_size': 12,
        'latent_space_size': 6,
        'nb_reuploading': 3,
        'nb_ansatz_layers': 3,
        'qdevice': qml.device('lightning.qubit', wires=6),
        'diff_method': 'adjoint',
        'nb_node_features': 3
    }

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    #Create the model
    model = qJetClassifier(model_params).to(device)
    print(model)

    #Define the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #Split the training dataset into training and validation datasets
    # use 20% of training data for validation
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset)

    qClassifier = LitQJetClassifier(model, loss_fn, optimizer)

    #Train 
    trainer = L.Trainer(max_epochs=10, callbacks=[EarlyStopping(monitor='train_loss', mode = 'min')], default_root_dir="/data/tdesrous/qJetClassifier/logs")
    trainer.fit(qClassifier, train_dataloader, valid_dataloader)