import pytorch_lightning as pl
import torch
import torch.nn as nn


class StockLSTM(pl.LightningModule):
    def __init__(
        self,
        input_size=1,
        hidden_size=50,
        num_layers=1,
        output_size=1,
        learning_rate=0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.training and batch_idx == 0:
            if not hasattr(self, "epoch_train_losses"):
                self.epoch_train_losses = []
            self.epoch_train_losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if not self.training and batch_idx == 0:
            if not hasattr(self, "epoch_val_losses"):
                self.epoch_val_losses = []
            self.epoch_val_losses.append(loss.item())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
