import pytorch_lightning as pl
import torch
from data import prepare_data, start_prepare
from dvc.api import DVCFileSystem
from model import StockLSTM
from torch.utils.data import DataLoader


def load_data():
    fs = DVCFileSystem()
    fs.get("../YDEX.csv", "../YDEX.csv")


def train_model(stock_data):
    sequence_length = 60
    batch_size = 64
    epochs = 30

    train_dataset, test_dataset, scaler = prepare_data(stock_data, sequence_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = StockLSTM(
        input_size=1, hidden_size=50, num_layers=1, output_size=1, learning_rate=0.001
    )

    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, train_loader, val_loader)
    torch.save(model, "../models/model_full.pt")
    torch.save(scaler, "../models/scaler.pt")

    return model, scaler


if __name__ == "__main__":
    load_data()
    all_data = start_prepare()
    model, scaler = train_model(all_data)
