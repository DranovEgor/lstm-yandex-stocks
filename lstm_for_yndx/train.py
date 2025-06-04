import hydra
import pytorch_lightning as pl
import torch
from data import prepare_data, start_prepare
from model import StockLSTM
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def train_model(stock_data, cfg):
    sequence_length = cfg["sequence_length"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]

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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    all_data = start_prepare()
    params = OmegaConf.to_container(cfg["params"])
    model, scaler = train_model(all_data, params)


if __name__ == "__main__":
    main()
