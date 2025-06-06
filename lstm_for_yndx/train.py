from pathlib import Path

import git
import hydra
import matplotlib.pyplot as plt
import mlflow
import pytorch_lightning as pl
import torch
from data import prepare_data, start_prepare
from model import StockLSTM
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def setup_mlflow(cfg):
    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    mlflow.start_run()

    try:
        repo = git.Repo(search_parent_directories=True)
        mlflow.log_param("git_commit", repo.head.object.hexsha)
    except Exception:
        mlflow.log_param("git_commit", "unknown")


def log_metrics_and_plots(trainer, model, cfg):
    plots_dir = Path("../plots")
    plots_dir.mkdir(exist_ok=True)

    if hasattr(model, "epoch_train_losses") and hasattr(model, "epoch_val_losses"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(model.epoch_train_losses, label="Train Loss")
        ax.plot(model.epoch_val_losses, label="Validation Loss")
        ax.set_title("Training and Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        loss_plot_path = plots_dir / "loss_plot.png"
        plt.savefig(loss_plot_path)
        mlflow.log_artifact(loss_plot_path)
        plt.close()
        print(f"Saved loss plot with {len(model.epoch_train_losses)} points")
    else:
        print("Warning: No loss data available for plotting")

    try:
        metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}
        mlflow.log_metrics(metrics)
        print("Logged metrics:", metrics)
    except Exception as e:
        print("Error logging metrics:", str(e))


def train_model(stock_data, cfg):
    sequence_length = cfg["sequence_length"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]

    setup_mlflow(cfg)

    train_dataset, test_dataset, scaler = prepare_data(stock_data, sequence_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = StockLSTM(
        input_size=1,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        output_size=1,
        learning_rate=cfg["learning_rate"],
    )

    trainer = pl.Trainer(
        max_epochs=epochs, callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")]
    )
    trainer.fit(model, train_loader, val_loader)

    log_metrics_and_plots(trainer, model, cfg)

    torch.save(model, "../models/model_full.pt")
    torch.save(scaler, "../models/scaler.pt")

    mlflow.log_artifact("../models/model_full.pt")
    mlflow.log_artifact("../models/scaler.pt")

    mlflow.end_run()

    return model, scaler


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    all_data = start_prepare()
    params = OmegaConf.to_container(cfg["params"])
    model, scaler = train_model(all_data, params)


if __name__ == "__main__":
    main()
