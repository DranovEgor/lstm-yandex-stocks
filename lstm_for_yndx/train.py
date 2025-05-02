from model import StockLSTM
from data import start_prepare, prepare_data, StockDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch


def train_model(stock_data):
    sequence_length = 60
    batch_size = 64
    epochs = 30
    
    train_dataset, test_dataset, scaler = prepare_data(stock_data, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = StockLSTM(
        input_size=1,
        hidden_size=50,
        num_layers=1,
        output_size=1,
        learning_rate=0.001
    )
    
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, train_loader, val_loader)
    torch.save(model, "model_full.pt")
    torch.save(scaler, 'scaler.pt')
    
    return model, scaler

if __name__ == "__main__":
    all_data = start_prepare()
    model, scaler = train_model(all_data)

