import dvc.api
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def load_data():
    path = "downloaded_content/md5/5e/cadf373a71bc815bebeac93526d48b"
    with dvc.api.open(path, mode="rb") as f:
        data = pd.read_csv(f)
    return data


def start_prepare():
    data = load_data()
    data = data["Цена"]
    data = data.apply(lambda x: x[:-2])
    all_data = data.astype(float)
    all_data = all_data * 1000
    return all_data


def prepare_data(stock_data, sequence_length=60, test_size=0.2):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))

    train_data, test_data = train_test_split(
        scaled_data, test_size=test_size, shuffle=False
    )

    train_dataset = StockDataset(train_data, sequence_length)
    test_dataset = StockDataset(test_data, sequence_length)

    return train_dataset, test_dataset, scaler


class StockDataset(Dataset):
    def __init__(self, data, sequence_length=60):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx : idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        return torch.FloatTensor(sequence), torch.FloatTensor(target)
