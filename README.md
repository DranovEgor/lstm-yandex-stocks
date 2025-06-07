# 📈 LSTM for Yandex Stock Price Prediction

## 📌 Project Overview

This project implements an LSTM neural network to predict Yandex (MOEX: YNDX) stock prices for the next 10 days using historical price data. The model is trained on data from March 2023 to March 2025.

🔹 **Key Features**:
- 10-day ahead stock price forecasting
- LSTM architecture optimized for time-series data
- Complete training pipeline with DVC integration
- MLflow logging for experiment tracking
- Visualization of predictions

## 📊 Dataset

[Yandex Historical Data](https://ru.investing.com/equities/yandex-historical-data?ysclid=m81ihsakju683720297)

Dataset Characteristics:
- Time period: March 2023 - March 2025
- Features: Historical OHLC (Open-High-Low-Close) prices
- Frequency: Daily

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- DVC (for data versioning)
- Hydra (config management)
- MLFlow
- PyTorch Lightning

### Installation

```bash
# 1. Clone repository
git clone https://github.com/DranovEgor/lstm-yandex-stocks
cd lstm-yandex-stocks/lstm_for_yndx

# 2. Activate virtual environment
poetry shell

# 3. Install dependencies
poetry install
```

## 🚀 Training Pipeline

### 1. Data Loading

In order to download data using DVC, need to run the download_data.py.

```sh
# Download and preprocess data
poetry run python3 download_data.py
```

### 2. Model Training

```sh
# MLFlow server run
poetry run mlflow server --host 127.0.0.1 --port 8080
```

Running train process. If you're unable to load the MLflow tracking server, you can bypass this step and proceed directly to model inference using pre-trained models.

```sh
# Training
poetry run python3 train.py
```

## 🔮 Inference



```sh
# Generate 10-day predictions
poetry run python3 infer.py
```


## 📊 Sample Results

Prediction outputs are saved in `/predictions` directory:
- 📊 `*.png`: Prediction visualization plots
- 📄 `*.csv`: Forecasted values for next 10 days


## 📝 Logging

All training metrics are automatically logged using MLflow. The results are organized in plots.


## 📧 Contact

- GitHub: [DranovEgor](https://github.com/DranovEgor)
