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

### Installation

```bash
# 1. Clone repository
git clone https://github.com/DranovEgor/lstm-yandex-stocks
cd lstm_for_yndx/lstm_for_yndx

# 2. Install dependencies
poetry install

# 3. Activate virtual environment
poetry shell
```

## 🚀 Training Pipeline

### 1. Data Loading

In order to download data using DVC, need to run the download_data.py.

```sh
# Download and preprocess data
python3 download_data.py
```

### 2. Model Training

Running train process.

```sh
# Training
python3 train.py
```

## 🔮 Inference



```sh
# Generate 10-day predictions
python3 infer.py
```


## 📊 Sample Results

Prediction outputs are saved in `/predictions` directory:
- 📊 `*.png`: Prediction visualization plots
- 📄 `*.csv`: Forecasted values for next 10 days


## 📝 Logging

All training metrics are automatically logged using MLflow. The results are organized in plots.


## 📧 Contact

- GitHub: [DranovEgor](https://github.com/DranovEgor)
