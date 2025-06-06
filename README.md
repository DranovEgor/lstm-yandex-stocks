# LSTM for Yandex stock price

## Overview

The project is dedicated to predicting the stock price using the LSTM model. The task is to predict the price for the next 10 days, where only the past price is used as a training sample. As an example, the price of Yandex shares traded on MOEX in the period from March 2023 to March 2025 is considered.

[Data](https://ru.investing.com/equities/yandex-historical-data?ysclid=m81ihsakju683720297)

## Setup

### Requrements

- Python
- Poetry
- PyTorch Lightning
- DVC

### Installation

1. Download repository

```sh
git clone https://github.com/DranovEgor/lstm-yandex-stocks
```

2. Install dependencies with Poetry:

```sh
poetry install
```

3. Activate virtual enviroment:

```sh
poetry shell
```

4. All executable files are located in lstm_for_yndx.py.

```sh
cd lstm_for_yndx
```

## Train

### 1. Data loading

In order to download data using DVC, need to run the download_data.py.

```sh
python3 download_data.py
```

### 2. Train the model

Running train process.

```sh
python3 train.py
```

## Infer

Predict prices for the next 10 days.

```sh
python3 train.py
```
