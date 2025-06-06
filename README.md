# LSTM for Yandex stock price

## Overview

Most investment strategies are based on assessing the relationship between volatility and expected stock returns. This requires models that will predict the asset price. Some theories claim that most of the information about a series is contained in its previous values. Therefore, models with long-term memory are able to predict future price values. This is what the project is dedicated to.

## Dataset

The data used is the daily values ​​of Yandex stock prices from March 9, 2023 to March 9, 2025. The data is taken from the investing.com website. As a rule, investment companies take closing prices in their models, and we will do the same. In total, there were 484 values. Note that the selected period does not cover the 2022 crisis, which means that the model will not be trained on ‘noisy’ data.

[Data](https://ru.investing.com/equities/yandex-historical-data?ysclid=m81ihsakju683720297)

##
