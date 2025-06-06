import matplotlib.pyplot as plt
import numpy as np
import torch


def predict_future(model, scaler, initial_sequence, days_to_predict=10):
    model.eval()
    predictions = []

    if hasattr(initial_sequence, "values"):
        current_sequence = initial_sequence.values
    else:
        current_sequence = np.array(initial_sequence).copy()

    if current_sequence.ndim == 1:
        current_sequence = current_sequence.reshape(-1, 1)

    with torch.no_grad():
        for _ in range(days_to_predict):
            scaled_sequence = scaler.transform(current_sequence)
            input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
            prediction = model(input_tensor)
            predicted_value = scaler.inverse_transform(
                prediction.numpy().reshape(-1, 1)
            )
            predictions.append(predicted_value[0][0])
            current_sequence = np.vstack([current_sequence[1:], predicted_value])

    return predictions


def plot_predictions(actual_prices, predicted_prices, sequence_length):
    plt.figure(figsize=(12, 6))

    if hasattr(actual_prices, "values"):
        actual_prices = actual_prices.values

    plt.plot(actual_prices, label="Actual Prices", color="blue")

    last_actual = actual_prices[-1]
    pred_dates = range(
        len(actual_prices), len(actual_prices) + len(predicted_prices) + 1
    )
    plt.plot(
        pred_dates,
        [last_actual] + list(predicted_prices),
        label="Predicted Prices",
        color="red",
        linestyle="--",
    )

    plt.axvline(x=len(actual_prices) - 0.5, color="green", linestyle="--")

    plt.title("Stock Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig("../predictions/price_prediction.png")
