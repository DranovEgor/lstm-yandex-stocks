import torch
from data import start_prepare
from predict import plot_predictions, predict_future


def inference(data):
    model = torch.load("model_full.pt", weights_only=False)
    scaler = torch.load("scaler.pt", weights_only=False)
    model.eval()
    with torch.no_grad():
        last_sequence = data.values[-60:]
        scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
        input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
        prediction = model(input_tensor)
        predicted_price = scaler.inverse_transform(prediction.numpy().reshape(-1, 1))
        print(f"Predicted next day price: {predicted_price[0][0]}")

    sequence_length = 60
    days_to_predict = 10

    last_sequence = all_data[-sequence_length:]
    predicted_prices = predict_future(model, scaler, last_sequence, days_to_predict)
    plot_predictions(all_data, predicted_prices, sequence_length)
    print("Predicted prices for next", days_to_predict, "days:")
    for i, price in enumerate(predicted_prices, 1):
        print(f"Day1 {i}: {price:.2f}")


if __name__ == "__main__":
    all_data = start_prepare()
    inference(all_data)
