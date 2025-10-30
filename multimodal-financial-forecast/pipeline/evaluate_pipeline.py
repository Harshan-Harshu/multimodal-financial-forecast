import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.forecasting.lstm import LSTMForecaster


def load_test_data(csv_path="../data/processed/merged_features.csv", seq_len=30, test_ratio=0.2):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number]).dropna()
    data = df.values

    split = int(len(data) * (1 - test_ratio))
    test_data = data[split:]

    X_test, y_test = [], []
    for i in range(len(test_data) - seq_len - 1):
        X_test.append(test_data[i:i+seq_len])
        y_test.append(test_data[i+seq_len][0])  

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32).unsqueeze(1)

    return DataLoader(TensorDataset(X_test, y_test), batch_size=32), X_test.shape[2]


def evaluate_model(test_loader, model_path="./trained_models/lstm_forecaster.pt", config_path="./trained_models/config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    model = LSTMForecaster(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["output_size"]
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for X, y in test_loader:
            out = model(X)
            preds.append(out.numpy())
            targets.append(y.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print("\n📊 Evaluation Metrics:")
    print(f"➡️ MSE: {mse:.4f}")
    print(f"➡️ MAE: {mae:.4f}")
    print(f"➡️ R2 Score: {r2:.4f}")

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(targets, label='Actual', color='blue')
    plt.plot(preds, label='Predicted', color='orange')
    plt.title("📈 LSTM Forecast vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/evaluation_plot.png")
    plt.show()


if __name__ == "__main__":
    test_loader, input_size = load_test_data()
    print(f"🧪 Detected input feature size: {input_size}")
    evaluate_model(test_loader)
