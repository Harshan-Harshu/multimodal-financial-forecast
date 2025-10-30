import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
import pandas as pd
import argparse
from models.forecasting.lstm import LSTMForecaster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path="pipeline/trained_models/lstm_forecaster.pt", config_path="pipeline/trained_models/config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Missing config file: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Missing model file: {model_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    model = LSTMForecaster(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["output_size"]
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_batch(input_data, model=None):
    """
    input_data: numpy array of shape [batch_size, seq_len, input_size]
    returns: list of predictions
    """
    if model is None:
        model = load_model()

    if isinstance(input_data, list):
        input_data = np.array(input_data)

    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    return outputs.cpu().numpy().flatten().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference for LSTM Forecasting")
    parser.add_argument("--input", type=str, required=True, help="Path to .npy or .csv input file")
    args = parser.parse_args()

    if args.input.endswith(".npy"):
        input_data = np.load(args.input)
    elif args.input.endswith(".csv"):
        df = pd.read_csv(args.input).dropna().select_dtypes(include=[np.number])
        data = df.values
        if len(data.shape) == 2:
            input_data = np.expand_dims(data, axis=0) 
        else:
            input_data = data
    else:
        raise ValueError("Input must be a .npy or .csv file")

    model = load_model()
    predictions = predict_batch(input_data, model)
    print("📈 Predictions:", predictions)
