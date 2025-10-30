import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from models.forecasting.lstm import LSTMForecaster

df = pd.read_csv("../data/processed/merged_features.csv")
df = df.select_dtypes(include=[np.number]).dropna()
data = df.values

seq_len = 30
X, y = [], []
for i in range(len(data) - seq_len - 1):
    X.append(data[i:i+seq_len])
    y.append(data[i+seq_len][0])  

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

input_size = X.shape[2]
hidden_size = 256
num_layers = 2
output_size = 1

model = LSTMForecaster(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

os.makedirs("trained_models", exist_ok=True)
torch.save(model.state_dict(), "trained_models/lstm_forecaster.pt")

config = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "output_size": output_size
}
with open("trained_models/config.json", "w") as f:
    json.dump(config, f)

print("✅ Model trained and saved.")
