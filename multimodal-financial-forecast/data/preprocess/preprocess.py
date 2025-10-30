import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

price_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USD_price.csv")
output_path = os.path.join(BASE_DIR, "data", "processed", "merged_features.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"📂 Reading price data from: {price_path}")
if not os.path.exists(price_path):
    raise FileNotFoundError(f"❌ File not found: {price_path}")

price_df = pd.read_csv(price_path)
print("📊 Available columns in CSV:", list(price_df.columns))

date_col = None
for col in price_df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        date_col = col
        break

if date_col is None:
    raise ValueError("❌ No date/time column found in the CSV.")

price_df[date_col] = pd.to_datetime(price_df[date_col])
price_df = price_df.set_index(date_col).sort_index()

for col in price_df.columns:
    price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

price_df = price_df.ffill().dropna()

if price_df.empty:
    raise ValueError("❌ DataFrame is empty after cleaning.")

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(price_df)
normalized_df = pd.DataFrame(scaled_values, columns=price_df.columns, index=price_df.index)

normalized_df.to_csv(output_path)
print(f"✅ Preprocessed and saved to: {output_path}")
