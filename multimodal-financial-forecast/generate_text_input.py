import pandas as pd
import numpy as np
import os

DATA_PATH = "./data/processed/merged_features.csv"
SEQ_LEN = 30  
SAVE_PATH = "./test_input.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Cannot find data file at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df = df.select_dtypes(include=[np.number]).dropna()
data = df.values

if len(data) < SEQ_LEN:
    raise ValueError(f"❌ Not enough data: Need at least {SEQ_LEN} rows for one input sequence.")

sample = data[:SEQ_LEN] 

np.savetxt(SAVE_PATH, sample, delimiter=",")
print(f"✅ Test input saved to: {SAVE_PATH}")
print(f"ℹ️ Shape of saved data: {sample.shape}")
