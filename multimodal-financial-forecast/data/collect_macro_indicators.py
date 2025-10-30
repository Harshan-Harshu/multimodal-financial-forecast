import os
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

FRED_API_KEY = "8d0106ceaf5312fe0743b13b06edfdd7"  
OUTPUT_DIR = "data/raw/macro/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRED_SERIES = {
    "CPI": "CPIAUCSL",             
    "Unemployment": "UNRATE",    
    "InterestRate": "FEDFUNDS",   
}

def fetch_fred_series(series_id, label, start_date="2015-01-01"):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
    }

    print(f"📊 Fetching {label} data from FRED...")
    r = requests.get(url, params=params)
    data = r.json()

    if "observations" not in data:
        raise Exception(f"❌ Failed to fetch {label}: {data}")

    df = pd.DataFrame(data["observations"])
    df = df[["date", "value"]]
    df.columns = ["Date", label]
    df["Date"] = pd.to_datetime(df["Date"])
    df[label] = pd.to_numeric(df[label], errors="coerce")

    file_path = os.path.join(OUTPUT_DIR, f"{label}.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ Saved {label} data to {file_path}")
    return df


if __name__ == "__main__":
    all_data = []
    for label, series_id in FRED_SERIES.items():
        df = fetch_fred_series(series_id=series_id, label=label)
        all_data.append(df)

    print("🔗 Merging all indicators...")
    merged = all_data[0]
    for df in all_data[1:]:
        merged = pd.merge(merged, df, on="Date", how="outer")

    merged.sort_values("Date", inplace=True)
    merged.to_csv(os.path.join(OUTPUT_DIR, "macro_combined.csv"), index=False)
    print("✅ Combined macro indicators saved.")
