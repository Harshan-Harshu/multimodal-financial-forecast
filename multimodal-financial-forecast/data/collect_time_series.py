import yfinance as yf
import os
from datetime import datetime

def fetch_crypto(symbol="BTC-USD", period="30d", interval="1h", output_dir="data/raw"):
    print(f"📥 Downloading {symbol} data for period: {period}, interval: {interval}")
    
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"❌ No data found for {symbol}. Check symbol or internet connection.")

    df.reset_index(inplace=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    safe_symbol = symbol.replace("=", "_").replace("-", "_")
    file_path = os.path.join(output_dir, f"{safe_symbol}_price.csv")
    
    df.to_csv(file_path, index=False)
    print(f"✅ Saved {symbol} data to {file_path}")

    return df

if __name__ == "__main__":
    fetch_crypto(symbol="BTC-USD", period="30d", interval="1h")
