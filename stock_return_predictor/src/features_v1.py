import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    # --- Returns ---
    df["return_1d"] = df["Close"].pct_change(1)       # variation % jour J
    df["return_5d"] = df["Close"].pct_change(5)       # variation % sur 5 jours

    # --- Moving Averages ---
    df["sma_10"] = df["Close"].rolling(10).mean()     # moyenne mobile 10 jours
    df["sma_50"] = df["Close"].rolling(50).mean()     # moyenne mobile 50 jours
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]     # ratio : tendance court/long terme

    # --- Volatilité rolling ---
    df["volatility_10"] = df["return_1d"].rolling(10).std()  # écart-type des returns sur 10j

    # --- RSI (14 jours) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))

    # --- Target : est-ce que le lendemain monte ? ---
    df["target"] = (df["return_1d"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    raw = pd.read_csv("data/SPY_raw.csv", header=[0,1], index_col=0, parse_dates=True)
    # Flatten multi-level columns si nécessaire
    raw.columns = raw.columns.get_level_values(0)
    df = add_features(raw)
    print(df[["Close", "sma_ratio", "volatility_10", "rsi_14", "target"]].tail())
    df.to_csv("data/SPY_features_v1.csv")