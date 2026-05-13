import pandas as pd
import numpy as np
import yfinance as yf

def add_features(df: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:

    # --- Returns ---
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # --- Moving Averages ---
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]

    # --- Volatilité ---
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()

    # --- RSI ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))

    # --- Volume ---
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- VIX (indice de peur du marché) ---
    df["vix"] = vix["Close"].reindex(df.index)
    df["vix_change"] = vix["Close"].pct_change(1).reindex(df.index)

    # --- Target ---
    df["target"] = (df["return_1d"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    raw = pd.read_csv("data/SPY_raw.csv", header=[0,1], index_col=0, parse_dates=True)
    raw.columns = raw.columns.get_level_values(0)

    vix = yf.download("^VIX", start="2015-01-01", end="2026-01-01")
    vix.columns = vix.columns.get_level_values(0)

    df = add_features(raw, vix)
    print(df[["Close", "vix", "volume_ratio", "rsi_14", "target"]].tail())
    df.to_csv("data/SPY_features_v2.csv")