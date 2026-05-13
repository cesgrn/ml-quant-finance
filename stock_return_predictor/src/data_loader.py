import yfinance as yf
import pandas as pd

def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = get_price_data("SPY", "2015-01-01", "2026-01-01")
    df.to_csv("data/SPY_raw.csv")
    print(df.tail())