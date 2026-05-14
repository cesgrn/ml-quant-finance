import pandas as pd
import yfinance as yf
import os

import pandas as pd
import os

def get_sentiment_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/sent_train.csv")
    label_map = {0: "bearish", 1: "bullish", 2: "neutral"}
    df["label"] = df["label"].map(label_map)
    print(f"Dataset shape : {df.shape}")
    print(f"Label distribution :\n{df['label'].value_counts()}")
    print(df.head(3))
    return df

def get_earnings_dates(tickers: list) -> pd.DataFrame:
    records = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            if cal is not None and "Earnings Date" in cal:
                date = cal["Earnings Date"]
                records.append({"ticker": ticker, "earnings_date": date})
                print(f"{ticker} : next earnings {date}")
        except Exception as e:
            print(f"{ticker} : error — {e}")
    return pd.DataFrame(records)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = get_sentiment_dataset()
    df.to_csv("data/financial_phrasebank.csv", index=False)
    print(f"\nSaved {len(df)} sentences to data/financial_phrasebank.csv")

    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"]
    get_earnings_dates(tickers)