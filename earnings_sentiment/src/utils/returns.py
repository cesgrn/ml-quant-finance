import yfinance as yf
import pandas as pd
from datetime import timedelta

def get_return(ticker: str, date_str: str, days_forward: int = 1) -> float:
    try:
        event = pd.to_datetime(date_str, format="mixed")
        start = event - timedelta(days=5)
        end = event + timedelta(days=10)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        df["return"] = df["Close"].pct_change()
        dates = df.index.tolist()
        closest = min(dates, key=lambda x: abs(x - event))
        idx = dates.index(closest)
        if idx + days_forward < len(dates):
            return float(df["return"].iloc[idx + days_forward])
    except:
        return None
    return None