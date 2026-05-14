import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Earnings call dates (when each transcript was published)
EARNINGS_DATES = {
    "AAPL":  "2024-10-31",
    "MSFT":  "2024-10-30",
    "NVDA":  "2024-11-20",
    "GOOGL": "2025-02-05",
    "META":  "2024-10-30",
}

def get_returns_around_event(ticker: str, event_date: str, window: int = 5) -> dict:
    """Fetch stock returns from day -1 to day +5 around earnings date."""
    start = pd.Timestamp(event_date) - pd.Timedelta(days=10)
    end   = pd.Timestamp(event_date) + pd.Timedelta(days=10)

    df = yf.download(ticker, start=start, end=end, progress=False)
    df["return"] = df["Close"].pct_change()
    df.index = pd.to_datetime(df.index)

    # Find the event date index
    dates = df.index.tolist()
    event_ts = pd.Timestamp(event_date)

    # Find closest trading day to event date
    closest = min(dates, key=lambda x: abs(x - event_ts))
    idx = dates.index(closest)

    results = {"ticker": ticker, "event_date": event_date}
    for d in [1, 2, 3, 5]:
        if idx + d < len(dates):
            ret = float(df["return"].iloc[idx + d])
            results[f"return_t+{d}"] = round(ret, 4)
        else:
            results[f"return_t+{d}"] = None

    return results

if __name__ == "__main__":
    # Load sentiment scores
    scores = pd.read_csv("data/transcripts_scored.csv")

    # Fetch returns around each earnings date
    print("Fetching returns around earnings dates...")
    returns = []
    for ticker, date in EARNINGS_DATES.items():
        r = get_returns_around_event(ticker, date)
        returns.append(r)
        print(f"  {ticker} ({date}) : t+1={r.get('return_t+1', 'N/A'):.2%}")

    returns_df = pd.DataFrame(returns)

    # Merge with sentiment scores
    df = scores.merge(returns_df, on="ticker")
    df.to_csv("data/event_study_results.csv", index=False)

    # Display results
    print("\n--- Event Study Results ---")
    cols = ["ticker", "sentiment_score", "return_t+1", "return_t+2", "return_t+3", "return_t+5"]
    print(df[cols].to_string(index=False))

    # Correlation
    print("\n--- Correlation: Sentiment Score vs Returns ---")
    for col in ["return_t+1", "return_t+2", "return_t+3", "return_t+5"]:
        corr = df["sentiment_score"].corr(df[col])
        print(f"  sentiment_score vs {col} : {corr:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, col in enumerate(["return_t+1", "return_t+2", "return_t+3", "return_t+5"]):
        axes[i].scatter(df["sentiment_score"], df[col], color="royalblue")
        for _, row in df.iterrows():
            axes[i].annotate(row["ticker"],
                           (row["sentiment_score"], row[col]),
                           fontsize=8, ha="center", va="bottom")
        axes[i].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[i].set_xlabel("Sentiment Score")
        axes[i].set_ylabel("Stock Return")
        axes[i].set_title(col)

    plt.suptitle("FinBERT Sentiment vs Post-Earnings Returns", fontsize=13)
    plt.tight_layout()
    plt.savefig("data/event_study.png", dpi=150)
    print("\nPlot saved to data/event_study.png")