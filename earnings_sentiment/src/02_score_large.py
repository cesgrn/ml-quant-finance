import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

def load_finbert():
    print("Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model

def score_text(text: str, tokenizer, model) -> float:
    sentences = [s.strip() for s in text.split("\n") if len(s.strip()) > 30][:40]
    if not sentences:
        return 0.0

    all_pos, all_neg = [], []
    for i in range(0, len(sentences), 8):
        batch = sentences[i:i+8]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        all_pos.extend(probs[:, 0])
        all_neg.extend(probs[:, 1])

    return round(float(np.mean(all_pos)) - float(np.mean(all_neg)), 4)

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

if __name__ == "__main__":
    df = pd.read_pickle("data/raw/motley-fool-data.pkl")

    # Broader filter — any year, top tickers
    sp500 = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
             "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS",
             "BAC", "XOM", "PFE", "CSCO", "INTC", "NFLX", "ADBE", "CRM"]

    sample = df[df["ticker"].isin(sp500)].head(80)
    print(f"Processing {len(sample)} transcripts...")

    tokenizer, model = load_finbert()

    results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        if i % 10 == 0:
            print(f"  {i}/{len(sample)}...")

        score = score_text(str(row["transcript"]), tokenizer, model)

        # Fix date parsing — format: "Aug 27, 2020, 9:00 p.m. ET"
        try:
            date_clean = row["date"].split(",")[0:2]
            date_clean = ",".join(date_clean).strip()  # "Aug 27, 2020"
            ret = get_return(row["ticker"], date_clean, days_forward=1)
        except:
            ret = None

        results.append({
            "ticker": row["ticker"],
            "date": row["date"],
            "quarter": row["q"],
            "sentiment_score": score,
            "return_t1": ret
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/processed/large_scale_results.csv", index=False)

    # Filter only rows with valid returns
    valid = results_df.dropna(subset=["return_t1"])
    print(f"\nValid samples : {len(valid)} / {len(results_df)}")

    if len(valid) > 1:
        corr = valid["sentiment_score"].corr(valid["return_t1"])
        print(f"Correlation sentiment vs t+1 return : {corr:.3f}")
        print(f"\nSentiment score stats:")
        print(valid["sentiment_score"].describe().round(3))