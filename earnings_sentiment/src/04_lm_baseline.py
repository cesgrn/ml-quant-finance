import pandas as pd
import numpy as np
import pysentiment2 as ps

def score_lm(text: str) -> dict:
    lm = ps.LM()
    tokens = lm.tokenize(text)
    score = lm.get_score(tokens)
    pos = score["Positive"]
    neg = score["Negative"]
    total = pos + neg if (pos + neg) > 0 else 1
    return {
        "lm_positive": pos,
        "lm_negative": neg,
        "lm_score": round((pos - neg) / total, 4)
    }

if __name__ == "__main__":
    df = pd.read_csv("data/transcripts_scored.csv")

    print("Scoring with Loughran-McDonald dictionary...")
    # Load raw transcripts to score
    raw = pd.read_csv("data/transcripts_raw.csv")

    results = []
    for _, row in raw.iterrows():
        scores = score_lm(row["text"])
        scores["ticker"] = row["ticker"]
        results.append(scores)

    lm_df = pd.DataFrame(results)

    # Merge with FinBERT scores
    merged = df.merge(lm_df, on="ticker")
    merged.to_csv("data/comparison.csv", index=False)

    print("\n--- FinBERT vs Loughran-McDonald ---")
    print(merged[["ticker", "sentiment_score", "lm_score"]].to_string(index=False))

    print("\n--- Correlation with FinBERT score ---")
    corr = merged["sentiment_score"].corr(merged["lm_score"])
    print(f"  FinBERT vs LM : {corr:.3f}")