import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_finbert():
    print("Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model

def score_text(text: str, tokenizer, model) -> dict:
    # Split text into sentences and score each
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20][:50]
    
    all_pos, all_neg, all_neu = [], [], []
    
    for i in range(0, len(sentences), 8):
        batch = sentences[i:i+8]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        all_pos.extend(probs[:, 0])
        all_neg.extend(probs[:, 1])
        all_neu.extend(probs[:, 2])

    return {
        "positive": round(float(np.mean(all_pos)), 4),
        "negative": round(float(np.mean(all_neg)), 4),
        "neutral":  round(float(np.mean(all_neu)), 4),
        "sentiment_score": round(float(np.mean(all_pos)) - float(np.mean(all_neg)), 4),
        "n_sentences": len(sentences)
    }

if __name__ == "__main__":
    df = pd.read_csv("data/transcripts_raw.csv")
    tokenizer, model = load_finbert()

    results = []
    for _, row in df.iterrows():
        print(f"Scoring {row['ticker']}...")
        scores = score_text(row["text"], tokenizer, model)
        scores["ticker"] = row["ticker"]
        scores["url"] = row["url"]
        results.append(scores)

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/transcripts_scored.csv", index=False)

    print("\n--- Sentiment Scores per Transcript ---")
    print(results_df[["ticker", "positive", "negative", "neutral",
                       "sentiment_score", "n_sentences"]].to_string(index=False))