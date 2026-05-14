import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_finbert():
    # ProsusAI/finbert — fine-tuned for financial sentiment classification
    # Labels : positive, negative, neutral
    print("Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    print("FinBERT loaded.")
    return tokenizer, model

def score_batch(texts: list, tokenizer, model, batch_size=32) -> list:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        # FinBERT label order: positive=0, negative=1, neutral=2
        for prob in probs:
            results.append({
                "positive": round(float(prob[0]), 4),
                "negative": round(float(prob[1]), 4),
                "neutral":  round(float(prob[2]), 4),
                "finbert_label": ["positive", "negative", "neutral"][np.argmax(prob)],
                "finbert_score": round(float(prob[0]) - float(prob[1]), 4)
            })
        if i % 500 == 0:
            print(f"Scored {i}/{len(texts)} sentences...")
    return results

if __name__ == "__main__":
    df = pd.read_csv("data/financial_phrasebank.csv")

    tokenizer, model = load_finbert()

    print(f"\nScoring {len(df)} sentences with FinBERT...")
    scores = score_batch(df["text"].tolist(), tokenizer, model)
    scores_df = pd.DataFrame(scores)

    df = pd.concat([df, scores_df], axis=1)
    df.to_csv("data/transcripts_scored.csv", index=False)

    print("\n--- Sample results ---")
    print(df[["text", "label", "finbert_label", "finbert_score"]].head(10))

    # Accuracy vs ground truth
    # Map our labels to FinBERT labels
    label_map = {"bullish": "positive", "bearish": "negative", "neutral": "neutral"}
    df["label_mapped"] = df["label"].map(label_map)
    accuracy = (df["finbert_label"] == df["label_mapped"]).mean()
    print(f"\nFinBERT accuracy vs ground truth : {accuracy:.2%}")