import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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