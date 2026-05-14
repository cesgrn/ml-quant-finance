# Earnings Sentiment Analyzer — ML x Finance

Predicting post-earnings stock returns using NLP sentiment analysis on earnings call transcripts. Built as Project 2 of a broader ML x Quant Finance learning series.

This project directly addresses the key limitation discovered in Project 1: **technical price features alone cannot capture event-driven stocks like NVDA**. The hypothesis here is that the language used by executives during earnings calls contains predictive signal for short-term stock returns.

---

## Goal

1. Collect real earnings call transcripts for a basket of tickers
2. Score each transcript's sentiment using **FinBERT** — a transformer model pre-trained on financial text
3. Run an **event study** to test whether sentiment predicts post-earnings returns over 1 to 5 days
4. Compare FinBERT scores against a **Loughran-McDonald dictionary baseline**

---

## Stack

- **Data** : Motley Fool (scraping), Kaggle (18,755 transcripts), HuggingFace, yfinance
- **NLP** : HuggingFace Transformers, FinBERT (`ProsusAI/finbert`), pysentiment2
- **Analysis** : pandas, numpy, scipy
- **Visualization** : matplotlib, seaborn

---

## Key References

- **Yang, Uy & Huang (2020) — "FinBERT: A Pretrained Language Model for Financial Communications"**: FinBERT is BERT re-trained on 4.9 billion tokens of financial text (SEC filings, earnings call transcripts, analyst reports). It achieves 87.2% accuracy on the Financial PhraseBank. Used here as the main sentiment scorer.
- **Vaswani et al. (2017) — "Attention Is All You Need"**: the foundational Transformer paper. FinBERT is built on this architecture. Self-attention allows the model to understand context that word-count dictionaries cannot capture.
- **Loughran & McDonald (2011) — "When Is a Liability Not a Liability?"** *(Journal of Finance)*: the classic financial sentiment dictionary. Used here as a baseline.

---

## Project Structure

```
earnings_sentiment/
├── data/
│   ├── raw/
│   │   ├── motley-fool-data.pkl      ← 18,755 transcripts (Kaggle)
│   │   ├── sent_train.csv            ← Twitter Financial News (9,543 tweets)
│   │   └── transcripts_5.csv         ← 5 transcripts scraped from Motley Fool
│   └── processed/
│       ├── transcripts_scored.csv    ← FinBERT scores (5 transcripts)
│       ├── large_scale_results.csv   ← FinBERT scores (78 transcripts)
│       ├── event_study_results.csv   ← returns around earnings dates
│       ├── comparison_lm.csv         ← FinBERT vs LM comparison
│       └── event_study.png           ← sentiment vs returns scatter plots
├── src/
│   ├── utils/
│   │   ├── finbert.py                ← load_finbert(), score_text()
│   │   └── returns.py                ← get_return() via yfinance
│   ├── 01_collect.py                 ← data loader + earnings dates
│   ├── 01_scraper.py                 ← scrape transcripts from Motley Fool
│   ├── 02_score.py                   ← FinBERT scoring on 5 transcripts
│   ├── 02_score_large.py             ← FinBERT scoring on 78 transcripts
│   ├── 03_event_study.py             ← correlation sentiment vs returns
│   ├── 04_lm_baseline.py             ← Loughran-McDonald comparison
│   └── 05_validate_finbert.py        ← accuracy on Twitter dataset
└── README.md
```

---

## Data

### 1. Earnings Call Transcripts
- **Primary** : Kaggle — Motley Fool dataset (18,755 transcripts, 2020-2024)
- **Secondary** : 5 manually scraped transcripts from Motley Fool (AAPL, MSFT, NVDA, GOOGL, META — Q3/Q4 2024)

**Data collection challenges:**
- Finnhub transcripts API requires paid plan ($49.99/month)
- SEC EDGAR 8-K exhibit scraping returned no transcripts for these tickers
- Final solution: Kaggle dataset + direct URL scraping from Motley Fool

### 2. Twitter Financial News Sentiment (validation)
- **Source** : HuggingFace — `zeroshot/twitter-financial-news-sentiment`
- **Size** : 9,543 tweets — bearish / bullish / neutral
- **FinBERT accuracy** : **71.21%**

---

## Methodology & Results

### Step 1 — FinBERT Validation on Twitter Dataset ✅
**Accuracy : 71.21%** on 9,543 labeled financial tweets. Lower than the 87.2% on Financial PhraseBank because tweets are informal — FinBERT was trained on formal financial text (SEC filings, analyst reports).

---

### Step 2 — Transcript Scoring ✅

Each transcript is split into sentences, scored by FinBERT in batches of 8, and aggregated:

$$\text{Sentiment Score} = \overline{P(\text{positive})} - \overline{P(\text{negative})}$$

**5-transcript pilot (AAPL, MSFT, NVDA, GOOGL, META — 2024):**

| Ticker | Positive | Negative | Neutral | Sentiment Score |
|---|---|---|---|---|
| AAPL | 0.265 | 0.077 | 0.658 | **+0.188** |
| MSFT | 0.183 | 0.074 | 0.743 | **+0.108** |
| NVDA | 0.246 | 0.080 | 0.674 | **+0.166** |
| GOOGL | 0.214 | 0.069 | 0.717 | **+0.146** |
| META | 0.230 | 0.076 | 0.694 | **+0.154** |

**78-transcript large scale (S&P 500, 2020-2024):**

| Metric | Value |
|---|---|
| Mean sentiment score | 0.284 |
| Std | 0.131 |
| Min / Max | 0.075 / 0.531 |

---

### Step 3 — Event Study ✅

**5-transcript pilot:**

| Ticker | Sentiment | t+1 | t+2 | t+3 | t+5 |
|---|---|---|---|---|---|
| AAPL | +0.188 | -1.33% | -0.40% | +0.65% | +2.14% |
| MSFT | +0.108 | -6.05% | +0.99% | -0.47% | +2.12% |
| NVDA | +0.166 | +0.53% | -3.22% | -4.18% | -1.15% |
| GOOGL | +0.146 | +0.14% | -3.27% | +0.61% | -0.92% |
| META | +0.154 | -4.09% | -0.07% | -1.14% | -0.07% |

| Horizon | Correlation (5 samples) | Correlation (78 samples) |
|---|---|---|
| t+1 | +0.662 | **+0.002** |

> ⚠️ **Critical finding**: the t+1 correlation of +0.662 on 5 samples was statistical noise. Scaling to 78 transcripts reveals the true correlation is essentially zero. This is one of the most important lessons in quantitative finance: small samples produce misleadingly high correlations.

---

### Step 4 — Loughran-McDonald Baseline ✅

| Ticker | FinBERT | LM Dictionary |
|---|---|---|
| AAPL | +0.188 | +0.333 |
| MSFT | +0.108 | +0.158 |
| NVDA | +0.166 | +0.143 |
| GOOGL | +0.146 | +0.517 |
| META | +0.154 | +0.120 |

**Correlation between FinBERT and LM : 0.177** — near zero. The two methods measure fundamentally different things. GOOGL is the clearest example: LM gives +0.517 vs FinBERT's +0.146, because LM counts words without understanding negation or financial context.

---

## Final Conclusions

| Method | t+1 Correlation (5 samples) | t+1 Correlation (78 samples) |
|---|---|---|
| FinBERT | +0.662 | **+0.002** |
| Loughran-McDonald | not computed at scale | — |

**Absolute sentiment does not predict returns.** The market already prices in expected sentiment — what matters is the *surprise* relative to prior quarters. The next step would be to compute:

```python
sentiment_surprise = sentiment_score - mean(sentiment_score_last_4_quarters)
```

This delta-sentiment approach is what academic papers find predictive, consistent with Loughran & McDonald (2011) and the efficient market hypothesis.

---

## Key Lessons

- **Transfer learning works** — FinBERT required zero training, just inference.
- **Context matters** — negations and financial jargon make dictionary methods unreliable (FinBERT vs LM correlation: 0.177).
- **Small samples lie** — correlation of +0.662 on 5 samples became +0.002 on 78. Always scale before concluding.
- **Absolute sentiment ≠ predictive signal** — the market anticipates tone. What predicts returns is the *change* in tone vs prior quarters.
- **Data collection is the hardest part** — transcripts are not freely available via API.

---

## How to Run

```bash
# 1. Install dependencies
pip install transformers torch requests yfinance pandas beautifulsoup4 pysentiment2

# 2. Download motley-fool-data.pkl from Kaggle → place in data/raw/
# Download sent_train.csv from HuggingFace → place in data/raw/

# 3. Run full pipeline in order
python src/01_collect.py
python src/01_scraper.py
python src/02_score.py
python src/02_score_large.py
python src/03_event_study.py
python src/04_lm_baseline.py
python src/05_validate_finbert.py
```

---

*Project 2 complete. Moving to Project 3 — Neural Volatility Surface Forecaster.*
