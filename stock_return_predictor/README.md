# Stock Return Predictor — ML x Finance

Predicting next-day stock returns using machine learning. Built from scratch as part of a broader ML x Quant Finance learning project.

This project was inspired by and built upon two key references:
- **QuantStart — "Forecasting Financial Time Series Part I"**: a practical introduction to ML-based market direction forecasting using Logistic Regression, LDA and QDA on the S&P 500, which established the baseline methodology and feature engineering approach used here.
- **Gu, Kelly & Xiu (2020) — "Empirical Asset Pricing via Machine Learning"** *(Review of Financial Studies)*: the benchmark academic paper comparing ML methods for return prediction across 30,000 US stocks over 60 years. Key findings applied here: gradient boosted trees and neural networks dominate linear models, momentum and volatility are the most predictive features, and nonlinear feature interactions are critical for performance.

---

## Goal

Train a model capable of predicting whether a stock will go up or down the next day, then backtest a strategy based on these predictions against a simple Buy & Hold.

---

## Stack

- **Data** : yfinance (Yahoo Finance API)
- **ML** : scikit-learn, XGBoost
- **Backtest** : pandas, numpy
- **Visualization** : matplotlib

---

## Project Structure

```
stock_return_predictor/
├── data/
│   ├── SPY_raw.csv
│   ├── SPY_features_v1.csv
│   ├── SPY_features_v2.csv
│   ├── SPY_features_v3.csv
│   └── NVDA_features_v1.csv
├── src/
│   ├── data_loader.py
│   ├── features_v1.py
│   ├── features_v2.py
│   ├── features_v3.py
│   ├── features_nvda_v1.py
│   ├── backtest.py
│   └── models/
│       ├── random_forest_v1.py
│       ├── random_forest_v2.py
│       ├── xgboost_v1.py
│       ├── xgboost_v2.py
│       ├── xgboost_v3.py
│       ├── xgboost_v4.py
│       └── xgboost_nvda_v1.py
└── README.md
```

---

## Data

- **Tickers** : SPY (S&P 500 ETF), NVDA (individual stock)
- **Period** : January 2015 → December 2025
- **Frequency** : Daily
- **Source** : Yahoo Finance via `yfinance`

---

## Features

### V1 — Basic features (5 features)

| Feature | Description |
|---|---|
| `return_1d` | 1-day price return |
| `return_5d` | 5-day price return |
| `sma_ratio` | 10-day SMA / 50-day SMA ratio |
| `volatility_10` | Rolling 10-day standard deviation of returns |
| `rsi_14` | 14-day RSI (overbought/oversold indicator) |

### V2 — Enriched features (10 features)

| Feature added | Description | Why |
|---|---|---|
| `return_10d` | 10-day price return | Medium-term momentum |
| `volatility_20` | Rolling 20-day standard deviation | Monthly volatility regime |
| `volume_ratio` | Daily volume / 20-day average volume | Detects abnormal activity |
| `vix` | VIX index level | Market fear gauge |
| `vix_change` | VIX daily % change | Reversal signal |

### V3 — Same features, new target
Same 10 features as V2. Target changed to predict **significant up moves (> +0.5%)** to filter noise.

### V4 — VIX regime filter
Added `vix_stressed` (binary: 1 if VIX > 20-day rolling average). Only trade when model is confident **and** market is stressed.

---

## Phase 1 — SPY Results

### Baseline — Random Forest V1
- **Features** : V1 | **Target** : return > 0 | **Accuracy** : 49.08%

| | ML Strategy | Buy & Hold |
|---|---|---|
| Total return | -4.75% | +68.82% |
| Sharpe ratio | -0.14 | 1.60 |
| Max drawdown | -27.44% | -18.76% |

> ❌ Underperforms Buy & Hold. Basic features not enough on an efficient index.

---

### Random Forest V2
- **Features** : V2 | **Target** : return > 0 | **Accuracy** : 50.00%

> ⚠️ Marginal improvement. Random Forest hits its ceiling.

---

### XGBoost V1
- **Features** : V2 | **Target** : return > 0 | **Accuracy** : 49.63%

> ⚠️ No improvement. Problem is the prediction strategy, not the model.

---

### XGBoost V2 — Confidence threshold

| Threshold | Trades | Accuracy | Return | Sharpe | Max DD |
|---|---|---|---|---|---|
| 50% | 325/544 | 56.00% | +0.48% | 0.08 | -28.73% |
| 55% | 236/544 | 54.24% | -8.12% | -0.37 | -23.29% |
| 60% | 148/544 | 56.76% | -8.49% | -0.55 | -15.85% |

> ⚠️ Accuracy improves with filtering but uniform feature importance (~10% each) — no dominant signal.

---

### XGBoost V3 — Significant moves target (> +0.5%)

| Threshold | Trades | Return | Sharpe | Max DD |
|---|---|---|---|---|
| 50% | 35/544 | -8.16% | -1.17 | -9.98% |
| 55% | 21/544 | -2.53% | -0.65 | -4.46% |
| 60% | 11/544 | -1.68% | -0.54 | -3.03% |

> 🔍 Key discovery: VIX emerges as dominant feature (15.6%). Market fear predicts significant up moves.

---

### XGBoost V4 — VIX Regime Filter

| Threshold | Trades | Return | Sharpe | Max DD |
|---|---|---|---|---|
| 50% | 27/540 | -9.07% | -1.52 | -9.97% |
| 55% | 23/540 | -7.33% | -1.38 | -8.24% |
| **60%** | **10/540** | **-0.38%** | **-0.12** | **-1.95%** |

> 📈 Max drawdown dropped from -27.44% to -1.95%. Nearly breakeven at 60% threshold.

**SPY progression summary:**

| Model | Return | Sharpe | Max DD |
|---|---|---|---|
| RF v1 (baseline) | -4.75% | -0.14 | -27.44% |
| XGB v2 60% | -8.49% | -0.55 | -15.85% |
| XGB v3 60% | -1.68% | -0.54 | -3.03% |
| XGB v4 60% | **-0.38%** | **-0.12** | **-1.95%** |

> **Conclusion on SPY**: SPY is too efficient for technical features alone. We extracted most of the available signal — real edge would require macro data or sentiment.

---

## Phase 2 — NVDA Results

- **Features** : V4 features + vix_stressed
- **Model** : XGBClassifier (same setup as v4)
- **Buy & Hold NVDA** : +353.39% (AI boom period 2023-2025)

| Threshold | Trades | Return | Sharpe | Max DD |
|---|---|---|---|---|
| 50% | 351/544 | -32.62% | -0.32 | -63.22% |
| 55% | 244/544 | -53.98% | -1.04 | -68.02% |
| 60% | 137/544 | -60.41% | -1.58 | -67.79% |

**Top features:**
```
volume_ratio    0.098
vix_change      0.095
vix             0.094
sma_ratio       0.092
```

> ❌ Catastrophic underperformance. NVDA's +353% return during this period was entirely **event-driven** (AI boom, CUDA demand, data center contracts). Technical features have zero predictive power on narrative-driven momentum stocks. The model repeatedly missed 10-20% single-day moves triggered by earnings and macro announcements.

> 🔍 **Key insight**: uniform feature importance (~10% each) confirms the model found no pattern at all. NVDA during the AI boom was unpredictable from price data alone — you would need NLP on earnings calls and news to capture its real drivers. This is exactly what **Project 2 (Earnings Sentiment Analyzer)** is designed to solve.

---

## Final Conclusions

| Ticker | Problem | What would actually work |
|---|---|---|
| SPY | Too efficient | Macro data, alternative data |
| NVDA | Event-driven narrative | NLP on earnings calls and news — **→ Project 2** |

Technical ML on price data alone has hard limits. The edge lies in **alternative data** — sentiment, fundamentals, macro — which is where the quant frontier currently sits (confirmed by Gu, Kelly & Xiu 2020).

---

## Key Lessons

- **Never shuffle** financial data — time has a direction. A bad split introduces data leakage.
- **Accuracy is the wrong metric** — Sharpe ratio and max drawdown matter far more.
- **SPY is highly efficient** — real edge requires less exploited signals.
- **Uniform feature importance is a warning sign** — the model found no dominant signal.
- **Event-driven stocks need NLP** — price patterns cannot capture narrative momentum.
- **Always build a baseline** — without RF v1, there is no way to measure real progress.

---

## How to Run

```bash
# 1. Install dependencies
pip install yfinance pandas numpy scikit-learn xgboost matplotlib seaborn

# 2. Download data
python src/data_loader.py

# 3. Generate features
python src/features_nvda_v1.py

# 4. Train model
python src/models/xgboost_nvda_v1.py
```

---

*Project 1 complete. Moving to Project 2 — Earnings Sentiment Analyzer (FinBERT + NLP).*
