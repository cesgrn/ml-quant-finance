# ML x Quant Finance — A Learning Journey

> *Can machine learning beat the market? This repository documents an honest attempt to find out — and what we discovered along the way.*

---

## The Story

This project started with a simple question that every quant researcher eventually asks: **can we use machine learning to make money in financial markets?**

We did not start with the answer. We started with data, code, and a willingness to follow the evidence wherever it led. What emerged was not a trading system that prints money — it was something more valuable: a deep understanding of *why* markets are hard, *what* machine learning can and cannot do in finance, and *where* the real frontier of quantitative research actually sits.

Four projects. Four approaches. One honest conclusion per chapter.

---

## Chapter 1 — Can We Predict Stock Prices?

**Project**: Stock Return Predictor  
**Approach**: Random Forest and XGBoost on technical features (momentum, RSI, volatility, VIX)  
**Ticker**: SPY (S&P 500 ETF)

We started where everyone starts: price data. The idea was straightforward — if the stock went up yesterday and momentum is positive, maybe it goes up tomorrow too. We engineered 10 features, trained several models, iterated through confidence thresholds and VIX regime filters.

The market was not impressed.

Our best model — XGBoost with VIX regime filtering — achieved a total return of **-0.38%** on the test period versus **+68.82%** for Buy & Hold. We reduced the max drawdown from -27% to -2%, which showed the model *learned something* — but not enough to beat doing nothing.

Then we tried NVDA, thinking a more volatile, less efficient stock might give us an edge. NVDA returned **+353%** over the same period. Our model returned **-60%**. NVDA's rally was driven entirely by AI earnings surprises — CUDA announcements, data center contracts — none of which appears in a price chart.

**What we learned**: SPY is too efficient for technical features. NVDA is event-driven. In both cases, the information that actually moves prices is not in historical prices.

> *The market already knows what you know. The edge lies in what it doesn't know yet.*

---

## Chapter 2 — What If We Read the News?

**Project**: Earnings Sentiment Analyzer  
**Approach**: FinBERT (transformer NLP) on earnings call transcripts + Loughran-McDonald dictionary baseline  
**Tickers**: AAPL, MSFT, NVDA, GOOGL, META

Project 1 taught us that price features are insufficient. The missing ingredient was *information* — the kind that moves NVDA 20% in a day. So we went to the source: earnings calls. Every quarter, CEOs and CFOs speak directly to investors and analysts for an hour. The language they use — how optimistic, how hedged, how precise — should contain signal.

We built the full pipeline. Scraped real transcripts from Motley Fool. Scored them with FinBERT, a BERT model trained on 4.9 billion tokens of financial text (SEC filings, analyst reports, earnings calls). Compared it against the Loughran-McDonald dictionary — the classic word-count approach.

**The 5-transcript pilot looked promising**: a t+1 correlation of **+0.662** between sentiment score and next-day return. AAPL (highest sentiment: +0.188) outperformed. MSFT (lowest sentiment: +0.108) dropped -6%. The model seemed to see something real.

Then we scaled to 78 transcripts. The correlation collapsed to **+0.002**.

The pilot was noise. With 5 data points, any correlation is meaningless. A critical lesson: *small samples lie in finance more than anywhere else*.

The FinBERT vs Loughran-McDonald comparison was revealing in a different way. Their correlation was only **0.177** — they measure fundamentally different things. FinBERT understands that *"we are not concerned about growth"* is positive. The dictionary scores it negative because "concerned" appears. Context matters — and transformers capture it, dictionaries do not.

**What we learned**: sentiment is real but the signal requires scale — hundreds of transcripts, not five. And the market is fast: it often prices in earnings tone before the call ends.

> *NLP sees beyond price data. But seeing is not the same as acting faster than everyone else who is also looking.*

---

## Chapter 3 — Stop Predicting. Start Managing Risk.

**Project**: RL Derivative Hedging  
**Approach**: PPO reinforcement learning agent vs Black-Scholes delta hedge  
**Instrument**: Short call option on GBM-simulated underlying

Projects 1 and 2 hit the same wall: the market prices in information faster than we can act on it. So we asked a different question entirely. Instead of *predicting* where prices go, what if we focused on *managing risk* given that we cannot predict?

This is the world of derivatives hedging. When you sell a call option, your risk is not whether you predicted the market correctly — it is whether you can offset the option's sensitivity to price movements dynamically. Black-Scholes gives a formula: hold Δ shares per option. It is theoretically perfect in a frictionless, continuous-time world. Real markets are neither.

We trained a PPO agent to learn a hedging policy from scratch — no formula, just trial and error across 200,000 simulated trading episodes.

**The results were instructive:**

| Strategy | Mean PnL | Sharpe | CVaR 95% |
|---|---|---|---|
| Delta Hedge | -0.21 | -0.08 | **-6.27** |
| RL v1 (variance reward) | +0.38 | +0.05 | -15.59 |
| RL v2 (CVaR reward) | +2.67 | +0.23 | **-31.52** |

The RL agent beat the delta hedge on Sharpe and mean PnL. But its tail risk was catastrophic — 2.5x worse CVaR than the textbook formula. The agent had found a clever trick: take aggressive positions that make money on average while occasionally blowing up.

This is not hedging. This is gambling with a positive expected value.

The deeper lesson came from trying to fix it. We redesigned the reward to penalize CVaR at the episode level — exactly what the Deep Hedging paper recommends. It made things worse. The agent's CVaR exploded to -31.52. Why? The *credit assignment problem*: the agent receives the terminal CVaR penalty at step 50, but cannot connect it back to each of the 50 individual decisions that led there.

Standard PPO cannot optimize episode-level risk objectives. The Deep Hedging paper (Buehler et al., 2019) solves this with direct gradient descent on the full trajectory — a fundamentally different approach. That is not a flaw in our implementation. That is the research frontier.

**What we learned**: RL can find profitable hedging strategies, but controlling tail risk requires custom algorithms that treat the entire trajectory as a single optimization problem — not a sequence of independent decisions.

> *Predicting the market and managing risk in the market are two entirely different problems. The second one is harder.*

---

## Chapter 4 — What Does the Market Believe?

**Project**: Neural Volatility Surface Forecaster  
**Approach**: Black-Scholes IV inversion + neural network smoother + term structure forecaster  
**Data**: SPY options chain (real market data) + VIX historical time series

After three chapters of trying to extract alpha from markets, we stepped back and asked a more fundamental question: *what does the options market collectively believe about the future?*

The implied volatility surface is the answer. When you observe option prices across all strikes and expiries and invert Black-Scholes, you get a surface that encodes the market's probability distribution of future returns — not what it knows, but what it collectively *fears and expects*.

We built this surface from scratch. Fetched 3,038 liquid SPY options. Inverted Black-Scholes numerically using Brent's method for each (strike, expiry) pair — 96.5% success rate. The raw surface told a clear story: a pronounced left skew (OTM puts trading at ~100% IV for 10-day maturities vs ~16% ATM), a normal term structure (vol rising from ~15% at 6 days to ~18% at 387 days), and the classic pattern of fear asymmetry — investors pay far more to protect against crashes than to bet on rallies.

But the raw surface had a problem: cubic interpolation produced negative IV values at the boundaries. Mathematically absurd, financially meaningless.

We trained a neural network smoother, directly inspired by Ackerer et al. (2020): a 4-layer network with **ELU activation** (smooth, differentiable — required for arbitrage constraint gradients) and **Softplus output** (guarantees IV > 0 everywhere). The result: **RMSE of 3.94%**, zero arbitrage violations, no negative values. The Softplus alone fixed what cubic interpolation could not.

For the forecaster, we trained an MLP on 8 years of VIX term structure history to predict tomorrow's IV from the past 20 days. Overall **RMSE of 1.76%** — improving from 2.37% at 30 days to 1.15% at 180 days. Long-dated volatility mean-reverts slowly and is far more predictable than the VIX.

One sobering result: during the COVID spike, the 30d forecast predicted ~40% while actual reached ~52%. No model trained on historical data can anticipate an unprecedented shock. This is not a failure of implementation — it is a fundamental constraint of learning from the past.

**What we learned**: the IV surface is one of the richest sources of information in financial markets. It is not a prediction — it is a consensus. Neural networks can smooth and forecast it with good accuracy in normal regimes. Black swans remain unpredictable by definition.

> *The options market does not know the future. But it has thought harder about it than anyone.*

---

## The Honest Conclusion

| Chapter | Question | Answer |
|---|---|---|
| 1 | Can technical ML predict prices? | No — markets price in public information too fast |
| 2 | Can NLP on earnings calls add edge? | Signal exists but vanishes at scale; context matters |
| 3 | Can RL optimize hedging better than formulas? | Yes on average returns, no on tail risk with standard algorithms |
| 4 | Can we model what the market believes? | Yes — IV surfaces are learnable, smooth, and forecastable in normal regimes |

The honest answer to our original question — *can machine learning beat the market?* — is: **not easily, not consistently, and not with the approaches most people first try**.

Markets are hard precisely because they are competitive. Every edge gets arbitraged away. The real frontier is not in predicting prices — it is in modeling the structure of uncertainty itself: volatility surfaces, tail risks, regime shifts, and the behavior of markets under stress.

That frontier is exactly where the academic papers behind this project live. And it is where we ended up, naturally, after following the evidence through four honest attempts.

---

## Repository Structure

```
ml-quant-finance/
├── stock_return_predictor/     ← Project 1: ML on price data
├── earnings_sentiment/         ← Project 2: NLP on earnings calls
├── rl_hedging/                 ← Project 3: Reinforcement learning for hedging
└── volatility_surface/         ← Project 4: Neural IV surface modeling
```

Each project contains its own detailed `README.md` with full methodology, code, results, and lessons learned.

---

## Key References

- **Gu, Kelly & Xiu (2020)** — Empirical Asset Pricing via Machine Learning *(Review of Financial Studies)*
- **Yang, Uy & Huang (2020)** — FinBERT: A Pretrained Language Model for Financial Communications
- **Buehler, Gonon, Teichmann & Wood (2019)** — Deep Hedging *(Quantitative Finance)*
- **Horvath, Muguruza & Tomas (2021)** — Deep Learning Volatility *(Quantitative Finance)*
- **Ackerer, Tagasovska & Vatter (2020)** — Deep Smoothing of the Implied Volatility Surface *(NeurIPS)*
- **Loughran & McDonald (2011)** — When Is a Liability Not a Liability? *(Journal of Finance)*

---

## Stack

Python · PyTorch · scikit-learn · XGBoost · stable-baselines3 · HuggingFace Transformers · yfinance · pandas · numpy · scipy · matplotlib

---

*Built as a personal learning project. All results are out-of-sample. Nothing here is financial advice.*