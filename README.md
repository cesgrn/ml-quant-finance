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

**Project**: Derivative Hedging with Transaction Costs, in two parts  
**Part A**: PPO reinforcement learning agent vs Black-Scholes delta hedge ([`Hedging/rl_hedging`](Hedging/rl_hedging))  
**Part B**: Deep hedging with a no-transaction band network, trained directly on the risk of the P&L ([`Hedging/deep_hedging`](Hedging/deep_hedging))  
**Instrument**: Short at-the-money call on a GBM-simulated underlying, 1% transaction cost

Projects 1 and 2 hit the same wall: the market prices in information faster than we can act on it. So we asked a different question entirely. Instead of *predicting* where prices go, what if we focused on *managing risk* given that we cannot predict?

This is the world of derivatives hedging. When you sell a call option, your risk is not whether you predicted the market correctly. It is whether you can offset the option's sensitivity to price movements dynamically. Black-Scholes gives a formula: hold Δ shares per option. It is theoretically perfect in a frictionless, continuous-time world. Real markets are neither: every trade costs money, so rebalancing to Δ at every step is expensive.

**Part A: learning to hedge by trial and error.** We trained a PPO agent to learn a hedging policy from scratch, with no formula, over 200,000 simulated trading steps. Its first evaluation suggested it beat the delta hedge on mean P&L and Sharpe, at the cost of a much worse tail. A second agent with a CVaR-style reward looked even more extreme.

Part B's evaluation code found that those numbers were mostly an artefact of the environment:

- **No financing cost.** Holding shares earned the drift for free, so over-hedging looked profitable.
- **A one-period lag.** Each hedge started earning one step after it was chosen.
- **Different test paths.** The agents and the delta hedge were not compared on the same paths.

Re-evaluated with a correct self-financing P&L on identical paths:
- the first agent's positive mean disappears;
- the CVaR agent turns out not to hedge at all. Its policy collapsed to "never trade", because the per-step cost penalty was a clearer signal than the noisy terminal one.

**Part B: train on the risk itself.** Following *Deep Hedging* (Buehler et al., 2019) and the *No-Transaction Band Network* (Imaki et al., 2021), a small network is trained by gradient descent on the entropic risk of the *whole* P&L, through all 50 hedging decisions at once. The architecture does not output a hedge: it outputs a band around delta, and the position only trades when it leaves the band.

Results on the same 20,000 test paths. "Premium needed" is the indifference price: the premium that makes the dealer's risk zero, so lower is better.

| Strategy | Premium needed | CVaR 95% | Turnover |
|---|---|---|---|
| Black-Scholes delta | 14.50 | -6.00 | 2.78 |
| Leland delta (classical) | 13.58 | -4.81 | 2.55 |
| Whalley-Wilmott band (classical) | 13.59 | -5.05 | 1.35 |
| PPO, part A (re-evaluated) | 32.62 | -17.40 | 2.92 |
| **No-transaction band network** | **12.88** | **-4.21** | 1.59 |

The network beats every classical benchmark at every cost level from 0.1% to 2%, and on 5 out of 5 training seeds.

Looking inside it, it rediscovered both classical ideas at once:
- a Leland-type centre, with delta flattened towards 0.5;
- a Whalley-Wilmott-type no-trade band, but 2.5 to 3.5 times narrower.

**What we learned**: the gap between part A and part B is not "neural networks vs formulas". Both use neural networks. The difference is the training signal. Part A learned from step-by-step rewards and never connected its decisions to the final risk. Part B optimised the final risk directly and was given the right structure (a no-trade band), and it beat the textbook. The second lesson is just as useful: check the accounting before believing a result, especially a profitable one.

> *Predicting the market and managing risk in the market are two entirely different problems. The second one is solvable, if you optimise the right objective.*

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
| 3 | Can ML hedge better than formulas? | Not with off-the-shelf RL. Yes when trained directly on the risk of the full P&L: 12.88 vs 13.58 premium at 1% cost |
| 4 | Can we model what the market believes? | Yes — IV surfaces are learnable, smooth, and forecastable in normal regimes |

The honest answer to our original question — *can machine learning beat the market?* — is: **not easily, not consistently, and not with the approaches most people first try**.

Markets are hard precisely because they are competitive. Every edge gets arbitraged away. The real frontier is not in predicting prices — it is in modeling the structure of uncertainty itself: volatility surfaces, tail risks, regime shifts, and the behavior of markets under stress.

The one chapter where machine learning clearly beat the textbook is chapter 3, and that is not a coincidence. Hedging is not a forecasting problem. It is an optimisation problem with a well-defined objective, and that is where a learned model can do measurably better than a closed-form answer.

That frontier is exactly where the academic papers behind this project live. And it is where we ended up, naturally, after following the evidence through four honest attempts.

---

## Repository Structure

```
ml-quant-finance/
├── stock_return_predictor/     ← Project 1: ML on price data
├── earnings_sentiment/         ← Project 2: NLP on earnings calls
├── Hedging/                    ← Project 3: Hedging with transaction costs
│   ├── rl_hedging/             ←   Part A: PPO reinforcement learning
│   └── deep_hedging/           ←   Part B: deep hedging, no-transaction band network
└── volatility_surface/         ← Project 4: Neural IV surface modeling
```

Each project contains its own detailed `README.md` with full methodology, code, results, and lessons learned.

---

## Key References

- **Gu, Kelly & Xiu (2020)** — Empirical Asset Pricing via Machine Learning *(Review of Financial Studies)*
- **Yang, Uy & Huang (2020)** — FinBERT: A Pretrained Language Model for Financial Communications
- **Buehler, Gonon, Teichmann & Wood (2019)** — Deep Hedging *(Quantitative Finance)*
- **Imaki, Imajo, Ito, Minami & Nakagawa (2021)** — No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging *(arXiv:2103.01775; Journal of Financial Data Science, 2023)*
- **Whalley & Wilmott (1997)** — An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs *(Mathematical Finance)*
- **Horvath, Muguruza & Tomas (2021)** — Deep Learning Volatility *(Quantitative Finance)*
- **Ackerer, Tagasovska & Vatter (2020)** — Deep Smoothing of the Implied Volatility Surface *(NeurIPS)*
- **Loughran & McDonald (2011)** — When Is a Liability Not a Liability? *(Journal of Finance)*

---

## Stack

Python · PyTorch · scikit-learn · XGBoost · stable-baselines3 · HuggingFace Transformers · yfinance · pandas · numpy · scipy · matplotlib

---

*Built as a personal learning project. All results are out-of-sample. Nothing here is financial advice.*