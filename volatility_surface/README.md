# Neural Volatility Surface Forecaster — ML x Finance

Building, smoothing and forecasting the implied volatility surface using neural networks. Built as Project 4 of a broader ML x Quant Finance learning series.

**The core question**: volatility is not constant — it varies across strikes and maturities, forming a surface that evolves every day. Can a neural network learn to smooth this surface while respecting no-arbitrage constraints, and forecast how it moves tomorrow?

---

## Goal

1. Pull real options chain data for SPY via `yfinance`
2. Invert Black-Scholes numerically to compute **implied volatility (IV)** at each strike/expiry
3. Assemble and visualize the raw IV surface in 3D + skew and term-structure slices
4. Train a neural network to **smooth** the raw surface (arbitrage-free, inspired by Ackerer et al.)
5. Train a neural network to **forecast** how the surface evolves day to day (inspired by Horvath et al.)
6. Visualize everything — 3D surface, volatility skew, term structure

---

## Stack

- **Data** : yfinance (options chain)
- **IV computation** : scipy (Brent root-finding for BS inversion)
- **Neural network** : PyTorch
- **Visualization** : matplotlib, plotly (3D surface)
- **Analysis** : numpy, pandas

---

## Key References

- **Horvath, Muguruza & Tomas (2021) — "Deep Learning Volatility"** *(Quantitative Finance)*: trains a neural network offline to approximate the pricing function mapping model parameters to an IV surface treated as an 8×11 pixel image. Achieves 9,000–16,000x speedup vs Monte Carlo. Architecture used here: 4 hidden layers, 30 neurons, ELU activation.
- **Ackerer, Tagasovska & Vatter (2020) — "Deep Smoothing of the Implied Volatility Surface"** *(NeurIPS)*: fits and smooths a noisy IV surface using soft no-arbitrage constraints in the loss function. Two arbitrage conditions enforced: calendar spread (total variance non-decreasing in maturity) and butterfly (surface convex in strike).

---

## Project Structure

```
volatility_surface/
├── data/
│   ├── raw/
│   │   ├── spy_options_raw.csv       ← full options chain from yfinance
│   │   └── spy_options_liquid.csv    ← after liquidity filters (3,038 options)
│   └── processed/
│       ├── iv_surface_raw.csv        ← computed implied volatilities
│       ├── iv_grid.csv               ← IV pivot table (25 maturities × 60 strikes)
│       ├── vix_term_structure.csv    ← historical VIX term structure (2018-2025)
│       ├── iv_smoother.pt            ← trained smoother weights (excluded from git)
│       ├── iv_forecaster.pt          ← trained forecaster weights (excluded from git)
│       └── plots/
│           ├── raw_surface_3d.png
│           ├── smoothed_surface_3d.png
│           ├── skew_slices.png
│           ├── term_structure.png
│           ├── smoother_training.png
│           ├── forecast_vs_actual.png
│           └── forecaster_training.png
├── src/
│   ├── utils/
│   │   ├── black_scholes.py          ← BS pricing + IV inversion (Brent)
│   │   └── arbitrage_checks.py       ← calendar + butterfly arbitrage penalties
│   ├── 01_collect_data.py            ← fetch options chain from yfinance
│   ├── 02_compute_iv.py              ← Black-Scholes inversion → IV surface
│   ├── 03_visualize_surface.py       ← 3D plots + skew + term structure
│   ├── 04_smooth_surface.py          ← NN smoother with no-arbitrage constraints
│   └── 05_forecast_surface.py        ← NN surface forecaster
├── .gitignore
└── README.md
```

---

## Background — The Implied Volatility Surface

### What is implied volatility?

Black-Scholes gives a formula for option prices given a volatility $\sigma$. In reality, we observe option prices in the market and **invert** this formula to find the $\sigma$ that makes Black-Scholes match the market price. This is the **implied volatility** (IV).

$$C_{market}(K, T) = BS(\sigma_{IV}(K,T), S, K, T, r)$$

### Why does the surface matter?

If Black-Scholes were perfect, $\sigma_{IV}$ would be the same for all strikes and maturities — a flat surface. In reality:
- **Volatility skew**: OTM puts have higher IV than ATM options (fear of crashes)
- **Term structure**: short-dated options often have higher IV than long-dated (event risk)
- **Smile**: IV is higher for deep ITM and OTM options than ATM

The shape of this surface encodes everything the market believes about future uncertainty.

### No-arbitrage constraints (Ackerer et al.)

A valid IV surface must satisfy:
- **Calendar spread**: $\sigma^2(K,T_1) \cdot T_1 \leq \sigma^2(K,T_2) \cdot T_2$ for $T_1 < T_2$ — total variance non-decreasing
- **Butterfly**: the surface must be convex in strike — otherwise a butterfly spread would be riskless

---

## Methodology

### Step 1 — Data Collection
Fetch SPY options chain via yfinance for multiple expiry dates.

### Step 2 — IV Computation
Numerically invert Black-Scholes using Brent's method for each (strike, expiry) pair.

### Step 3 — Surface Visualization
Assemble and plot the raw IV surface in 3D, skew slices per maturity, and term structure at ATM.

### Step 4 — Neural Network Smoother
Inspired by Ackerer et al.: train a network to fit the surface while penalizing arbitrage violations.

$$L = MSE + \lambda_1 \cdot \text{calendar penalty} + \lambda_2 \cdot \text{butterfly penalty}$$

### Step 5 — Surface Forecaster
Inspired by Horvath et al.: train a network to predict tomorrow's IV surface given today's.

---

## Results

### Step 1 — Data Collection ✅

Fetched SPY options chain via yfinance.

- **Spot price** : $745.64
- **Total options** : 7,567 across 34 expiries
- **After liquidity filters** : 3,038 options across 25 expiries
- **Strike range** : $523 — $965 (70%–130% moneyness)
- **DTE range** : 6 — 387 days

Liquidity filters applied: calls only, bid > 0, moneyness 0.7–1.3, bid-ask spread < 50% of mid.

### Step 2 — IV Computation ✅

Inverted Black-Scholes numerically using Brent's method on each (strike, expiry) pair. Used mid price `(bid+ask)/2` rather than last traded price to avoid stale data.

| Metric | Value |
|---|---|
| Valid IVs computed | 2,926 / 3,038 (96.5%) |
| Mean IV | 19.48% |
| Min IV | 12.38% |
| Max IV | 112.65% |
| Std IV | 8.63% |
| Surface grid shape | 25 maturities × 60 moneyness levels |

### Step 3 — Surface Visualization ✅

Implemented `src/03_visualize_surface.py`. Three plots produced.

**Term Structure (ATM IV vs maturity):**
Normal upward-sloping structure — ~15% at 6 days, ~18% at 387 days. Slight dip around 35-50 days where no major events are expected. Consistent with a calm short-term market and growing uncertainty at longer horizons.

**Volatility Skew (IV vs moneyness by maturity):**
Classic left skew:
- 10d skew: drops from ~100% at K/S=0.7 to ~16% ATM — extreme crash protection premium
- Skew flattens significantly with maturity — 387d curve nearly flat
- Slight OTM call smile visible at short maturities (K/S > 1.0)

**3D Surface:**
Correct overall shape — red peak at deep OTM short-dated, flat green surface at ATM long-dated. However, **negative IV values appear at the edges** — this is a cubic interpolation artifact extrapolating outside the data range. IV cannot be negative.

### Step 4 — Neural Network Smoother ✅

Implemented `src/04_smooth_surface.py`. Architecture inspired by Horvath et al. (2021):
- 4 hidden layers × 30 neurons
- **ELU activation** (smooth, $C^\infty$ — required for arbitrage derivative computations)
- **Softplus output** — guarantees IV > 0 everywhere (eliminates the negative IV artifact)

**Loss function** (Ackerer et al.):
$$L = MSE + \lambda_1 \cdot \text{calendar penalty} + \lambda_2 \cdot \text{butterfly penalty}$$

With $\lambda_1 = 10$, $\lambda_2 = 5$.

**Results:**

| Metric | Value |
|---|---|
| Initial loss | 0.2290 |
| Final loss | 0.00155 |
| Loss reduction | 99.3% |
| Final RMSE | **3.94% IV units** |
| Arbitrage penalties | 0.000 (both calendar and butterfly) |

**Key finding**: the ELU + Softplus architecture naturally produces an arbitrage-free surface — no constraint violations were triggered during training. The Softplus output layer alone eliminated the main problem from Step 3 (negative IV from cubic interpolation).

**Before vs After:**
- Raw surface: negative IV artifacts at edges, discontinuities, noisy
- Smoothed surface: clean, positive everywhere, smooth skew preserved, natural transition from ~70% (OTM short-dated) to ~15% (ATM long-dated)

### Step 5 — Surface Forecaster ✅

Implemented `src/05_forecast_surface.py`. Used historical VIX term structure (2018–2025) as proxy for ATM IV across maturities. `^VIX1Y` delisted — used 3 maturities instead.

**Data**: 2,010 trading days, 3 maturities (30d, 90d, 180d). Train/test split 80/20.

**Architecture**: MLP with 4 hidden layers, ELU activation, Sigmoid output. Input: last 20 days × 3 maturities = 60 features. Output: next day's 3-point term structure.

**Forecast Accuracy (RMSE):**

| Maturity | RMSE | Why |
|---|---|---|
| 30d (VIX) | **2.37%** | Short-term vol is noisy and event-driven |
| 90d | **1.53%** | Less noise, smoother dynamics |
| 180d | **1.15%** | Long-term vol is slow-moving and mean-reverting |
| **Overall** | **1.76%** | Strong result across all maturities |

> The model tracks the actual term structure closely in calm markets. During the COVID spike the 30d forecast undershoots (~40% predicted vs ~52% actual) — neural network forecasters cannot anticipate unprecedented shocks from historical patterns. Performance improves with maturity, consistent with lower noise-to-signal ratio of long-dated volatility.

---

## Final Conclusions

| Step | What we built | Key result |
|---|---|---|
| Data collection | SPY options chain via yfinance | 3,038 liquid options, 25 expiries |
| IV computation | Black-Scholes inversion (Brent) | 96.5% success rate, mean IV 19.5% |
| Visualization | 3D surface + skew + term structure | Classic left skew, normal term structure |
| NN Smoother | 4-layer ELU+Softplus network | RMSE 3.94%, arbitrage-free |
| NN Forecaster | MLP on VIX term structure | Overall RMSE 1.76% |

**What the IV surface tells us about the market (May 2026):**
- ATM vol ~15–16% — calm market
- Strong left skew — crash protection premium
- Normal term structure — long-term uncertainty priced higher than short-term

---

## Key Lessons

- **IV is not constant** — the surface encodes collective market beliefs about future uncertainty
- **Cubic interpolation fails at boundaries** — neural networks with Softplus output solve this naturally
- **ELU over ReLU in finance** — smooth activation required for arbitrage penalty derivatives
- **Long-term vol is more predictable** — RMSE improves from 2.37% (30d) to 1.15% (180d)
- **Neural networks cannot predict black swans** — COVID spike systematically underestimated

---

## How to Run

```bash
# 1. Install dependencies
pip install yfinance numpy scipy matplotlib plotly torch scikit-learn

# 2. Run pipeline in order
python src/01_collect_data.py
python src/02_compute_iv.py
python src/03_visualize_surface.py
python src/04_smooth_surface.py
python src/05_forecast_surface.py
```

---

## Connection to Previous Projects

| Project | Question | Finding |
|---|---|---|
| 1 — Stock Return Predictor | Can we predict prices? | Markets too efficient for technical features |
| 2 — Earnings Sentiment | Does NLP add edge? | Signal exists but vanishes at scale |
| 3 — RL Hedging | How to manage risk without predicting? | RL beats delta hedge on Sharpe but standard PPO fails on CVaR |
| **4 — Volatility Surface** | **What does the market expect about future volatility?** | **The IV surface encodes collective market uncertainty** |

---

*Project 4 complete. All 4 projects done — ready for the global README.*