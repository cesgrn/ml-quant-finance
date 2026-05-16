# RL Derivative Hedging — ML x Finance

Training a reinforcement learning agent to dynamically hedge a short call option portfolio. Built as Project 3 of a broader ML x Quant Finance learning series.

**The core question**: if we cannot predict markets (Project 1) and sentiment signals disappear at scale (Project 2), how do we manage risk as efficiently as possible? RL hedging is the answer — instead of predicting price direction, we learn an optimal hedging policy that minimizes P&L variance and transaction costs.

---

## Goal

1. Simulate a short call option portfolio under geometric Brownian motion
2. Implement a **delta-hedge baseline** (textbook Black-Scholes)
3. Train an **RL agent** (PPO) to dynamically adjust hedge ratios
4. Design a reward function that penalizes both P&L variance and transaction costs
5. Compare RL vs delta-hedge across different volatility regimes

---

## Stack

- **Simulation** : numpy, scipy (Black-Scholes pricing)
- **RL Framework** : stable-baselines3 (PPO), gymnasium
- **Analysis** : pandas, numpy, matplotlib
- **Deep Learning** : PyTorch (policy network backend)

---

## Key References

- **Buehler, Gonon, Teichmann & Wood (2019) — "Deep Hedging"** *(Quantitative Finance)*: the landmark paper. Shows that RL-based hedging outperforms delta hedging under realistic market conditions (transaction costs, discrete rebalancing, model uncertainty). Required reading before building this project.
- **Kolm & Ritter (2019) — "Dynamic Replication and Hedging: A Reinforcement Learning Approach"** *(Journal of Financial Data Science)*: the clearest practical write-up of how to set up the RL hedging problem — state space, action space, reward design.

---

## Project Structure

```
rl_hedging/
├── data/
│   ├── gbm_paths.png               ← 50 simulated GBM price paths
│   └── results/
│       ├── delta_hedge_pnl.png     ← P&L distribution of delta hedge
│       ├── hedge_comparison.png    ← RL v1 vs delta hedge comparison
│       ├── full_comparison.png     ← 3-way comparison plot
│       ├── training_curve.png      ← PPO v1 learning curve
│       └── training_curve_cvar.png ← PPO v2 CVaR learning curve
│       (*.zip model files excluded from git — regenerate with 03/05)
├── src/
│   ├── utils/
│   │   ├── black_scholes.py        ← BS pricing + greeks
│   │   └── simulation.py           ← GBM price path simulation
│   ├── option_env.py               ← option environment v1 (variance reward)
│   ├── option_env_cvar.py          ← option environment v2 (CVaR reward)
│   ├── 02_delta_hedge.py           ← Black-Scholes delta hedge baseline
│   ├── 03_train_rl.py              ← PPO v1 agent training
│   ├── 04_evaluate.py              ← 3-way comparison evaluation
│   └── 05_train_cvar.py            ← PPO v2 CVaR agent training
├── .gitignore
└── README.md
```

---

## Background — Why RL for Hedging?

### The classic approach: Delta Hedging
Black-Scholes tells us the **delta** (Δ) of an option — the sensitivity of the option price to the underlying price. A delta-hedged portfolio holds Δ shares of the underlying for every short call, rebalancing continuously.

**Problem**: real markets have transaction costs, discrete rebalancing, and the underlying does not follow perfect GBM. Delta hedging ignores all of this.

### The RL approach
Instead of following a fixed formula, an RL agent learns a **policy** — a mapping from market state to hedge ratio — by maximizing a reward that explicitly penalizes:
- P&L variance (risk)
- Transaction costs (realism)

The agent learns to trade off between these competing objectives, something Black-Scholes cannot do.

---

## Methodology

### Step 1 — Option Environment
Simulate a short call option portfolio:
- Underlying follows Geometric Brownian Motion (GBM)
- Option priced via Black-Scholes
- Agent observes: current price, time to expiry, current delta, current position
- Agent acts: choose hedge ratio (continuous action space)

### Step 2 — Delta Hedge Baseline
Implement the textbook Black-Scholes delta hedge as a benchmark.

### Step 3 — RL Agent Training
Train a PPO agent with reward:

$$r_t = -\text{P\&L variance} - \lambda \cdot |\Delta \text{position}| \cdot \text{transaction cost}$$

Where $\lambda$ controls the transaction cost penalty.

### Step 4 — Evaluation
Compare RL vs delta hedge on:
- P&L distribution (mean, std, VaR)
- Sharpe ratio of the hedged portfolio
- Performance across low/medium/high volatility regimes

---

## Results

### Step 1 — Black-Scholes Utilities ✅

Implemented `src/utils/black_scholes.py` with price, delta and gamma functions.

Sanity check on ATM call (S=100, K=100, T=1, r=5%, σ=20%):

| Metric | Value |
|---|---|
| Call Price | 10.4506 |
| Delta | 0.6368 |
| Gamma | 0.0188 |

> Delta slightly above 0.5 because the risk-free drift pushes the call to be in-the-money in expectation. Gamma is low at T=1 year — it explodes as expiry approaches, which is precisely what makes hedging expensive near maturity.

### Step 2 — GBM Price Simulation ✅

Implemented `src/utils/simulation.py` using the exact GBM discretization:

$$\log\left(\frac{S_{t+dt}}{S_t}\right) = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma\sqrt{dt}\cdot Z, \quad Z \sim \mathcal{N}(0,1)$$

The $-\frac{\sigma^2}{2}$ term is the **Itô correction** — without it, the expected price would drift too high due to the convexity of the exponential.

Sanity check (S0=100, μ=5%, σ=20%, T=1 year, 252 steps, 1000 paths):

| Metric | Value | Expected |
|---|---|---|
| Shape | (1000, 253) | 1000 paths × 252 steps + initial |
| Initial price | 100.00 | S0 = 100 ✓ |
| Final price mean | 105.09 | S0 × e^μT ≈ 105.13 ✓ |
| Final price std | 20.77 | σ × S0 × √T = 20.00 ✓ |

### Step 3 — Option Hedging Environment ✅

Implemented `src/01_environment.py` — a Gymnasium environment simulating a short call option portfolio.

**State space** (4 features observed by the agent at each step):

| Feature | Description |
|---|---|
| `stock_price_norm` | Current price normalized by S0 |
| `time_remaining` | Fraction of option lifetime remaining [1→0] |
| `bs_delta` | Theoretical Black-Scholes delta — provided as a hint |
| `current_hedge` | Agent's current hedge position |

**Action space**: continuous hedge ratio in [0, 1].

**Reward function**: $r_t = -\text{P\&L}_t^2 - \lambda \cdot |\Delta\delta| \cdot S_t$

Penalizes P&L variance (squared) and transaction costs simultaneously — consistent with the Deep Hedging paper's convex risk measure framework.

Sanity check (random actions, 1 episode):

| Metric | Value |
|---|---|
| Initial bs_delta | 0.6368 |
| Final stock price | 108.34 |
| Total P&L (random) | -6.27 |

> Negative P&L with random actions is expected — the RL agent will learn to do better.

### Step 4 — Delta Hedge Baseline ✅

Implemented `src/02_delta_hedge.py` — the textbook Black-Scholes delta hedge strategy. At each step, the hedge ratio is set exactly equal to the theoretical BS delta. No learning involved — this is our benchmark.

**Results (1000 episodes, 50 rebalancing steps, 1% transaction cost):**

| Metric | Value | Interpretation |
|---|---|---|
| Mean P&L | -0.1358 | Slightly negative due to transaction costs |
| Std P&L | 2.6437 | Residual risk from discrete rebalancing |
| Sharpe | -0.0514 | Near zero — delta hedge neutralizes risk, not generates profit |
| VaR 95% | -4.6623 | Worst 5% of episodes lose more than 4.66 |
| **CVaR 95%** | **-6.1243** | **Average loss in worst 5% — baseline to beat** |

> The slightly negative mean P&L is entirely explained by transaction costs — in a frictionless market with continuous rebalancing, delta hedge P&L would be exactly 0. The residual std of 2.64 comes from discrete rebalancing (50 steps instead of continuous).

> **These are the numbers the RL agent must beat.**

### Step 5 — PPO Agent Training ✅

Implemented `src/03_train_rl.py`. Trained a PPO agent for 200,000 timesteps across 4 parallel environments.

**Architecture (MlpPolicy):**
- Input layer: 4 features
- 2 hidden layers × 64 units (Tanh activation)
- Output: 1 continuous action (hedge ratio)
- Separate actor and critic networks with identical architecture

**Training progression:**

| Timesteps | Mean Reward | Key Signal |
|---|---|---|
| 8,192 | -164 | Random exploration |
| 65,536 | -142 | Starting to learn |
| 114,688 | -66 | Clear improvement |
| 163,840 | -40 | Convergence zone |
| **204,800** | **-36** | **Final policy** |

**Convergence indicators:**
- `std` (policy): 0.97 → 0.39 — agent converges from exploration to a precise strategy
- `explained_variance`: 0.002 → 0.70 — critic learns to predict returns accurately
- `value_loss`: 1730 → 38 — critic error drops massively
- `clip_fraction`: stayed low (~0.05) — PPO updates remained conservative and stable

> The reward is always negative by design ($r = -\text{P\&L}^2 - \text{tc}$). A perfect hedge would give reward = 0. The agent improved from -164 to -36 — a 78% improvement.

### Step 6 — Evaluation: RL Agent vs Delta Hedge ✅

Implemented `src/04_evaluate.py`. Evaluated both strategies on 1000 out-of-sample episodes (different seed than training).

| Metric | Delta Hedge | RL Agent | Better? |
|---|---|---|---|
| Mean P&L | -0.2118 | **+0.3823** | RL ↑ |
| Std P&L | 2.6104 | 7.0613 | DH ↑ |
| Sharpe | -0.0811 | **+0.0541** | RL ↑ |
| VaR 95% | -4.7335 | -12.0077 | DH ↑ |
| **CVaR 95%** | **-6.2710** | **-15.5933** | **DH ↑** |

**Analysis:**

The RL agent achieves a positive mean P&L (+0.38) and positive Sharpe (+0.054) — it learned to extract value from the market. However, its tail risk is dramatically worse: CVaR of -15.59 vs -6.27 for the delta hedge.

**Why this happens — the key lesson:**

The agent optimized $r = -\text{P\&L}^2 - \text{tc}$ and found an unintended strategy: take aggressive positions that generate positive average P&L while accepting large occasional losses. The reward penalizes step-level variance but does not control tail risk (CVaR).

This is precisely the problem highlighted in the Deep Hedging paper — **reward design is critical**. The paper uses CVaR as the global episode-level loss function rather than a step-level quadratic penalty, which explicitly controls tail risk.

### Step 7 — CVaR Reward Agent ✅

Implemented `src/option_env_cvar.py` and `src/05_train_cvar.py`.

**Reward design change:**
- V1: $r_t = -\text{P\&L}_t^2 - \text{tc}_t$ at every step
- V2: $r_t = -\text{tc}_t$ at intermediate steps, $r_T = \text{P\&L}_{total} - \lambda \cdot \max(0, -\text{P\&L}_{total})^2$ at terminal step

**Full 3-way comparison (1000 out-of-sample episodes):**

| Metric | Delta Hedge | RL v1 (variance) | RL v2 (CVaR) |
|---|---|---|---|
| Mean P&L | -0.2118 | +0.3823 | **+2.6685** |
| Std P&L | 2.6104 | 7.0613 | 11.6753 |
| Sharpe | -0.0811 | +0.0541 | **+0.2286** |
| VaR 95% | -4.7335 | -12.0077 | -22.9659 |
| **CVaR 95%** | **-6.2710** | **-15.5933** | **-31.5244** |

**Why the CVaR reward failed — the credit assignment problem:**

The terminal CVaR penalty is given at step 50 only. PPO cannot connect this terminal signal back to each of the 50 individual hedging decisions. The agent ignores the tail risk penalty and maximizes mean P&L instead.

This is precisely why the Deep Hedging paper does not use PPO — it uses **direct gradient descent** on the full episode CVaR, treating the entire trajectory as a single differentiable computation.

---

## Final Conclusions

| Strategy | Mean P&L | Sharpe | CVaR 95% | Verdict |
|---|---|---|---|---|
| Delta Hedge | -0.21 | -0.08 | **-6.27** | ✅ Best tail risk |
| RL v1 (variance) | +0.38 | +0.05 | -15.59 | ⚠️ Higher profit, worse tail |
| RL v2 (CVaR) | +2.67 | **+0.23** | -31.52 | ❌ Best Sharpe, worst tail |

- Standard RL (PPO) beats delta hedge on mean P&L and Sharpe but worsens tail risk
- Episode-level CVaR objectives require custom training algorithms, not standard PPO
- The Deep Hedging paper's key contribution is solving this with direct gradient descent on the risk measure

---

## Key Hypothesis

> An RL agent that explicitly optimizes for P&L variance and transaction costs will outperform a textbook delta hedge in realistic market conditions — especially when transaction costs are non-negligible or volatility is misspecified.

---

## How to Run

```bash
# 1. Install dependencies
pip install numpy pandas scipy matplotlib torch stable-baselines3 gymnasium

# 2. Run pipeline in order
python src/option_env.py        # test the simulation environment
python src/02_delta_hedge.py    # run delta hedge baseline
python src/03_train_rl.py       # train PPO agent
python src/05_train_cvar.py     # train PPO v2 CVaR agent
python src/04_evaluate.py       # 3-way comparison
```

---

## Connection to Previous Projects

| Project | Question | Finding |
|---|---|---|
| 1 — Stock Return Predictor | Can we predict prices? | Markets are too efficient for technical features |
| 2 — Earnings Sentiment | Does NLP add edge? | Signal exists but vanishes at scale |
| **3 — RL Hedging** | **How to manage risk without predicting?** | **RL beats delta hedge on Sharpe but standard PPO fails on CVaR — custom algorithms needed** |

---

*Project 3 complete. Moving to Project 4 — Neural Volatility Surface Forecaster.*