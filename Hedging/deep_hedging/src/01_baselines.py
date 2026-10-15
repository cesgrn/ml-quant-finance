"""
Step 1 — Classic benchmarks under correct self-financing accounting.

Evaluates three non-learned strategies on the same 20,000 test paths:
  * no hedge
  * Black-Scholes delta hedge (rebalanced at every date)
  * Whalley-Wilmott no-transaction band

for every transaction-cost level in COST_GRID, and saves:
  data/results/baselines.csv
  data/results/baselines_pnl.png   (P&L histograms at the part-A cost of 1%)

Run from the deep_hedging folder:  python src/01_baselines.py
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from utils.hedging import simulate_hedge
from utils.market import simulate_gbm
from utils.risk import summary
from utils.strategies import bs_delta, no_hedge, whalley_wilmott

if __name__ == "__main__":
    os.makedirs("data/results", exist_ok=True)

    paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T,
                         cfg.N_STEPS, seed=cfg.TEST_SEED)
    market = dict(k=cfg.K, T=cfg.T, r=cfg.R, sigma=cfg.SIGMA)

    rows, pnl_at_part_a_cost = [], {}
    for cost in cfg.COST_GRID:
        strategies = {
            "No hedge": no_hedge,
            "BS delta": bs_delta,
            "Whalley-Wilmott": whalley_wilmott(cost, cfg.RISK_AVERSION, cfg.R),
        }
        for name, strategy in strategies.items():
            pnl, hedges = simulate_hedge(paths, strategy, cost=cost, **market)
            stats = summary(pnl, cfg.CVAR_LEVEL, cfg.RISK_AVERSION)
            stats["avg_turnover"] = float(
                (hedges.diff(dim=1).abs().sum(dim=1) + hedges[:, 0].abs()).mean())
            rows.append({"strategy": name, "cost": cost, **stats})
            if cost == cfg.COST:
                pnl_at_part_a_cost[name] = pnl.numpy()

    df = pd.DataFrame(rows)
    df.to_csv("data/results/baselines.csv", index=False)

    pd.set_option("display.width", 140)
    print(f"\nTest set: {cfg.N_TEST_PATHS:,} paths (seed {cfg.TEST_SEED}), "
          f"P&L in time-0 dollars, self-financing accounting\n")
    for cost, block in df.groupby("cost"):
        print(f"--- transaction cost = {cost:.2%} ---")
        print(block.drop(columns="cost").set_index("strategy").round(3).to_string(), "\n")

    # Histograms at the part-A cost level
    fig, ax = plt.subplots(figsize=(10, 5))
    # "No hedge" is left out: its P&L spreads from -40 to +10 and hides the others
    for name in ["BS delta", "Whalley-Wilmott"]:
        ax.hist(pnl_at_part_a_cost[name], bins=120, range=(-12, 4), alpha=0.55, label=name)
    ax.set_title(f"Terminal P&L of a hedged short ATM call, cost = {cfg.COST:.0%}")
    ax.set_xlabel("P&L (time-0 dollars)")
    ax.set_ylabel("Number of paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig("data/results/baselines_pnl.png", dpi=150)
    print("Saved data/results/baselines.csv and data/results/baselines_pnl.png")
