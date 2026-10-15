import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils import black_scholes as bs
from utils.hedging import market_state
from utils.market import simulate_gbm
from utils.models import NoTransactionBandNet
from utils.strategies import (bs_delta, leland_delta, leland_volatility,
                              whalley_wilmott, ww_half_width)
from utils.training import evaluate, train

TAU = 0.5                    # date at which we inspect the band
S_ATM, S_LOW = 100.0, 85.0   # at the money / out of the money

# Figure: run 07_plot_sweep.py (redraws from the CSVs, no retraining).

if __name__ == "__main__":
    dt = cfg.T / cfg.N_STEPS
    test_paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T,
                              cfg.N_STEPS, seed=cfg.TEST_SEED)
    spots = torch.tensor([S_ATM, S_LOW], dtype=torch.float64)
    rows, bands = [], []

    for cost in cfg.COST_GRID:
        print(f"\n=== cost {cost:.2%} ===")
        torch.manual_seed(cfg.TRAIN_SEED)
        model = NoTransactionBandNet(cfg.T)
        train(model, cfg, cost, seed=cfg.TRAIN_SEED, log_every=500)
        torch.save(model.state_dict(), f"data/results/no-transaction_band_{cost}.pt")

        strategies = {
            "BS delta": bs_delta,
            "Leland delta": leland_delta(cost, cfg.SIGMA, cfg.R, cfg.K, dt),
            "Whalley-Wilmott": whalley_wilmott(cost, cfg.RISK_AVERSION, cfg.R),
            "No-transaction band": model,
        }
        for name, strategy in strategies.items():
            stats = evaluate(strategy, test_paths, cfg, cost)
            rows.append({"cost": cost, "strategy": name, **stats})

        with torch.no_grad():
            s = market_state(spots, TAU, cfg.K, cfg.R, cfg.SIGMA)
            lower, upper = model.band(s)
            ww = (ww_half_width(s, cost, cfg.RISK_AVERSION, cfg.R)
                  if cost > 0 else torch.zeros(2))
            leland = bs.call_delta(spots, cfg.K, TAU, cfg.R,
                                   leland_volatility(cfg.SIGMA, cost, dt))
            bands.append({
                "cost": cost,
                "learned_half_width_atm": float((upper[0] - lower[0]) / 2),
                "ww_half_width_atm": float(ww[0]),
                "learned_center_otm": float((upper[1] + lower[1]) / 2),
                "leland_delta_otm": float(leland[1]),
                "bs_delta_otm": float(s.bs_delta[1]),
            })

    results = pd.DataFrame(rows)
    bands = pd.DataFrame(bands)
    results.to_csv("data/results/cost_sweep.csv", index=False)
    bands.to_csv("data/results/cost_sweep_bands.csv", index=False)

    print("\nIndifference price by cost:")
    print(results.pivot(index="cost", columns="strategy", values="price").round(3))
    print("\nBand shape (tau = 0.5):")
    print(bands.round(3).to_string(index=False))