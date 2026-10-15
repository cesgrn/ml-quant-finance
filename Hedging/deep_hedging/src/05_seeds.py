import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils.market import simulate_gbm
from utils.models import FeedForwardHedger, NoTransactionBandNet
from utils.training import evaluate, train

SEEDS = [0, 1, 2, 3, 4]


def iterations_to_reach(history, level, window=20):
    """First iteration where the smoothed training loss goes below `level`."""
    smooth = pd.Series(history).rolling(window).mean().to_numpy()
    below = np.where(smooth < level)[0]
    return int(below[0]) + 1 if len(below) else float("nan")


if __name__ == "__main__":
    cost = cfg.COST
    test_paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS, seed=cfg.TEST_SEED)
    ww_risk = pd.read_csv(f"data/results/comparison_{cost}.csv",
                          index_col="strategy").loc["Whalley-Wilmott", "entropic_risk"]

    rows = []
    for seed in SEEDS:
        for name, Model in [("Feed-forward", FeedForwardHedger), ("No-transaction band", NoTransactionBandNet)]:
            torch.manual_seed(seed)
            model = Model(cfg.T)
            print(f"\nSeed {seed}: {name}")
            history = train(model, cfg, cost, seed=seed, log_every=500)
            stats = evaluate(model, test_paths, cfg, cost)
            rows.append({"seed": seed, "model": name,
                         "risk": stats["entropic_risk"], "price": stats["price"],
                         "cvar_95": stats["cvar_95"], "turnover": stats["turnover"],
                         "iters_to_beat_WW": iterations_to_reach(history, ww_risk)})

    df = pd.DataFrame(rows)
    df.to_csv(f"data/results/seeds_{cost}.csv", index=False)

    print("\nMean and standard deviation over seeds:")
    print(df.drop(columns="seed").groupby("model").agg(["mean", "std"]).round(3))

    risk = df.pivot(index="seed", columns="model", values="risk")
    gap = risk["Feed-forward"] - risk["No-transaction band"]
    print(f"\nBand network better on {int((gap > 0).sum())}/{len(gap)} seeds; "
          f"mean gap {gap.mean():.3f} ± {gap.std():.3f}")