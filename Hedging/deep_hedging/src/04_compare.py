import os
import sys
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils.market import simulate_gbm
from utils.models import FeedForwardHedger, NoTransactionBandNet
from utils.strategies import bs_delta, leland_delta, whalley_wilmott
from utils.training import evaluate


def load(Model, filename):
    model = Model(cfg.T)
    model.load_state_dict(torch.load(f"data/results/{filename}"))
    return model.eval()


if __name__ == "__main__":
    cost = cfg.COST
    dt = cfg.T / cfg.N_STEPS
    test_paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS, seed=cfg.TEST_SEED)

    strategies = {
        "BS delta": bs_delta,
        "Leland delta": leland_delta(cost, cfg.SIGMA, cfg.R, cfg.K, dt),
        "Whalley-Wilmott": whalley_wilmott(cost, cfg.RISK_AVERSION, cfg.R),
        "Feed-forward": load(FeedForwardHedger, f"feed-forward_{cost}.pt"),
        "No-transaction band": load(NoTransactionBandNet, f"no-transaction_band_{cost}.pt"),
    }
    results = pd.DataFrame([{"strategy": n, **evaluate(s, test_paths, cfg, cost)}
                            for n, s in strategies.items()]).set_index("strategy")
    print(results[["mean_pnl", "std_pnl", "cvar_95", "entropic_risk", "price", "turnover"]].round(3))
    results.to_csv(f"data/results/comparison_{cost}.csv")