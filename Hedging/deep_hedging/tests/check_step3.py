import sys, torch
sys.path.append("src")
import config as cfg
from utils.market import simulate_gbm
from utils.hedging import simulate_hedge

paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS, seed=cfg.TEST_SEED)
market = dict(k=cfg.K, T=cfg.T, r=cfg.R, sigma=cfg.SIGMA)

strategies = {
    "no hedge":  lambda s: torch.zeros_like(s.spot),
    "BS delta":  lambda s: s.bs_delta,
    "one share": lambda s: torch.ones_like(s.spot),
}
for name, strategy in strategies.items():
    for cost in [0.0, 0.01]:
        pnl, hedges = simulate_hedge(paths, strategy, cost=cost, **market)
        print(f"{name:10s} cost={cost:<5} mean={pnl.mean():+.3f}  std={pnl.std():.3f}")