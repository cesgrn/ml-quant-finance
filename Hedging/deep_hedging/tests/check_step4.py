import sys, torch
sys.path.append("src")
import config as cfg
from utils.market import simulate_gbm
from utils.hedging import simulate_hedge
from utils.risk import entropic_risk, expected_shortfall, indifference_price

paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS, seed=cfg.TEST_SEED)
market = dict(k=cfg.K, T=cfg.T, r=cfg.R, sigma=cfg.SIGMA)

for name, strat in {"no hedge": lambda s: torch.zeros_like(s.spot), "BS delta": lambda s: s.bs_delta}.items():
    for cost in [0.0, 0.01]:
        pnl, _ = simulate_hedge(paths, strat, cost=cost, **market)
        pnl0, _ = simulate_hedge(paths, strat, cost=cost, premium=0.0, **market)
        print(f"{name:9s}, cost={cost:<5}: entropic={entropic_risk(pnl):7.3f}, "
              f"ES95={expected_shortfall(pnl):7.3f} price={indifference_price(pnl0):8.3f}")
        
print("risk of constant 2:", float(entropic_risk(torch.full((10,), 2.0))))