import sys
sys.path.append("src")
import config as cfg
from utils.market import simulate_gbm
from utils.hedging import simulate_hedge
from utils.risk import entropic_risk
from utils.strategies import bs_delta, whalley_wilmott 

paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS, seed=cfg.TEST_SEED)
market = dict(k=cfg.K, T=cfg.T, r=cfg.R, sigma=cfg.SIGMA)

print(f"{'cost':>6} | {'delta: risk':>11} {'turnover':>8} | {'WW: risk':>8} {'turnover':>8}")
for cost in cfg.COST_GRID:
    row = []
    for strat in [bs_delta, whalley_wilmott(cost, cfg.RISK_AVERSION, cfg.R)]:
        pnl, hedges = simulate_hedge(paths, strat, cost=cost, **market)
        turnover = (hedges.diff(dim=1).abs().sum(dim=1) + hedges[:, 0].abs()).mean()
        row += [float(entropic_risk(pnl, cfg.RISK_AVERSION)), float(turnover)]
    print(f"{cost:6.2%} | {row[0]:11.3f} {row[1]:8.2f} | {row[2]:8.3f} {row[3]:8.2f}")