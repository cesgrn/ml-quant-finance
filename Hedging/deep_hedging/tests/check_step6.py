import sys, torch
sys.path.append("src")
import config as cfg
from utils.market import simulate_gbm
from utils.hedging import simulate_hedge
from utils.risk import entropic_risk
from utils.models import FeedForwardHedger, NoTransactionBandNet

paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS, seed=cfg.TEST_SEED)
market = dict(k=cfg.K, T=cfg.T, r=cfg.R, sigma=cfg.SIGMA)

for Model in [FeedForwardHedger, NoTransactionBandNet]:
    torch.manual_seed(0)
    model = Model(cfg.T)
    n_params = sum(p.numel() for p in model.parameters())

    # 1. Zero the last layer: the model must become the delta hedge
    with torch.no_grad():
        model.mlp.net[-1].weight.zero_()
        model.mlp.net[-1].bias.zero_()
        pnl, _ = simulate_hedge(paths, model, cost=cfg.COST, **market)
    print(f"{Model.__name__:22s} params={n_params}  zeroed: mean={pnl.mean():+.3f} "
          f"risk={entropic_risk(pnl, cfg.RISK_AVERSION):.3f}")   # expected -2.724 and 4.048

    # 2. Gradient check on a small float32 batch with random weights
    model = Model(cfg.T)
    small = paths[:2000].float()
    pnl, _ = simulate_hedge(small, model, cost=cfg.COST, **market)
    entropic_risk(pnl, cfg.RISK_AVERSION).backward()
    grad = model.mlp.net[0].weight.grad
    print(f"{'':22s} gradient on first layer: norm={grad.norm():.4f}, finite={torch.isfinite(grad).all().item()}")