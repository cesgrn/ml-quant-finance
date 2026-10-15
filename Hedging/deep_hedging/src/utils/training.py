import time
import torch
from .hedging import simulate_hedge
from .market import simulate_gbm_torch
from .risk import entropic_risk, indifference_price, summary

def train(model, cfg, cost, n_iters=None, log_every=100, seed=None):
    n_iters = n_iters or cfg.N_ITERS
    torch.manual_seed(cfg.TRAIN_SEED if seed is None else seed)
    model.to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    history, start = [], time.time()

    for step in range(n_iters):
        paths = simulate_gbm_torch(cfg.BATCH_SIZE, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS, device=cfg.DEVICE)
        pnl, _ = simulate_hedge(paths, model, k=cfg.K, T=cfg.T, r=cfg.R, sigma=cfg.SIGMA, cost=cost)
        loss = entropic_risk(pnl, cfg.RISK_AVERSION)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"  iter {step + 1:5d}  loss={loss.item():.4f}  ({time.time() - start:.0f}s)")
    return history

@torch.no_grad()
def evaluate(strategy, paths, cfg, cost): 
    if isinstance(strategy, torch.nn.Module):
        strategy.to("cpu")
    market = dict(k=cfg.K, T=cfg.T, r=cfg.R, sigma=cfg.SIGMA, cost=cost)
    pnl, hedges = simulate_hedge(paths, strategy, **market)
    pnl0, _ = simulate_hedge(paths, strategy, premium=0.0, **market)
    stats = summary(pnl, cfg.CVAR_LEVEL, cfg.RISK_AVERSION)
    stats["price"] = indifference_price(pnl0, cfg.RISK_AVERSION)
    stats["turnover"] = float((hedges.diff(dim=1).abs().sum(dim=1) + hedges[:, 0].abs()).mean())
    return stats