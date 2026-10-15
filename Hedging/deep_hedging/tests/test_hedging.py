"""
Sanity checks for the hedging engine. Run from the deep_hedging folder:
    python -m pytest tests -q
"""
import math
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config as cfg
from utils import black_scholes as bs
from utils.hedging import simulate_hedge
from utils.market import simulate_gbm, simulate_gbm_torch
from utils.risk import entropic_risk, expected_shortfall
from utils.strategies import bs_delta, clamp_into_band, no_hedge, whalley_wilmott

MARKET = dict(k=cfg.K, T=cfg.T, r=cfg.R, sigma=cfg.SIGMA)


def paths(n=20_000, n_steps=cfg.N_STEPS, seed=1):
    return simulate_gbm(n, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, n_steps, seed=seed)


def test_black_scholes_reference_value():
    # ATM call, S=K=100, T=1, r=5%, sigma=20%: textbook value 10.4506, delta 0.6368
    s = torch.tensor([100.0], dtype=torch.float64)
    assert abs(float(bs.call_price(s, 100.0, 1.0, 0.05, 0.2)) - 10.4506) < 1e-3
    assert abs(float(bs.call_delta(s, 100.0, 1.0, 0.05, 0.2)) - 0.6368) < 1e-3


def test_delta_and_gamma_match_finite_differences():
    s = torch.tensor([80.0, 100.0, 120.0], dtype=torch.float64)
    h = 1e-3
    up = bs.call_price(s + h, 100.0, 0.5, 0.05, 0.2)
    mid = bs.call_price(s, 100.0, 0.5, 0.05, 0.2)
    down = bs.call_price(s - h, 100.0, 0.5, 0.05, 0.2)
    assert torch.allclose(bs.call_delta(s, 100.0, 0.5, 0.05, 0.2), (up - down) / (2 * h), atol=1e-6)
    assert torch.allclose(bs.call_gamma(s, 100.0, 0.5, 0.05, 0.2), (up - 2 * mid + down) / h ** 2, atol=1e-4)


def test_paths_are_martingales_after_discounting():
    p = paths(200_000)
    discounted_final = p[:, -1] * math.exp(-cfg.R * cfg.T)
    assert abs(float(discounted_final.mean()) - cfg.S0) < 0.15


def test_numpy_and_torch_generators_agree_in_distribution():
    a = paths(100_000)[:, -1]
    b = simulate_gbm_torch(100_000, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS,
                           generator=torch.Generator().manual_seed(0), dtype=torch.float64)[:, -1]
    assert abs(float(a.mean() - b.mean())) < 0.3
    assert abs(float(a.std() - b.std())) < 0.3


def test_unhedged_pnl_has_zero_mean():
    # Selling at the fair price and not hedging: zero expected P&L under Q
    pnl, _ = simulate_hedge(paths(), no_hedge, cost=0.0, **MARKET)
    assert abs(float(pnl.mean())) < 0.25


def test_delta_hedge_without_costs_is_almost_riskless():
    pnl, _ = simulate_hedge(paths(), bs_delta, cost=0.0, **MARKET)
    assert abs(float(pnl.mean())) < 0.05
    assert float(pnl.std()) < 1.2


def test_hedging_error_shrinks_with_more_rebalancing():
    std_coarse = float(simulate_hedge(paths(n_steps=25), bs_delta, cost=0.0, **MARKET)[0].std())
    std_fine = float(simulate_hedge(paths(n_steps=100), bs_delta, cost=0.0, **MARKET)[0].std())
    # Discrete hedging error decreases like 1/sqrt(n_steps)
    assert std_fine < 0.6 * std_coarse


def test_costs_only_reduce_pnl():
    p = paths(5_000)
    free, _ = simulate_hedge(p, bs_delta, cost=0.0, **MARKET)
    costly, _ = simulate_hedge(p, bs_delta, cost=0.01, **MARKET)
    assert bool((costly <= free + 1e-12).all())


def test_band_trades_less_than_delta():
    p = paths(5_000)
    _, h_delta = simulate_hedge(p, bs_delta, cost=0.01, **MARKET)
    _, h_band = simulate_hedge(p, whalley_wilmott(0.01, cfg.RISK_AVERSION, cfg.R), cost=0.01, **MARKET)
    assert float(h_band.diff(dim=1).abs().sum()) < float(h_delta.diff(dim=1).abs().sum())


def test_clamp_into_band():
    prev = torch.tensor([0.1, 0.5, 0.9, 0.5])
    lower = torch.tensor([0.3, 0.3, 0.3, 0.7])
    upper = torch.tensor([0.7, 0.7, 0.7, 0.3])
    out = clamp_into_band(prev, lower, upper)
    assert torch.allclose(out, torch.tensor([0.3, 0.5, 0.7, 0.5]))


def test_risk_measures():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0] * 20, dtype=torch.float64)
    # Entropic risk of a constant is minus that constant
    assert abs(float(entropic_risk(torch.full((10,), 2.0)) + 2.0)) < 1e-6
    # Entropic risk is above minus the mean (risk aversion penalizes dispersion)
    assert float(entropic_risk(x)) > -float(x.mean())
    # Expected shortfall at 80% = average of the worst 20% of losses
    assert abs(float(expected_shortfall(x, 0.8)) - 3.0) < 1e-9


def test_pnl_is_differentiable():
    weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    pnl, _ = simulate_hedge(paths(1_000), lambda s: weight * s.bs_delta, cost=0.01, **MARKET)
    entropic_risk(pnl).backward()
    assert weight.grad is not None and torch.isfinite(weight.grad)
