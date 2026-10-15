"""
Self-financing hedging simulation.

Convention (fixes two issues found in part A (rl_hedging)):
  * The hedge h_i chosen at date t_i is held over [t_i, t_{i+1}] (no one-step lag).
  * Buying shares is financed by a cash account paying the risk-free rate r.
    Everything is expressed in time-0 dollars by discounting the stock price,
    S~_i = exp(-r t_i) S_i. Holding stock therefore earns no free drift.

Terminal P&L of a dealer who sold one call for `premium` and hedges with h:

    PnL = premium - exp(-rT) * payoff
          + sum_i h_i * (S~_{i+1} - S~_i)               (hedging gains)
          - cost * sum_i |h_i - h_{i-1}| * S~_i         (transaction costs, h_{-1} = 0)

As in part A and in the No-Transaction Band paper, the position is not
liquidated at maturity (no closing cost).
"""
from dataclasses import dataclass
from typing import Callable

import torch

from . import black_scholes as bs
from .market import time_grid


@dataclass
class MarketState:
    """Everything a strategy may look at when it chooses the hedge at date t_i."""
    step: int
    spot: torch.Tensor            # S_i, shape (n_paths,)
    time_to_maturity: float       # T - t_i
    log_moneyness: torch.Tensor   # log(S_i / K)
    bs_delta: torch.Tensor        # Black-Scholes delta at (S_i, T - t_i)
    bs_gamma: torch.Tensor        # Black-Scholes gamma at (S_i, T - t_i)
    prev_hedge: torch.Tensor      # h_{i-1} (0 before the first trade)


# A strategy maps the current state to the new hedge ratio h_i, shape (n_paths,)
Strategy = Callable[[MarketState], torch.Tensor]

def market_state(spot, tau, k, r, sigma, prev_hedge=None):
    """Market state for given stock prices at a single date (used for analysis plots)."""
    return MarketState(
        step=0, spot=spot, time_to_maturity=tau,
        log_moneyness=torch.log(spot / k),
        bs_delta=bs.call_delta(spot, k, tau, r, sigma),
        bs_gamma=bs.call_gamma(spot, k, tau, r, sigma),
        prev_hedge=torch.zeros_like(spot) if prev_hedge is None else prev_hedge,
    )

def call_payoff(paths: torch.Tensor, k: float) -> torch.Tensor:
    return torch.clamp(paths[:, -1] - k, min=0.0)


def simulate_hedge(paths: torch.Tensor, strategy: Strategy, *, k: float, T: float,
                   r: float, sigma: float, cost: float, premium=None):
    """
    Run `strategy` on every path and return (pnl, hedges).

    pnl    : terminal P&L in time-0 dollars, shape (n_paths,)
    hedges : hedge ratios h_0 ... h_{n-1}, shape (n_paths, n_steps)

    Fully differentiable: if `strategy` contains a neural network, gradients
    of any function of `pnl` flow back to its weights (this is deep hedging).
    """
    n_paths, n_points = paths.shape
    n_steps = n_points - 1
    t = time_grid(T, n_steps, dtype=paths.dtype, device=paths.device)
    discounted = paths * torch.exp(-r * t)            # S~_i

    if premium is None:
        premium = bs.call_price(paths[:, 0], k, T, r, sigma)

    pnl = torch.as_tensor(premium, dtype=paths.dtype, device=paths.device) * torch.ones(
        n_paths, dtype=paths.dtype, device=paths.device)
    prev = torch.zeros(n_paths, dtype=paths.dtype, device=paths.device)
    hedges = []

    for i in range(n_steps):
        tau = T - i * (T / n_steps)
        spot = paths[:, i]
        state = MarketState(
            step=i,
            spot=spot,
            time_to_maturity=tau,
            log_moneyness=torch.log(spot / k),
            bs_delta=bs.call_delta(spot, k, tau, r, sigma),
            bs_gamma=bs.call_gamma(spot, k, tau, r, sigma),
            prev_hedge=prev,
        )
        h = strategy(state)
        pnl = pnl - cost * torch.abs(h - prev) * discounted[:, i]
        pnl = pnl + h * (discounted[:, i + 1] - discounted[:, i])
        hedges.append(h)
        prev = h

    pnl = pnl - torch.exp(torch.as_tensor(-r * T, dtype=paths.dtype)) * call_payoff(paths, k)
    return pnl, torch.stack(hedges, dim=1)
