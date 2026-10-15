"""Classic (non-learned) hedging strategies, used as benchmarks."""
import math
import torch
from . import black_scholes as bs
from .hedging import MarketState


def no_hedge(state: MarketState) -> torch.Tensor:
    """Never trade: keeps the full risk of the short call."""
    return torch.zeros_like(state.spot)


def bs_delta(state: MarketState) -> torch.Tensor:
    """Textbook Black-Scholes delta hedge, rebalanced at every date."""
    return state.bs_delta


def clamp_into_band(prev: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """
    Move the previous hedge to the nearest edge of [lower, upper], or keep it if
    it is already inside: this is the no-transaction band rule.
    If lower > upper, return the midpoint (same convention as Imaki et al. 2021).
    """
    clamped = torch.minimum(torch.maximum(prev, lower), upper)
    return torch.where(lower <= upper, clamped, 0.5 * (lower + upper))

def ww_half_width(state, cost, risk_aversion, r):
    return (1.5 * cost * math.exp(-r * state.time_to_maturity) * state.spot
            * state.bs_gamma**2 / risk_aversion) ** (1.0 / 3.0)

def whalley_wilmott(cost: float, risk_aversion: float, r: float):
    """
    Whalley & Wilmott (1997) asymptotic no-transaction band for exponential utility:

        half-width = ( 3/2 * cost * exp(-r tau) * S * Gamma^2 / lambda )^(1/3)

    centred on the Black-Scholes delta. Valid for small costs. Returns a strategy.
    """
    def strategy(state):
        if cost == 0.0:
            return state.bs_delta
        width = ww_half_width(state, cost, risk_aversion, r)
        return clamp_into_band(state.prev_hedge, state.bs_delta - width, state.bs_delta + width)
    return strategy

def leland_volatility(sigma, cost, dt):
    """Leland (1985) adjusted volatility; k = 2 * cost is the round-trip cost."""
    return sigma * math.sqrt(1 + math.sqrt(2 / math.pi) * (2 * cost) / (sigma * math.sqrt(dt)))


def leland_delta(cost, sigma, r, k, dt):
    """Black-Scholes delta computed with Leland's volatility, rebalanced at every date."""
    sigma_l = leland_volatility(sigma, cost, dt)
    def strategy(state):
        return bs.call_delta(state.spot, k, state.time_to_maturity, r, sigma_l)
    return strategy
