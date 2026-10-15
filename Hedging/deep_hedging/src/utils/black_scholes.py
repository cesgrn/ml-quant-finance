"""Black-Scholes price and Greeks of a European call, vectorized in PyTorch."""
import math

import torch
from torch.special import ndtr  # standard normal CDF


def _d1(s, k, tau, r, sigma):
    return (torch.log(s / k) + (r + 0.5 * sigma ** 2) * tau) / (sigma * torch.sqrt(tau))


def _as_tensor(tau, like):
    return torch.as_tensor(tau, dtype=like.dtype, device=like.device)


def call_price(s, k, tau, r, sigma):
    """Black-Scholes call price. `tau` is the time to maturity (must be > 0)."""
    tau = _as_tensor(tau, s)
    d1 = _d1(s, k, tau, r, sigma)
    d2 = d1 - sigma * torch.sqrt(tau)
    return s * ndtr(d1) - k * torch.exp(-r * tau) * ndtr(d2)


def call_delta(s, k, tau, r, sigma):
    """Black-Scholes delta of a call: dV/dS, between 0 and 1."""
    return ndtr(_d1(s, k, _as_tensor(tau, s), r, sigma))


def call_gamma(s, k, tau, r, sigma):
    """Black-Scholes gamma of a call: d2V/dS2."""
    tau = _as_tensor(tau, s)
    d1 = _d1(s, k, tau, r, sigma)
    pdf = torch.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return pdf / (s * sigma * torch.sqrt(tau))
