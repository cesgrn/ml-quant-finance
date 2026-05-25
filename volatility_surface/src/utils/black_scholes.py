import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_call_price(S: float, K: float, T: float,
                  r: float, sigma: float) -> float:
    """
    Black-Scholes call price.
    S     : spot price
    K     : strike price
    T     : time to expiry in years
    r     : risk-free rate
    sigma : volatility
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(market_price: float, S: float, K: float,
                       T: float, r: float = 0.04) -> float:
    """
    Compute implied volatility by inverting Black-Scholes using Brent's method.

    Brent's method is a root-finding algorithm — it finds the sigma such that:
        BS(sigma) - market_price = 0

    It combines bisection (safe but slow) and secant method (fast but can diverge)
    to guarantee convergence within [sigma_low, sigma_high].

    Returns NaN if:
    - Option has no intrinsic value (deep OTM, market price < intrinsic)
    - No solution found in [0.001, 20.0] range
    - Time to expiry is zero or negative
    """
    if T <= 0:
        return np.nan

    # Intrinsic value — minimum possible price
    intrinsic = max(S - K * np.exp(-r * T), 0.0)

    # Market price below intrinsic — data error, skip
    if market_price <= intrinsic + 1e-6:
        return np.nan

    # Define the function whose root we seek: f(sigma) = BS(sigma) - market_price
    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma) - market_price

    try:
        # Brent's method — search between 0.1% and 2000% volatility
        iv = brentq(objective, 1e-4, 20.0, xtol=1e-6, maxiter=500)
        return float(iv)
    except (ValueError, RuntimeError):
        # brentq raises ValueError if f(a) and f(b) have the same sign
        # (no root in interval) — happens for illiquid options
        return np.nan