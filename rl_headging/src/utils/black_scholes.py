import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type="call") -> float:
    """
    Black-Scholes option price.
    S     : current stock price
    K     : strike price
    T     : time to expiry (in years)
    r     : risk-free rate
    sigma : volatility
    """
    if T <= 0:
        # At expiry — intrinsic value only
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)

def delta(S, K, T, r, sigma, option_type="call") -> float:
    """
    Black-Scholes delta — sensitivity of option price to underlying price.
    For a call: delta ∈ [0, 1]
    For a put:  delta ∈ [-1, 0]
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1)

def gamma(S, K, T, r, sigma) -> float:
    """
    Gamma — rate of change of delta with respect to underlying price.
    Same for calls and puts.
    """
    if T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

if __name__ == "__main__":
    # Sanity check
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    price = black_scholes_price(S, K, T, r, sigma)
    d = delta(S, K, T, r, sigma)
    g = gamma(S, K, T, r, sigma)

    print(f"ATM Call Price : {price:.4f}")
    print(f"Delta          : {d:.4f}")
    print(f"Gamma          : {g:.4f}")