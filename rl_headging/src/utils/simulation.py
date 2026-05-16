import numpy as np

def simulate_gbm(S0: float, mu: float, sigma: float,
                 T: float, n_steps: int, n_paths: int,
                 seed: int = 42) -> np.ndarray:
    """
    Simulate stock price paths under Geometric Brownian Motion.
    S0      : initial stock price
    mu      : drift (risk-free rate under risk-neutral measure)
    sigma   : volatility
    T       : time horizon in years
    n_steps : number of time steps
    n_paths : number of simulated paths
    Returns : array of shape (n_paths, n_steps + 1)
    """
    np.random.seed(seed)
    dt = T / n_steps

    # Generate random shocks
    Z = np.random.standard_normal((n_paths, n_steps))

    # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Build price paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp(log_returns[:, t-1])

    return paths

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    paths = simulate_gbm(S0=100, mu=0.05, sigma=0.2,
                         T=1.0, n_steps=252, n_paths=1000)

    print(f"Shape          : {paths.shape}")
    print(f"Initial price  : {paths[:, 0].mean():.2f}")
    print(f"Final price    : {paths[:, -1].mean():.2f}")
    print(f"Final price std: {paths[:, -1].std():.2f}")

    plt.figure(figsize=(10, 4))
    plt.plot(paths[:50].T, alpha=0.3, color="royalblue")
    plt.title("50 GBM Price Paths — SPY simulation")
    plt.xlabel("Trading days")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("data/gbm_paths.png", dpi=150)
    print("Plot saved to data/gbm_paths.png")