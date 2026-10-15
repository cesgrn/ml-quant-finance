import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src/ to path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.black_scholes import black_scholes_price, delta
from utils.simulation import simulate_gbm

def run_delta_hedge(
        S0: float = 100.0,        # initial stock price
        K: float = 100.0,         # strike price
        T: float = 1.0,           # time to expiry in years
        r: float = 0.05,          # risk-free rate
        sigma: float = 0.2,       # volatility
        n_steps: int = 50,        # number of rebalancing steps
        transaction_cost: float = 0.01,  # cost per unit traded
        n_paths: int = 1000,      # number of simulated episodes
        seed: int = 42
) -> dict:
    """
    Run the textbook Black-Scholes delta hedge strategy.

    At each step, the agent sets its hedge ratio exactly equal
    to the theoretical Black-Scholes delta — no learning involved.
    This is our baseline to beat with the RL agent.

    Returns a dict with P&L statistics across all simulated paths.
    """

    dt = T / n_steps

    # Simulate all price paths at once for efficiency
    # Shape: (n_paths, n_steps + 1)
    paths = simulate_gbm(
        S0=S0, mu=r, sigma=sigma,
        T=T, n_steps=n_steps,
        n_paths=n_paths, seed=seed
    )

    # Collect the total P&L for each path
    total_pnl = np.zeros(n_paths)

    for i in range(n_paths):
        path = paths[i]           # price path for this episode
        hedge = 0.0               # start with no position
        episode_pnl = 0.0

        for t in range(n_steps):
            S = path[t]           # current stock price
            S_next = path[t + 1]  # next stock price

            # Time remaining in years at this step
            time_left = (n_steps - t) / n_steps * T

            # --- Delta hedge action ---
            # Set hedge = theoretical BS delta
            # This is the "correct" hedge in a frictionless market
            new_hedge = delta(S=S, K=K, T=max(time_left, 1e-6),
                              r=r, sigma=sigma)

            # --- Transaction cost ---
            # Cost proportional to how much we change our position
            hedge_change = abs(new_hedge - hedge)
            tc = transaction_cost * hedge_change * S

            # --- P&L this step ---
            # Gain from holding (hedge) shares over the price move
            stock_pnl = hedge * (S_next - S)

            # Option P&L (we are short — we LOSE when option value increases)
            time_left_next = (n_steps - t - 1) / n_steps * T
            opt_old = black_scholes_price(S=S, K=K,
                                          T=max(time_left, 1e-6),
                                          r=r, sigma=sigma)
            opt_new = black_scholes_price(S=S_next, K=K,
                                          T=max(time_left_next, 1e-6),
                                          r=r, sigma=sigma)
            option_pnl = -(opt_new - opt_old)

            step_pnl = stock_pnl + option_pnl - tc
            episode_pnl += step_pnl
            hedge = new_hedge     # update hedge position

        total_pnl[i] = episode_pnl

    # --- Compute statistics ---
    results = {
        "mean_pnl":    np.mean(total_pnl),
        "std_pnl":     np.std(total_pnl),
        "sharpe":      np.mean(total_pnl) / (np.std(total_pnl) + 1e-8),
        # VaR 95%: worst 5% of outcomes
        "var_95":      np.percentile(total_pnl, 5),
        # CVaR 95%: average of worst 5% — used in Deep Hedging paper
        "cvar_95":     np.mean(total_pnl[total_pnl <= np.percentile(total_pnl, 5)]),
        "total_pnl":   total_pnl
    }

    return results


if __name__ == "__main__":
    print("Running delta hedge baseline on 1000 episodes...")
    results = run_delta_hedge()

    print(f"\n--- Delta Hedge Baseline Results ---")
    print(f"Mean P&L   : {results['mean_pnl']:.4f}")
    print(f"Std P&L    : {results['std_pnl']:.4f}")
    print(f"Sharpe     : {results['sharpe']:.4f}")
    print(f"VaR 95%    : {results['var_95']:.4f}")
    print(f"CVaR 95%   : {results['cvar_95']:.4f}")

    # Plot P&L distribution
    os.makedirs("data/results", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(results["total_pnl"], bins=50, color="royalblue",
             alpha=0.7, edgecolor="white")
    plt.axvline(results["mean_pnl"], color="orange",
                linestyle="--", label=f"Mean: {results['mean_pnl']:.3f}")
    plt.axvline(results["var_95"], color="red",
                linestyle="--", label=f"VaR 95%: {results['var_95']:.3f}")
    plt.title("Delta Hedge — P&L Distribution (1000 episodes)")
    plt.xlabel("Total P&L")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/results/delta_hedge_pnl.png", dpi=150)
    print("\nPlot saved to data/results/delta_hedge_pnl.png")