import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from option_env import OptionHedgingEnv
from utils.black_scholes import delta, black_scholes_price
from utils.simulation import simulate_gbm

# ─────────────────────────────────────────────
# HELPER — run RL agent on N episodes
# ─────────────────────────────────────────────

def evaluate_rl_agent(model, n_episodes: int = 1000, seed: int = 99) -> dict:
    """
    Run the trained RL agent on n_episodes fresh episodes
    and collect the total P&L for each one.

    We use a different seed than training (seed=99 vs seed=42)
    to ensure we are testing on truly out-of-sample price paths
    — this is the RL equivalent of a test set.
    """
    np.random.seed(seed)

    env = OptionHedgingEnv(
        S0=100.0, K=100.0, T=1.0, r=0.05,
        sigma=0.2, n_steps=50, transaction_cost=0.01
    )

    total_pnl = np.zeros(n_episodes)

    for i in range(n_episodes):
        # Reset gives a fresh price path for each episode
        obs, _ = env.reset(seed=seed + i)
        episode_pnl = 0.0
        done = False

        while not done:
            # model.predict returns (action, state)
            # deterministic=True means we use the mean action,
            # not sampling from the policy distribution
            # This is important for evaluation — we want the best action,
            # not an exploratory one
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_pnl += info["pnl"]
            done = terminated or truncated

        total_pnl[i] = episode_pnl

    return {
        "mean_pnl":  np.mean(total_pnl),
        "std_pnl":   np.std(total_pnl),
        "sharpe":    np.mean(total_pnl) / (np.std(total_pnl) + 1e-8),
        "var_95":    np.percentile(total_pnl, 5),
        "cvar_95":   np.mean(total_pnl[total_pnl <= np.percentile(total_pnl, 5)]),
        "total_pnl": total_pnl
    }


# ─────────────────────────────────────────────
# HELPER — run delta hedge on N episodes
# ─────────────────────────────────────────────

def evaluate_delta_hedge(n_episodes: int = 1000, seed: int = 99) -> dict:
    """
    Run the textbook delta hedge on the same price paths as the RL agent
    so the comparison is fair — same market conditions, different strategy.
    """
    np.random.seed(seed)

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    n_steps = 50
    tc = 0.01

    # Generate the same price paths used by the RL agent
    paths = simulate_gbm(
        S0=S0, mu=r, sigma=sigma,
        T=T, n_steps=n_steps,
        n_paths=n_episodes, seed=seed
    )

    total_pnl = np.zeros(n_episodes)

    for i in range(n_episodes):
        path = paths[i]
        hedge = 0.0
        episode_pnl = 0.0

        for t in range(n_steps):
            S = path[t]
            S_next = path[t + 1]
            time_left = (n_steps - t) / n_steps * T

            # Delta hedge: set hedge = BS delta at every step
            new_hedge = delta(S=S, K=K,
                              T=max(time_left, 1e-6),
                              r=r, sigma=sigma)

            # Transaction cost for changing position
            hedge_change = abs(new_hedge - hedge)
            transaction_cost = tc * hedge_change * S

            # P&L from stock position
            stock_pnl = hedge * (S_next - S)

            # P&L from short option position
            time_left_next = (n_steps - t - 1) / n_steps * T
            opt_old = black_scholes_price(S, K,
                                          max(time_left, 1e-6), r, sigma)
            opt_new = black_scholes_price(S_next, K,
                                          max(time_left_next, 1e-6), r, sigma)
            option_pnl = -(opt_new - opt_old)

            episode_pnl += stock_pnl + option_pnl - transaction_cost
            hedge = new_hedge

        total_pnl[i] = episode_pnl

    return {
        "mean_pnl":  np.mean(total_pnl),
        "std_pnl":   np.std(total_pnl),
        "sharpe":    np.mean(total_pnl) / (np.std(total_pnl) + 1e-8),
        "var_95":    np.percentile(total_pnl, 5),
        "cvar_95":   np.mean(total_pnl[total_pnl <= np.percentile(total_pnl, 5)]),
        "total_pnl": total_pnl
    }


# ─────────────────────────────────────────────
# MAIN — load model, evaluate, compare, plot
# ─────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs("data/results", exist_ok=True)

    # ── Load trained model ──
    print("Loading trained PPO agent...")
    model = PPO.load("data/results/ppo_hedging_agent")

    # ── Evaluate both strategies on 1000 episodes ──
    print("Evaluating RL agent on 1000 episodes...")
    rl_results = evaluate_rl_agent(model, n_episodes=1000)

    print("Evaluating delta hedge on 1000 episodes...")
    dh_results = evaluate_delta_hedge(n_episodes=1000)

    # ── Print comparison table ──
    print("\n" + "="*50)
    print(f"{'Metric':<20} {'Delta Hedge':>15} {'RL Agent':>15}")
    print("="*50)
    metrics = [
        ("Mean P&L",  "mean_pnl"),
        ("Std P&L",   "std_pnl"),
        ("Sharpe",    "sharpe"),
        ("VaR 95%",   "var_95"),
        ("CVaR 95%",  "cvar_95"),
    ]
    for label, key in metrics:
        dh_val = dh_results[key]
        rl_val = rl_results[key]
        # Mark improvement with arrow
        better = "↑" if rl_val > dh_val else "↓"
        print(f"{label:<20} {dh_val:>15.4f} {rl_val:>15.4f} {better}")
    print("="*50)

    # ── Plot P&L distributions side by side ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, results, title, color in zip(
        axes,
        [dh_results, rl_results],
        ["Delta Hedge", "RL Agent (PPO)"],
        ["royalblue", "seagreen"]
    ):
        ax.hist(results["total_pnl"], bins=60,
                color=color, alpha=0.7, edgecolor="white")
        ax.axvline(results["mean_pnl"], color="orange", linestyle="--",
                   linewidth=2, label=f"Mean: {results['mean_pnl']:.3f}")
        ax.axvline(results["var_95"], color="red", linestyle="--",
                   linewidth=2, label=f"VaR 95%: {results['var_95']:.3f}")
        ax.axvline(results["cvar_95"], color="darkred", linestyle=":",
                   linewidth=2, label=f"CVaR 95%: {results['cvar_95']:.3f}")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Total P&L")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.suptitle("RL Agent vs Delta Hedge — P&L Distribution (1000 episodes)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("data/results/hedge_comparison.png", dpi=150)
    print("\nComparison plot saved to data/results/hedge_comparison.png")

    # ── Load and evaluate CVaR agent ──
    print("\nEvaluating CVaR PPO agent on 1000 episodes...")
    from option_env_cvar import OptionHedgingEnvCVaR
    from stable_baselines3 import PPO as PPO2

    cvar_model = PPO2.load("data/results/ppo_cvar_agent")
    cvar_env = OptionHedgingEnvCVaR(
        S0=100.0, K=100.0, T=1.0, r=0.05,
        sigma=0.2, n_steps=50, transaction_cost=0.01
    )

    cvar_pnl = np.zeros(1000)
    for i in range(1000):
        obs, _ = cvar_env.reset(seed=99 + i)
        episode_pnl = 0.0
        done = False
        while not done:
            action, _ = cvar_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = cvar_env.step(action)
            episode_pnl += info["pnl"]
            done = terminated or truncated
        cvar_pnl[i] = episode_pnl

    cvar_results = {
        "mean_pnl":  np.mean(cvar_pnl),
        "std_pnl":   np.std(cvar_pnl),
        "sharpe":    np.mean(cvar_pnl) / (np.std(cvar_pnl) + 1e-8),
        "var_95":    np.percentile(cvar_pnl, 5),
        "cvar_95":   np.mean(cvar_pnl[cvar_pnl <= np.percentile(cvar_pnl, 5)]),
        "total_pnl": cvar_pnl
    }

    # ── Full comparison table ──
    print("\n" + "="*65)
    print(f"{'Metric':<20} {'Delta Hedge':>14} {'RL v1':>14} {'RL CVaR':>14}")
    print("="*65)
    for label, key in metrics:
        dh  = dh_results[key]
        rl  = rl_results[key]
        cv  = cvar_results[key]
        print(f"{label:<20} {dh:>14.4f} {rl:>14.4f} {cv:>14.4f}")
    print("="*65)

    # ── 3-way plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, results, title, color in zip(
        axes,
        [dh_results, rl_results, cvar_results],
        ["Delta Hedge", "RL v1 (variance)", "RL v2 (CVaR)"],
        ["royalblue", "seagreen", "darkorange"]
    ):
        ax.hist(results["total_pnl"], bins=60,
                color=color, alpha=0.7, edgecolor="white")
        ax.axvline(results["mean_pnl"], color="black", linestyle="--",
                   linewidth=2, label=f"Mean: {results['mean_pnl']:.3f}")
        ax.axvline(results["cvar_95"], color="red", linestyle=":",
                   linewidth=2, label=f"CVaR: {results['cvar_95']:.3f}")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Total P&L")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.suptitle("3-Way Comparison: Delta Hedge vs RL v1 vs RL CVaR",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("data/results/full_comparison.png", dpi=150)
    print("\nFull comparison plot saved to data/results/full_comparison.png")