import os
import sys
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils.market import simulate_gbm
from utils.models import NoTransactionBandNet
from utils.strategies import bs_delta, leland_delta, whalley_wilmott
from utils.training import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
PART_A = os.path.join(HERE, "..", "..", "rl_hedging", "data", "results")


def ppo_strategy(filename):
    agent = PPO.load(os.path.join(PART_A, filename), device="cpu")

    def strategy(state):
        n = state.spot.shape[0]
        obs = np.stack([
            (state.spot / cfg.S0).numpy(),
            np.full(n, state.time_to_maturity / cfg.T),
            state.bs_delta.numpy(),
            state.prev_hedge.numpy(),
        ], axis=1).astype(np.float32)
        action, _ = agent.predict(obs, deterministic=True)
        return torch.as_tensor(action[:, 0], dtype=state.spot.dtype)
    return strategy


if __name__ == "__main__":
    cost = cfg.COST
    dt = cfg.T / cfg.N_STEPS
    test_paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS, seed=cfg.TEST_SEED)

    band = NoTransactionBandNet(cfg.T)
    band.load_state_dict(torch.load(f"data/results/no-transaction_band_{cost}.pt"))

    strategies = {
        "BS delta": bs_delta,
        "Part A: PPO v1 (variance reward)": ppo_strategy("ppo_hedging_agent.zip"),
        "Part A: PPO v2 (CVaR reward)": ppo_strategy("ppo_cvar_agent.zip"),
        "Leland delta": leland_delta(cost, cfg.SIGMA, cfg.R, cfg.K, dt),
        "Whalley-Wilmott": whalley_wilmott(cost, cfg.RISK_AVERSION, cfg.R),
        "Part B: band network": band.eval(),
    }
    results = pd.DataFrame([{"strategy": n, **evaluate(s, test_paths, cfg, cost)}
                            for n, s in strategies.items()]).set_index("strategy")
    cols = ["mean_pnl", "std_pnl", "sharpe", "cvar_95", "entropic_risk", "price", "turnover"]
    print(results[cols].round(3).to_string())
    results.to_csv("data/results/part_a_vs_part_b.csv")