"""
Step 9 — Look inside the part A PPO policies.

We don't trust the clipped action alone: we print the raw Gaussian mean μ(obs)
and std σ that the policy network outputs, before SB3 clips to [0, 1].
"""
import math
import datetime as dt
from pathlib import Path

import numpy as np
import torch
import stable_baselines3 as sb3
from stable_baselines3 import PPO

# deep_hedging/src/this_file -> parents[2] = Hedging/
RESULTS = Path(__file__).resolve().parents[2] / "rl_hedging" / "data" / "results"
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0


def bs_delta(s, tau):
    """Black-Scholes call delta N(d1), same formula as part A's utils."""
    d1 = (math.log(s / K) + (R + 0.5 * SIGMA**2) * tau) / (SIGMA * math.sqrt(tau))
    return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))


def sample_observations():
    """A small grid of part A observations: [S/S0, tau/T, bs_delta, prev_hedge]."""
    rows = []
    for tau in (1.0, 0.5, 0.1):
        for s in (80.0, 100.0, 120.0):
            for prev in (0.0, 0.5):
                rows.append([s / S0, tau / T, bs_delta(s, tau), prev])
    return np.array(rows, dtype=np.float32)


def inspect(name):
    path = RESULTS / f"{name}.zip"
    agent = PPO.load(path, device="cpu")

    print(f"\n=== {name} ===")
    print("file modified :", dt.datetime.fromtimestamp(path.stat().st_mtime))
    print("timesteps     :", agent.num_timesteps)
    print("action space  :", agent.action_space)

    obs = sample_observations()

    # 1) Raw Gaussian parameters, before any clipping
    obs_t, _ = agent.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = agent.policy.get_distribution(obs_t).distribution
        mu = dist.mean[:, 0].numpy()
        sd = dist.stddev[:, 0].numpy()

    # 2) What part A's evaluation actually used: clip(mu, 0, 1)
    det, _ = agent.predict(obs, deterministic=True)

    # 3) Stochastic actions: how often does a sample land strictly inside (0, 1)?
    torch.manual_seed(0)
    samples = np.stack([agent.predict(obs, deterministic=False)[0][:, 0]
                        for _ in range(200)])
    inside = ((samples > 0) & (samples < 1)).mean(axis=0)

    print(f"{'S':>5} {'tau':>5} {'BSd':>6} {'prev':>5} | "
          f"{'raw mu':>8} {'sigma':>6} {'det':>6} {'P(0<a<1)':>9}")
    for o, m, s_, d, f in zip(obs, mu, sd, det[:, 0], inside):
        print(f"{o[0]*S0:5.0f} {o[1]:5.2f} {o[2]:6.3f} {o[3]:5.1f} | "
              f"{m:8.3f} {s_:6.3f} {d:6.3f} {f:9.2f}")


if __name__ == "__main__":
    print("stable-baselines3", sb3.__version__, "| torch", torch.__version__)
    inspect("ppo_hedging_agent")
    inspect("ppo_cvar_agent")