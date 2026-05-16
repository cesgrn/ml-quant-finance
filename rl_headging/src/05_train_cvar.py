import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from option_env_cvar import OptionHedgingEnvCVaR


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean(
                [ep["r"] for ep in self.model.ep_info_buffer]
            )
            self.rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)
        return True


def train_cvar_agent(total_timesteps: int = 200_000, seed: int = 42):

    os.makedirs("data/results", exist_ok=True)

    def make_env():
        env = OptionHedgingEnvCVaR(
            S0=100.0, K=100.0, T=1.0, r=0.05,
            sigma=0.2, n_steps=50,
            transaction_cost=0.01,
            cvar_lambda=2.0    # penalize losses 2x more than variance
        )
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=4, seed=seed)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        verbose=1,
        seed=seed,
        tensorboard_log=None
    )

    print(f"Training CVaR PPO agent for {total_timesteps:,} timesteps...")
    callback = TrainingCallback()
    model.learn(total_timesteps=total_timesteps,
                callback=callback, progress_bar=True)

    model.save("data/results/ppo_cvar_agent")
    print("Model saved to data/results/ppo_cvar_agent.zip")

    # Plot learning curve
    if len(callback.rewards) > 0:
        plt.figure(figsize=(10, 4))
        rewards = np.array(callback.rewards)
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        plt.plot(callback.timesteps[:len(smoothed)], smoothed,
                 color="seagreen", label="Smoothed reward")
        plt.plot(callback.timesteps, rewards,
                 color="lightgreen", alpha=0.4, label="Raw reward")
        plt.title("PPO CVaR Training Curve")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig("data/results/training_curve_cvar.png", dpi=150)

    return model


if __name__ == "__main__":
    model = train_cvar_agent(total_timesteps=200_000)
    print("\nCVaR training complete.")