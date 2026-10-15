import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src/ to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from option_env import OptionHedgingEnv

# ─────────────────────────────────────────────
# CALLBACK — track training progress
# ─────────────────────────────────────────────

class TrainingCallback(BaseCallback):
    """
    Custom callback called at every rollout end during training.
    A rollout = one batch of experience collected before a gradient update.
    We use it to record the mean reward over time so we can plot
    the learning curve after training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []          # stores mean reward at each rollout
        self.timesteps = []        # stores total timesteps at each rollout

    def _on_rollout_end(self) -> bool:
        """Called after each rollout — log the mean episode reward."""
        # self.model.ep_info_buffer contains info from recent episodes
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean(
                [ep["r"] for ep in self.model.ep_info_buffer]
            )
            self.rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)
        return True    # return True to continue training
    
    def _on_step(self) -> bool:
        """
        Called at every environment step during training.
        Must return True to continue training, False to stop.
        We don't need anything here — all logging is in _on_rollout_end.
        """
        return True


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_ppo_agent(
    total_timesteps: int = 200_000,   # total env steps to train for
    n_envs: int = 4,                  # number of parallel environments
    learning_rate: float = 3e-4,      # Adam optimizer learning rate
    n_steps: int = 2048,              # steps per env per rollout
    batch_size: int = 64,             # minibatch size for gradient updates
    n_epochs: int = 10,               # epochs per PPO update
    gamma: float = 0.99,              # discount factor for future rewards
    clip_range: float = 0.2,          # PPO clipping parameter
    seed: int = 42,
):
    """
    Train a PPO agent on the option hedging environment.

    PPO (Proximal Policy Optimization) is an actor-critic algorithm:
    - The ACTOR (policy network) maps observations → hedge action
    - The CRITIC (value network) estimates the expected cumulative reward
    PPO updates the policy using gradient ascent, but clips the update
    to avoid destabilizing large policy changes — hence "proximal".

    Key hyperparameters:
    - gamma: how much to discount future rewards (0.99 = care a lot about future)
    - clip_range: how aggressively to update the policy (0.2 = conservative)
    - n_steps: how much experience to collect before each update
    - n_epochs: how many gradient steps per batch of experience
    """

    os.makedirs("data/results", exist_ok=True)

    # ── Create vectorized environment ──
    # We run n_envs=4 environments in parallel to collect experience faster
    # Monitor wraps each env to track episode rewards and lengths
    def make_env():
        env = OptionHedgingEnv(
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            sigma=0.2,
            n_steps=50,
            transaction_cost=0.01
        )
        return Monitor(env)    # Monitor records episode stats automatically

    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)

    # ── Define the PPO agent ──
    # policy="MlpPolicy" means the actor and critic are both
    # multi-layer perceptrons (fully connected neural networks)
    # Default architecture: two hidden layers of 64 units each
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        verbose=1,
        seed=seed,
        tensorboard_log=None
    )

    print(f"Training PPO agent for {total_timesteps:,} timesteps...")
    print(f"Using {n_envs} parallel environments")
    print(f"Policy network: {model.policy}\n")

    # ── Train ──
    callback = TrainingCallback()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # ── Save the trained model ──
    model_path = "data/results/ppo_hedging_agent"
    model.save(model_path)
    print(f"\nModel saved to {model_path}.zip")

    # ── Plot learning curve ──
    if len(callback.rewards) > 0:
        plt.figure(figsize=(10, 4))
        # Smooth with rolling average to reduce noise
        rewards = np.array(callback.rewards)
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(callback.timesteps[:len(smoothed)], smoothed,
                 color="royalblue", label="Smoothed reward")
        plt.plot(callback.timesteps, rewards,
                 color="lightblue", alpha=0.4, label="Raw reward")
        plt.title("PPO Training Curve — Option Hedging")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig("data/results/training_curve.png", dpi=150)
        print("Learning curve saved to data/results/training_curve.png")

    return model


if __name__ == "__main__":
    model = train_ppo_agent(total_timesteps=200_000)
    print("\nTraining complete.")