import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils.black_scholes import black_scholes_price, delta
from utils.simulation import simulate_gbm


class OptionHedgingEnvCVaR(gym.Env):
    """
    Improved environment using episode-level CVaR reward.

    Key difference from v1:
    - V1 reward: -pnl^2 - tc at every step (penalizes step-level variance)
    - V2 reward: -tc at every step + CVaR penalty at terminal step

    The CVaR reward is inspired by the Deep Hedging paper.
    It explicitly penalizes tail losses rather than average variance.

    CVaR formula (from Deep Hedging paper section 3):
        CVaR_alpha(X) = min_z { z + 1/(1-alpha) * E[max(-X - z, 0)] }

    We approximate this using a fixed z = 0 (conservative):
        penalty = max(-total_pnl, 0)^2

    This strongly penalizes negative total P&L outcomes
    while being indifferent to positive ones — exactly what a
    risk manager wants.
    """

    def __init__(self,
                 S0: float = 100.0,
                 K: float = 100.0,
                 T: float = 1.0,
                 r: float = 0.05,
                 sigma: float = 0.2,
                 n_steps: int = 50,
                 transaction_cost: float = 0.01,
                 cvar_lambda: float = 2.0,   # weight on CVaR penalty
                 ):

        super().__init__()

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.transaction_cost = transaction_cost
        # cvar_lambda controls how much we penalize tail losses
        # higher = more risk-averse agent
        self.cvar_lambda = cvar_lambda

        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.current_step = None
        self.stock_price = None
        self.current_hedge = None
        self.cumulative_pnl = None    # track total P&L across the episode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        path = simulate_gbm(
            S0=self.S0, mu=self.r, sigma=self.sigma,
            T=self.T, n_steps=self.n_steps, n_paths=1,
            seed=np.random.randint(0, 100000)
        )

        self.price_path = path[0]
        self.current_step = 0
        self.stock_price = self.price_path[0]
        self.current_hedge = 0.0
        self.cumulative_pnl = 0.0    # reset episode P&L accumulator

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        time_remaining = 1.0 - (self.current_step / self.n_steps)
        bs_delta = delta(
            S=self.stock_price, K=self.K,
            T=max(time_remaining * self.T, 1e-6),
            r=self.r, sigma=self.sigma
        )
        return np.array([
            self.stock_price / self.S0,
            time_remaining,
            bs_delta,
            self.current_hedge,
        ], dtype=np.float32)

    def step(self, action: np.ndarray):
        new_hedge = float(action[0])

        # --- Transaction cost ---
        hedge_change = abs(new_hedge - self.current_hedge)
        tc = self.transaction_cost * hedge_change * self.stock_price

        # --- Move to next step ---
        self.current_step += 1
        new_stock_price = self.price_path[self.current_step]

        # --- P&L ---
        stock_pnl = self.current_hedge * (new_stock_price - self.stock_price)

        time_remaining_old = 1.0 - ((self.current_step - 1) / self.n_steps)
        time_remaining_new = 1.0 - (self.current_step / self.n_steps)

        opt_old = black_scholes_price(
            S=self.stock_price, K=self.K,
            T=max(time_remaining_old * self.T, 1e-6),
            r=self.r, sigma=self.sigma
        )
        opt_new = black_scholes_price(
            S=new_stock_price, K=self.K,
            T=max(time_remaining_new * self.T, 1e-6),
            r=self.r, sigma=self.sigma
        )
        option_pnl = -(opt_new - opt_old)
        step_pnl = stock_pnl + option_pnl - tc

        # Accumulate episode P&L
        self.cumulative_pnl += step_pnl

        terminated = (self.current_step >= self.n_steps)

        if not terminated:
            # --- Intermediate reward: only penalize transaction costs ---
            # We defer the main penalty to the end of the episode
            # so the agent optimizes the full trajectory, not just each step
            reward = -tc
        else:
            # --- Terminal reward: CVaR-inspired penalty ---
            # Reward the mean P&L but heavily penalize negative outcomes
            # max(0, -pnl)^2 is zero when pnl >= 0 (no penalty for profits)
            # and grows quadratically for losses (penalizes tail risk)
            cvar_penalty = self.cvar_lambda * max(0, -self.cumulative_pnl) ** 2
            reward = self.cumulative_pnl - cvar_penalty

        self.stock_price = new_stock_price
        self.current_hedge = new_hedge

        return self._get_observation(), reward, terminated, False, {
            "pnl": step_pnl,
            "cumulative_pnl": self.cumulative_pnl,
            "transaction_cost": tc,
        }