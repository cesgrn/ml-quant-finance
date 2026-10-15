import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils.black_scholes import black_scholes_price, delta
from utils.simulation import simulate_gbm

class OptionHedgingEnv(gym.Env):
    """
    Gymnasium environment for hedging a short call option.

    The agent is SHORT one call option — it sold the option and collected
    the premium. Its job is to hold shares of the underlying stock to
    offset the risk of the option moving against it.

    At each timestep the agent decides how many shares to hold (hedge ratio).
    The reward penalizes both P&L variance and transaction costs.
    """

    def __init__(self,
                 S0: float = 100.0,      # initial stock price
                 K: float = 100.0,       # strike price of the option
                 T: float = 1.0,         # time to expiry in years
                 r: float = 0.05,        # risk-free rate
                 sigma: float = 0.2,     # volatility of the underlying
                 n_steps: int = 50,      # number of hedging steps per episode
                 transaction_cost: float = 0.01,  # cost per unit traded (1%)
                 ):

        super().__init__()

        # Store environment parameters
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps            # size of each time step
        self.transaction_cost = transaction_cost

        # --- Action space ---
        # The agent chooses a hedge ratio in [0, 1]
        # 0 = hold no shares (fully unhedged)
        # 1 = hold 1 share per short call (fully hedged)
        # We use a continuous action space as in Deep Hedging paper
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        # --- Observation space ---
        # The agent observes 4 features at each timestep:
        # [current stock price, time to expiry, current delta, current hedge position]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Internal state — reset at the start of each episode
        self.current_step = None
        self.stock_price = None
        self.current_hedge = None
        self.option_price_initial = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment at the start of a new episode.
        Generates a new GBM price path and sets the initial state.
        """
        super().reset(seed=seed)

        # Generate a single GBM price path for this episode
        # Shape: (1, n_steps + 1) — one path, n_steps+1 prices
        path = simulate_gbm(
            S0=self.S0,
            mu=self.r,           # use risk-neutral drift (r, not mu)
            sigma=self.sigma,
            T=self.T,
            n_steps=self.n_steps,
            n_paths=1,
            seed=np.random.randint(0, 100000)  # random seed for variety
        )

        # Store the full path — the agent will walk through it step by step
        self.price_path = path[0]       # shape: (n_steps + 1,)
        self.current_step = 0
        self.stock_price = self.price_path[0]
        self.current_hedge = 0.0        # start with no hedge position

        # Record the initial option price (premium collected when selling)
        self.option_price_initial = black_scholes_price(
            S=self.stock_price,
            K=self.K,
            T=self.T,
            r=self.r,
            sigma=self.sigma
        )

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        Build the observation vector the agent sees at each step.
        Contains enough information to compute the theoretical delta hedge.
        """
        # Time remaining as a fraction of total duration
        time_remaining = 1.0 - (self.current_step / self.n_steps)

        # Theoretical Black-Scholes delta at current state
        # This is what a classical hedger would use
        bs_delta = delta(
            S=self.stock_price,
            K=self.K,
            T=time_remaining * self.T,   # actual time left in years
            r=self.r,
            sigma=self.sigma
        )

        return np.array([
            self.stock_price / self.S0,  # normalized price (starts at 1.0)
            time_remaining,              # fraction of time remaining [1→0]
            bs_delta,                    # theoretical delta [0→1]
            self.current_hedge,          # current hedge position [0→1]
        ], dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        Execute one hedging step.

        action: hedge ratio chosen by the agent, shape (1,), values in [0,1]

        Returns:
            observation : new state
            reward      : P&L of this step minus transaction cost
            terminated  : True if episode is done (option expired)
            truncated   : always False
            info        : dict with debug info
        """
        # Extract scalar action from array
        new_hedge = float(action[0])

        # --- Transaction cost ---
        # Cost is proportional to how much we change our hedge position
        # |new_hedge - current_hedge| is the number of shares traded
        # Multiplied by stock price to get dollar cost
        hedge_change = abs(new_hedge - self.current_hedge)
        tc = self.transaction_cost * hedge_change * self.stock_price

        # Move to next time step
        self.current_step += 1
        new_stock_price = self.price_path[self.current_step]

        # --- P&L calculation ---
        # We are SHORT one call option, LONG (hedge) shares
        # P&L = gain from stock position - loss from option value change - transaction cost
        stock_pnl = self.current_hedge * (new_stock_price - self.stock_price)

        # Option P&L: we are short, so we LOSE when option value increases
        time_remaining_old = 1.0 - ((self.current_step - 1) / self.n_steps)
        time_remaining_new = 1.0 - (self.current_step / self.n_steps)

        option_price_old = black_scholes_price(
            S=self.stock_price, K=self.K,
            T=max(time_remaining_old * self.T, 1e-6),
            r=self.r, sigma=self.sigma
        )
        option_price_new = black_scholes_price(
            S=new_stock_price, K=self.K,
            T=max(time_remaining_new * self.T, 1e-6),
            r=self.r, sigma=self.sigma
        )

        # Short option: we lose when option price goes up
        option_pnl = -(option_price_new - option_price_old)

        # Total P&L this step
        pnl = stock_pnl + option_pnl - tc

        # --- Reward ---
        # We want to minimize variance of P&L, not just maximize average
        # Penalize squared P&L to discourage large swings in either direction
        reward = -pnl**2 - tc

        # Update state
        self.stock_price = new_stock_price
        self.current_hedge = new_hedge

        # Episode ends when option expires
        terminated = (self.current_step >= self.n_steps)

        return self._get_observation(), reward, terminated, False, {
            "pnl": pnl,
            "stock_pnl": stock_pnl,
            "option_pnl": option_pnl,
            "transaction_cost": tc,
            "bs_delta": delta(S=self.stock_price, K=self.K,
                             T=max(time_remaining_new * self.T, 1e-6),
                             r=self.r, sigma=self.sigma)
        }


if __name__ == "__main__":
    # Quick sanity check — run one episode with random actions
    env = OptionHedgingEnv()
    obs, _ = env.reset()

    print(f"Initial observation : {obs}")
    print(f"  stock_price_norm  : {obs[0]:.4f}")
    print(f"  time_remaining    : {obs[1]:.4f}")
    print(f"  bs_delta          : {obs[2]:.4f}")
    print(f"  current_hedge     : {obs[3]:.4f}")

    total_pnl = 0
    for step in range(env.n_steps):
        # Random action for now — RL agent will replace this
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        total_pnl += info["pnl"]
        if terminated:
            break

    print(f"\nEpisode complete")
    print(f"Total P&L         : {total_pnl:.4f}")
    print(f"Final stock price : {env.stock_price:.2f}")