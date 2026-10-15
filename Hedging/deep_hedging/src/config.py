"""
Shared parameters for every script in part B.

The market and the contract are the same as in part A (Hedging/rl_hedging):
short one at-the-money call, S0 = K = 100, T = 1 year, r = 5%, sigma = 20%,
50 rebalancing dates, 1% proportional transaction cost.
"""

# --- Market and contract (identical to part A) ---
S0 = 100.0          # initial stock price
K = 100.0           # strike of the short call
T = 1.0             # maturity in years
R = 0.05            # risk-free rate (continuously compounded)
SIGMA = 0.20        # volatility of the underlying
N_STEPS = 50        # number of hedging dates
COST = 0.01         # proportional transaction cost (1% of traded value), as in part A

# Cost levels used for the sweep in later steps (0% to 2%)
COST_GRID = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]

# --- Test set ---
# Same generator and seed as part A's delta-hedge evaluation (simulate_gbm, seed=99),
# so the first 1,000 test paths are exactly the paths part A used.
TEST_SEED = 99
N_TEST_PATHS = 20_000

# --- Risk measures ---
RISK_AVERSION = 1.0   # lambda of the entropic risk measure, in 1/dollars
CVAR_LEVEL = 0.95     # confidence level of the expected shortfall (CVaR)

# Training
DEVICE = "cpu"        # "mps" to try your Mac's GPU (see the note below)
N_ITERS = 1000        # gradient steps
BATCH_SIZE = 10_000   # fresh paths per step
LR = 1e-3             # Adam learning rate (the paper's value)
TRAIN_SEED = 0
