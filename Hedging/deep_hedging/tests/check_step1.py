import math, sys
sys.path.append("src")
import config as cfg
from utils.market import simulate_gbm, simulate_gbm_torch

paths = simulate_gbm(cfg.N_TEST_PATHS, cfg.S0, cfg.R, cfg.SIGMA, cfg.T, cfg.N_STEPS, cfg.TEST_SEED)
print("shape:", tuple(paths.shape))
print("first column:", paths[:3, 0].tolist())
final = paths[:, -1]
print("mean of S_T:", float(final.mean()))
print("mean of e^(-rT) S_T:", float((final * math.exp(-cfg.R * cfg.T)).mean()))