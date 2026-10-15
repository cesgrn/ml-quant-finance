import sys, torch
sys.path.append("src")
from utils import black_scholes as bs

S = torch.tensor([100.0], dtype=torch.float64)
print("Price:", bs.call_price(S, 100.0, 1.0, 0.05, 0.2).item())
print("Delta:", bs.call_delta(S, 100.0, 1.0, 0.05, 0.2).item())

h = 1e-3
slope = (bs.call_price(S + h, 100.0, 1.0, 0.05, 0.2) - bs.call_price(S - h, 100.0, 1.0, 0.05, 0.2)) / (2 * h)
print("Numerical slope:", float(slope))

g_num = (bs.call_delta(S + h, 100.0, 1.0, 0.05, 0.2) - bs.call_delta(S - h, 100.0, 1.0, 0.05, 0.2)) / (2 * h)
print("Gamma", float(bs.call_gamma(S, 100.0, 1.0, 0.05, 0.2)), "| Numerical:", float(g_num))

S_range = torch.tensor([60.0, 80.0, 100.0, 120.0, 140.0], dtype=torch.float64)
print("Deltas:", bs.call_delta(S_range, 100.0, 1.0, 0.05, 0.2).tolist())