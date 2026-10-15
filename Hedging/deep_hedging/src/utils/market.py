"""Price paths for the underlying (geometric Brownian motion)."""
import math

import numpy as np
import torch


def simulate_gbm(n_paths: int, s0: float, r: float, sigma: float, T: float,
                 n_steps: int, seed: int, dtype=torch.float64) -> torch.Tensor:
    """
    Exact GBM simulation under the risk-neutral measure.

    Uses NumPy's legacy generator with the same draw order as part A's
    `simulate_gbm`, so a given seed gives exactly the same paths in both parts.
    Generating on the CPU with NumPy also makes paths identical on every machine
    (CPU, CUDA or Apple MPS).

    Returns a tensor of shape (n_paths, n_steps + 1); column 0 is s0.
    """
    rng = np.random.RandomState(seed)
    z = rng.standard_normal((n_paths, n_steps))
    dt = T / n_steps
    log_increments = (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return torch.as_tensor(s0 * np.exp(log_paths), dtype=dtype)


def simulate_gbm_torch(n_paths: int, s0: float, r: float, sigma: float, T: float,
                       n_steps: int, seed=None, device="cpu",
                       dtype=torch.float32) -> torch.Tensor:
    """
    Same model, generated directly with PyTorch.

    Used for training, where a fresh batch of paths is drawn at every step:
    it avoids copying from NumPy and runs on the GPU if available.
    Note: Apple MPS does not support float64, so training uses float32.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    z = torch.randn(n_paths, n_steps, generator=generator, dtype=dtype, device="cpu").to(device)
    dt = T / n_steps
    log_increments = (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    zeros = torch.zeros(n_paths, 1, dtype=dtype, device=device)
    return s0 * torch.exp(torch.cat([zeros, log_increments.cumsum(dim=1)], dim=1))


def time_grid(T: float, n_steps: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Hedging dates t_0 = 0, ..., t_n = T (exact, no floating-point truncation)."""
    return torch.arange(n_steps + 1, dtype=dtype, device=device) * (T / n_steps)
