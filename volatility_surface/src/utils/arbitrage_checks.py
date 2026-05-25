import torch

def calendar_spread_penalty(model, moneyness_grid: torch.Tensor,
                             dte_grid: torch.Tensor) -> torch.Tensor:
    """
    Calendar spread arbitrage penalty.

    No calendar arbitrage requires that total variance is
    non-decreasing in time to expiry:
        w(K, T) = sigma(K,T)^2 * T  must satisfy dw/dT >= 0

    We enforce this by penalizing violations:
        penalty = mean(max(0, w(T1) - w(T2))^2) for T1 < T2

    This is a soft constraint — it adds to the loss but doesn't
    hard-enforce the condition. Inspired by Ackerer et al. (2020).
    """
    penalties = []

    # For each moneyness level, check across adjacent maturities
    for k in moneyness_grid:
        k_tensor = k.expand(len(dte_grid))
        inputs = torch.stack([k_tensor, dte_grid], dim=1)

        with torch.no_grad():
            ivs = model(inputs).squeeze()

        # Total variance = sigma^2 * T
        total_var = ivs**2 * dte_grid

        # Penalty for decreasing total variance
        for i in range(len(total_var) - 1):
            violation = torch.relu(total_var[i] - total_var[i+1])
            penalties.append(violation**2)

    if penalties:
        return torch.stack(penalties).mean()
    return torch.tensor(0.0)


def butterfly_penalty(model, moneyness_grid: torch.Tensor,
                      dte_val: torch.Tensor) -> torch.Tensor:
    """
    Butterfly arbitrage penalty.

    No butterfly arbitrage requires that the IV surface is
    convex in log-moneyness — equivalently, the local volatility
    must be positive everywhere.

    We approximate this by penalizing negative second derivatives
    of IV w.r.t. moneyness:
        d^2(sigma)/d(k^2) >= 0

    Computed via finite differences on the moneyness grid.
    """
    if len(moneyness_grid) < 3:
        return torch.tensor(0.0)

    dte_expanded = dte_val.expand(len(moneyness_grid))
    inputs = torch.stack([moneyness_grid, dte_expanded], dim=1)
    ivs = model(inputs).squeeze()

    # Second derivative via finite differences
    # d2iv/dk2 ≈ (iv[i+1] - 2*iv[i] + iv[i-1]) / dk^2
    dk = moneyness_grid[1] - moneyness_grid[0]
    d2iv = (ivs[2:] - 2 * ivs[1:-1] + ivs[:-2]) / (dk**2)

    # Penalize negative curvature (concavity = butterfly arbitrage)
    penalty = torch.relu(-d2iv)**2
    return penalty.mean()