"""Risk measures (used as training losses) and evaluation metrics."""
import math

import numpy as np
import torch


def entropic_risk(pnl: torch.Tensor, risk_aversion: float = 1.0) -> torch.Tensor:
    """
    Entropic risk  rho(X) = (1/lambda) * log E[exp(-lambda X)].

    Minimizing it is equivalent to maximizing expected exponential utility, the
    objective of the No-Transaction Band paper. Computed with logsumexp so that
    large losses do not overflow, even in float32.
    """
    x = -risk_aversion * pnl
    return (torch.logsumexp(x, dim=0) - math.log(pnl.numel())) / risk_aversion


def expected_shortfall(pnl: torch.Tensor, level: float = 0.95) -> torch.Tensor:
    """ 
    Expected shortfall (CVaR) of the losses: average loss over the worst
    (1 - level) fraction of paths. Differentiable, usable as a training loss.
    Note the sign: it returns a positive number for a loss, whereas part A
    reported CVaR as the (negative) average P&L of the tail.
    """
    n_tail = max(1, int(math.ceil((1.0 - level) * pnl.numel())))
    worst = torch.topk(-pnl, n_tail).values
    return worst.mean()


def indifference_price(pnl_without_premium: torch.Tensor, risk_aversion: float = 1.0) -> float:
    """
    Price at which the dealer is indifferent between selling the option and not:
    the premium p such that rho(p + PnL_0) = 0, i.e. p = rho(PnL_0),
    where PnL_0 is the hedged P&L computed with a zero premium.
    """
    return float(entropic_risk(pnl_without_premium, risk_aversion))


def summary(pnl: torch.Tensor, level: float = 0.95, risk_aversion: float = 1.0) -> dict:
    """Same metrics as part A, plus the entropic risk."""
    x = pnl.detach().double().cpu().numpy()
    var = np.percentile(x, 100 * (1 - level))
    return {
        "mean_pnl": x.mean(),
        "std_pnl": x.std(),
        "sharpe": x.mean() / (x.std() + 1e-12),
        "var_95": var,
        "cvar_95": x[x <= var].mean(),   # part A's sign convention (negative = loss)
        "entropic_risk": float(entropic_risk(pnl.detach().double(), risk_aversion)),
    }
