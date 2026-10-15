import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils import black_scholes as bs
from utils.hedging import market_state
from utils.models import NoTransactionBandNet
from utils.strategies import leland_volatility, ww_half_width
from utils.style import STYLE, setup, style

if __name__ == "__main__":
    setup()
    cost = cfg.COST
    dt = cfg.T / cfg.N_STEPS
    sigma_leland = leland_volatility(cfg.SIGMA, cost, dt)

    model = NoTransactionBandNet(cfg.T)
    model.load_state_dict(torch.load(f"data/results/no-transaction_band_{cost}.pt"))
    model.eval()

    spots = torch.linspace(70, 130, 300, dtype=torch.float64)
    x = spots.numpy()
    band_color = STYLE["No-transaction band"]["color"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    with torch.no_grad():
        for ax, tau in zip(axes, [0.9, 0.5, 0.1]):
            s = market_state(spots, tau, cfg.K, cfg.R, cfg.SIGMA)
            lower, upper = model.band(s)
            w = ww_half_width(s, cost, cfg.RISK_AVERSION, cfg.R)
            leland = bs.call_delta(spots, cfg.K, tau, cfg.R, sigma_leland)

            # Learned band: filled area plus thin edges, in the band network's colour
            ax.fill_between(x, lower.numpy(), upper.numpy(), color=band_color,
                            alpha=0.22, lw=0, label="Learned band")
            ax.plot(x, lower.numpy(), color=band_color, lw=1.2)
            ax.plot(x, upper.numpy(), color=band_color, lw=1.2)

            # Classic benchmarks: grey/black, told apart by line style
            ax.plot(x, (s.bs_delta - w).numpy(), label="Whalley-Wilmott band",
                    **style("Whalley-Wilmott"))
            ax.plot(x, (s.bs_delta + w).numpy(), **style("Whalley-Wilmott"))
            ax.plot(x, leland.numpy(), label="Leland delta", **style("Leland delta"))
            ax.plot(x, s.bs_delta.numpy(), label="BS delta", **style("BS delta"))

            ax.axvline(cfg.K, color="#c3c2b7", lw=0.8, zorder=0)
            ax.set_title(f"Time to maturity: {tau:.1f} year")
            ax.set_xlabel("Stock price S")

            atm = torch.argmin((spots - cfg.K).abs())
            print(f"tau={tau}: at S=K, learned band = [{lower[atm]:.3f}, {upper[atm]:.3f}], "
                  f"WW band = [{(s.bs_delta - w)[atm]:.3f}, {(s.bs_delta + w)[atm]:.3f}]")

    axes[0].set_ylabel("Hedge ratio (shares)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"No-transaction bands, cost = {cost:.0%}, lambda = {cfg.RISK_AVERSION}")
    fig.tight_layout()
    fig.savefig(f"data/results/bands_{cost}.png", dpi=150)