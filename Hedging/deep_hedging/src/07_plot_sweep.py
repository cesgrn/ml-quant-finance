import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.style import setup, style

MARKERS = {"BS delta": "D", "Leland delta": "^",
           "Whalley-Wilmott": "s", "No-transaction band": "o"}
REF = "#a8a7a1"   # light grey for the theoretical slope guides

if __name__ == "__main__":
    setup()
    results = pd.read_csv("data/results/cost_sweep.csv")
    bands = pd.read_csv("data/results/cost_sweep_bands.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: indifference price vs cost
    for name, marker in MARKERS.items():
        block = results[results["strategy"] == name]
        ax1.plot(block["cost"] * 100, block["price"], marker=marker, ms=6,
                 label=name, **style(name))
    ax1.set_xlabel("Transaction cost (%)")
    ax1.set_ylabel("Indifference price")
    ax1.set_title("Price a dealer must charge")
    ax1.legend()

    # Right: band half-width vs cost, log-log
    b = bands[bands["cost"] > 0]
    c, w = b["cost"] * 100, b["learned_half_width_atm"]
    ax2.loglog(c, w, marker="o", ms=6, label="Learned band",
               **style("No-transaction band"))
    ax2.loglog(c, b["ww_half_width_atm"], marker="s", ms=6, label="Whalley-Wilmott",
               **style("Whalley-Wilmott"))
    for slope, ls in [(1 / 3, ":"), (1 / 2, "-.")]:
        ax2.loglog(c, w.iloc[0] * (c / c.iloc[0]) ** slope, ls=ls, color=REF, lw=1.2,
                   label=f"slope {slope:.2g}")
    ax2.set_xticks([0.1, 0.25, 0.5, 1, 2], labels=["0.1", "0.25", "0.5", "1", "2"])
    ax2.set_yticks([0.01, 0.02, 0.05, 0.1], labels=["0.01", "0.02", "0.05", "0.1"])
    ax2.minorticks_off()
    ax2.set_xlabel("Transaction cost (%)")
    ax2.set_ylabel("Band half-width at S = K, tau = 0.5")
    ax2.set_title("How the band widens with costs")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("data/results/cost_sweep.png", dpi=150)