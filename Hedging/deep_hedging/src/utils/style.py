"""One visual identity per strategy, shared by every figure."""
import matplotlib.pyplot as plt

STYLE = {
    "No-transaction band": dict(color="#2a78d6", ls="-",  lw=2.2),
    "Feed-forward":        dict(color="#eb6834", ls="-",  lw=1.8),
    "PPO v1 (part A)":     dict(color="#1baf7a", ls="-",  lw=1.8),
    "Whalley-Wilmott":     dict(color="#52514e", ls="--", lw=1.5),
    "Leland delta":        dict(color="#52514e", ls="-.", lw=1.5),
    "BS delta":            dict(color="#0b0b0b", ls=":",  lw=1.5),
    "No hedge":            dict(color="#a8a7a1", ls="-",  lw=1.2),
}


def style(name, **overrides):
    """Matplotlib kwargs for a strategy; overrides win (e.g. lw=1 for noisy curves)."""
    return {**STYLE[name], **overrides}


def setup():
    """Recessive axes: light grid, no top/right frame, readable fonts."""
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2f2e2b",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "font.size": 11, "legend.frameon": False,
    })