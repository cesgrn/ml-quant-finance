import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def plot_3d_surface(df_iv: pd.DataFrame, title: str, filename: str):
    """
    Plot the implied volatility surface in 3D.

    X axis = moneyness (K/S) — the skew dimension
    Y axis = days to expiry   — the term structure dimension
    Z axis = implied volatility

    We interpolate the scattered data onto a regular grid
    using scipy griddata before plotting.
    """
    from scipy.interpolate import griddata

    # Extract coordinates and values
    x = df_iv["moneyness"].values           # K/S
    y = df_iv["days_to_expiry"].values      # DTE
    z = df_iv["iv_computed"].values         # IV

    # Create regular grid for smooth surface
    # 50 points in each dimension
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate scattered IV data onto grid
    # method='cubic' gives smooth interpolation
    zi = griddata((x, y), z, (xi, yi), method="cubic")

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface with color map
    surf = ax.plot_surface(
        xi, yi, zi,
        cmap=cm.RdYlGn_r,     # red=high vol, green=low vol
        alpha=0.85,
        linewidth=0,
        antialiased=True
    )

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.1, label="Implied Volatility")

    # Labels and formatting
    ax.set_xlabel("Moneyness (K/S)", fontsize=10, labelpad=10)
    ax.set_ylabel("Days to Expiry", fontsize=10, labelpad=10)
    ax.set_zlabel("Implied Volatility", fontsize=10, labelpad=10)
    ax.set_title(title, fontsize=13, pad=20)

    # Format Z axis as percentage
    ax.zaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, pos: f"{val:.0%}")
    )

    # Best viewing angle
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_skew_slices(df_iv: pd.DataFrame, filename: str):
    """
    Plot volatility skew for several maturities on the same chart.

    The skew shows how IV varies across strikes for a fixed maturity.
    Key features to observe:
    - Left skew: OTM puts (K/S < 1) have higher IV than ATM
      → market fears crashes more than rallies
    - Smile: both OTM puts and OTM calls have elevated IV
    """
    # Select representative maturities spread across the term structure
    target_dtes = [10, 30, 60, 90, 180, 365]
    available_dtes = sorted(df_iv["days_to_expiry"].unique())

    # Find closest available DTE for each target
    selected_dtes = []
    for target in target_dtes:
        closest = min(available_dtes, key=lambda x: abs(x - target))
        if closest not in selected_dtes:
            selected_dtes.append(closest)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(selected_dtes)))

    for dte, color in zip(selected_dtes, colors):
        subset = df_iv[df_iv["days_to_expiry"] == dte].sort_values("moneyness")
        if len(subset) < 3:
            continue
        ax.plot(
            subset["moneyness"],
            subset["iv_computed"],
            "o-",
            color=color,
            linewidth=1.5,
            markersize=3,
            label=f"{dte}d",
            alpha=0.85
        )

    ax.axvline(1.0, color="gray", linestyle="--",
               alpha=0.5, label="ATM (K/S=1)")
    ax.set_xlabel("Moneyness (K/S)", fontsize=11)
    ax.set_ylabel("Implied Volatility", fontsize=11)
    ax.set_title("SPY Volatility Skew by Maturity", fontsize=13)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, pos: f"{val:.0%}")
    )
    ax.legend(title="DTE", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_term_structure(df_iv: pd.DataFrame, filename: str):
    """
    Plot the ATM term structure — how IV varies with maturity at ATM.

    ATM = options closest to moneyness = 1.0 (K = S).

    Key features to observe:
    - Inverted term structure (short > long): market fears near-term events
    - Normal term structure (short < long): calm short-term, uncertainty long-term
    - Humped: some intermediate maturity has highest vol
    """
    records = []
    for dte in sorted(df_iv["days_to_expiry"].unique()):
        subset = df_iv[df_iv["days_to_expiry"] == dte]
        # Find option closest to ATM (moneyness closest to 1.0)
        idx = (subset["moneyness"] - 1.0).abs().idxmin()
        atm_iv = subset.loc[idx, "iv_computed"]
        atm_moneyness = subset.loc[idx, "moneyness"]
        records.append({
            "dte": dte,
            "atm_iv": atm_iv,
            "atm_moneyness": atm_moneyness
        })

    ts = pd.DataFrame(records).sort_values("dte")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts["dte"], ts["atm_iv"], "o-",
            color="royalblue", linewidth=2, markersize=5)
    ax.fill_between(ts["dte"], ts["atm_iv"],
                    alpha=0.15, color="royalblue")
    ax.set_xlabel("Days to Expiry", fontsize=11)
    ax.set_ylabel("ATM Implied Volatility", fontsize=11)
    ax.set_title("SPY ATM Term Structure of Volatility", fontsize=13)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, pos: f"{val:.0%}")
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    os.makedirs("data/processed/plots", exist_ok=True)

    # Load computed IVs
    df = pd.read_csv("data/processed/iv_surface_raw.csv")
    print(f"Loaded {len(df)} options with computed IVs\n")

    # 3D surface
    print("Plotting 3D surface...")
    plot_3d_surface(
        df,
        title="SPY Implied Volatility Surface",
        filename="data/processed/plots/raw_surface_3d.png"
    )

    # Skew slices
    print("Plotting volatility skew...")
    plot_skew_slices(df, "data/processed/plots/skew_slices.png")

    # ATM term structure
    print("Plotting term structure...")
    plot_term_structure(df, "data/processed/plots/term_structure.png")

    print("\nAll plots saved to data/processed/plots/")