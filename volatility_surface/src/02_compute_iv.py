import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.black_scholes import implied_volatility

# Risk-free rate — use approximate 3-month Treasury rate
RISK_FREE_RATE = 0.04   # 4% — approximate current level

def compute_iv_surface(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute implied volatility for each option in the dataset.

    For each option we need:
    - market_price : use the mid price (bid+ask)/2 — more reliable than lastPrice
    - S            : spot price
    - K            : strike price
    - T            : time to expiry in years (days / 365)
    - r            : risk-free rate

    We then invert Black-Scholes numerically to find the sigma that
    makes the model price equal the market mid price.
    """
    df = df.copy()

    # Use mid price — more reliable than lastPrice (last trade could be stale)
    df["mid"] = (df["bid"] + df["ask"]) / 2

    # Time to expiry in years
    df["T"] = df["days_to_expiry"] / 365.0

    print(f"Computing IV for {len(df)} options...")
    print("This may take 30-60 seconds...\n")

    ivs = []
    for i, row in df.iterrows():
        iv = implied_volatility(
            market_price=row["mid"],
            S=row["spot"],
            K=row["strike"],
            T=row["T"],
            r=RISK_FREE_RATE
        )
        ivs.append(iv)

        if i % 500 == 0:
            print(f"  Processed {i}/{len(df)} options...")

    df["iv_computed"] = ivs

    # Drop options where IV computation failed
    n_before = len(df)
    df = df.dropna(subset=["iv_computed"])
    n_after = len(df)
    print(f"\nValid IVs computed: {n_after} / {n_before}")

    # Remove extreme IVs (< 1% or > 300% — data errors)
    df = df[(df["iv_computed"] > 0.01) & (df["iv_computed"] < 3.0)]
    print(f"After IV range filter: {len(df)} options")

    # Summary statistics
    print(f"\nIV statistics:")
    print(f"  Mean IV  : {df['iv_computed'].mean():.2%}")
    print(f"  Min IV   : {df['iv_computed'].min():.2%}")
    print(f"  Max IV   : {df['iv_computed'].max():.2%}")
    print(f"  Std IV   : {df['iv_computed'].std():.2%}")

    return df


def build_surface_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the IV surface as a pivot table.
    Rows    = expiry dates (term structure dimension)
    Columns = moneyness buckets (skew dimension)

    We use moneyness (K/S) instead of raw strikes so the surface
    is comparable across different spot prices and dates.
    This is the standard representation used in both papers.
    """
    # Round moneyness to 2 decimal places for grid alignment
    df["moneyness_round"] = df["moneyness"].round(2)

    # Pivot: rows = DTE, columns = moneyness, values = IV
    surface = df.pivot_table(
        values="iv_computed",
        index="days_to_expiry",
        columns="moneyness_round",
        aggfunc="mean"   # average if multiple options at same point
    )

    print(f"\nIV Surface grid shape: {surface.shape}")
    print(f"  Maturities : {surface.index.tolist()}")
    print(f"  Moneyness range: "
          f"{surface.columns.min():.2f} — {surface.columns.max():.2f}")

    return surface


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    # Load liquid options
    df = pd.read_csv("data/raw/spy_options_liquid.csv")
    print(f"Loaded {len(df)} liquid options\n")

    # Compute IVs
    df_iv = compute_iv_surface(df)
    df_iv.to_csv("data/processed/iv_surface_raw.csv", index=False)
    print("\nSaved to data/processed/iv_surface_raw.csv")

    # Build grid
    surface = build_surface_grid(df_iv)
    surface.to_csv("data/processed/iv_grid.csv")
    print("IV grid saved to data/processed/iv_grid.csv")