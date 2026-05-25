import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_options_chain(ticker: str = "SPY") -> pd.DataFrame:
    """
    Fetch the full options chain for a ticker across all available expiry dates.

    yfinance returns options data as two DataFrames per expiry:
    - calls: all call options at that expiry
    - puts: all put options at that expiry

    We collect both across all expiries and tag each row with:
    - option_type: 'call' or 'put'
    - expiry: the expiration date
    - days_to_expiry: calendar days from today to expiry

    We focus on calls for IV surface construction (standard practice)
    but keep puts for comparison.
    """
    stock = yf.Ticker(ticker)

    # Get current stock price for reference
    spot = stock.fast_info["lastPrice"]
    print(f"{ticker} spot price: ${spot:.2f}")

    # Get all available expiry dates
    expiries = stock.options
    print(f"Available expiries: {len(expiries)}")
    for e in expiries:
        print(f"  {e}")

    today = datetime.today()
    records = []

    for expiry in expiries:
        # Compute days to expiry
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        dte = (expiry_dt - today).days

        # Skip expired or very near-term options (< 5 days)
        # and very long-dated options (> 400 days) — too illiquid
        if dte < 5 or dte > 400:
            continue

        # Fetch calls and puts for this expiry
        chain = stock.option_chain(expiry)

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["option_type"] = opt_type
            df["expiry"] = expiry
            df["days_to_expiry"] = dte
            df["spot"] = spot
            # Moneyness = strike / spot — useful for filtering liquid options
            df["moneyness"] = df["strike"] / spot
            records.append(df)

        print(f"  {expiry} ({dte} days): "
              f"{len(chain.calls)} calls, {len(chain.puts)} puts")

    df_all = pd.concat(records, ignore_index=True)
    print(f"\nTotal options collected: {len(df_all)}")
    print(f"Columns: {df_all.columns.tolist()}")
    return df_all, spot


def filter_liquid_options(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Keep only liquid options — those with meaningful market prices.

    Filters applied:
    1. Calls only (standard for IV surface construction)
    2. Remove options with zero bid (not traded)
    3. Keep moneyness between 0.7 and 1.3 (70% to 130% of spot)
       — deeper OTM options have unreliable IVs
    4. Remove options with zero last price
    5. Remove options with very wide bid-ask spread (illiquid)
    """
    # Calls only
    df = df[df["option_type"] == "call"].copy()

    # Remove zero bid (untradeable)
    df = df[df["bid"] > 0]

    # Keep near-the-money options — 0.7 to 1.3 moneyness
    df = df[(df["moneyness"] >= 0.7) & (df["moneyness"] <= 1.3)]

    # Remove zero last price
    df = df[df["lastPrice"] > 0]

    # Remove very wide bid-ask (bid-ask > 50% of mid price indicates illiquidity)
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread_ratio"] = (df["ask"] - df["bid"]) / df["mid"]
    df = df[df["spread_ratio"] < 0.5]

    print(f"\nAfter liquidity filters: {len(df)} options")
    print(f"Expiries covered: {df['expiry'].nunique()}")
    print(f"Strike range: {df['strike'].min():.0f} — {df['strike'].max():.0f}")
    print(f"DTE range: {df['days_to_expiry'].min()} — "
          f"{df['days_to_expiry'].max()} days")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    # Fetch full options chain
    df, spot = fetch_options_chain("SPY")

    # Save raw data
    df.to_csv("data/raw/spy_options_raw.csv", index=False)
    print("\nRaw data saved to data/raw/spy_options_raw.csv")

    # Apply liquidity filters
    df_liquid = filter_liquid_options(df, spot)
    df_liquid.to_csv("data/raw/spy_options_liquid.csv", index=False)
    print("Liquid options saved to data/raw/spy_options_liquid.csv")