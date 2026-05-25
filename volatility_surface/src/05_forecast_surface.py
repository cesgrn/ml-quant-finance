import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# DATA — historical VIX term structure
# ─────────────────────────────────────────────

def fetch_vix_term_structure(start: str = "2018-01-01",
                              end: str = "2025-12-31") -> pd.DataFrame:
    """
    Fetch historical VIX term structure from Yahoo Finance.

    We use 3 VIX indices as a proxy for the ATM IV term structure:
    - ^VIX   : 30-day implied volatility
    - ^VIX3M : 90-day implied volatility
    - ^VIX6M : 180-day implied volatility

    This gives us the "backbone" of the IV surface — how ATM vol
    varies across maturities — over multiple years of history.
    """
    tickers = {
        "^VIX":   "iv_30d",
        "^VIX3M": "iv_90d",
        "^VIX6M": "iv_180d",
    }

    dfs = {}
    for ticker, col_name in tickers.items():
        print(f"Fetching {ticker}...")
        data = yf.download(ticker, start=start, end=end, progress=False)["Close"]
        # Flatten MultiIndex if present
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        dfs[col_name] = data / 100.0
        print(f"  Got {len(data)} days")

    # Build DataFrame from Series with aligned index
    df = pd.concat(dfs, axis=1).dropna()
    df.index = pd.to_datetime(df.index)

    print(f"\nLoaded {len(df)} trading days ({df.index[0].date()} "
          f"to {df.index[-1].date()})")
    print(df.describe().round(4))
    return df


# ─────────────────────────────────────────────
# DATASET — rolling window sequences
# ─────────────────────────────────────────────

def build_sequences(df: pd.DataFrame,
                    lookback: int = 20) -> tuple:
    """
    Build input/output sequences for the forecaster.

    Input  (X): last `lookback` days of 4-point term structure
                shape: (N, lookback, 4)
    Output (y): next day's 4-point term structure
                shape: (N, 4)

    This is a standard time-series forecasting setup — the model
    learns: given the past 20 days of VIX term structure,
    predict tomorrow's.

    We normalize each feature to [0,1] using min-max scaling
    to help training stability.
    """
    values = df.values.astype(np.float32)

    # Min-max normalization per column
    v_min = values.min(axis=0)
    v_max = values.max(axis=0)
    values_norm = (values - v_min) / (v_max - v_min + 1e-8)

    X, y = [], []
    for i in range(lookback, len(values_norm)):
        X.append(values_norm[i-lookback:i])   # past lookback days
        y.append(values_norm[i])               # next day

    X = np.array(X)   # shape: (N, lookback, 4)
    y = np.array(y)   # shape: (N, 4)

    # Train/test split — 80/20, no shuffling (time series!)
    split = int(len(X) * 0.8)
    return (torch.tensor(X[:split]), torch.tensor(y[:split]),
            torch.tensor(X[split:]), torch.tensor(y[split:]),
            v_min, v_max)


# ─────────────────────────────────────────────
# MODEL — MLP forecaster
# ─────────────────────────────────────────────

class TermStructureForecaster(nn.Module):
    """
    MLP forecaster for the VIX term structure.

    Input : flattened sequence of past 20 days × 4 points = 80 features
    Output: next day's 4-point term structure

    We use a simple MLP rather than LSTM for two reasons:
    1. The term structure has strong mean-reversion — local patterns
       matter more than long-range dependencies
    2. MLPs are easier to train and interpret

    Architecture inspired by Horvath et al.: 4 hidden layers, ELU activation.
    """

    def __init__(self, lookback: int = 20, n_features: int = 4,
                 hidden_size: int = 64):
        super().__init__()

        input_size = lookback * n_features   # flatten the sequence

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, n_features),
            # Sigmoid output: keeps predictions in [0, 1]
            # (since inputs are normalized to [0, 1])
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten: (batch, lookback, features) → (batch, lookback*features)
        x = x.reshape(x.shape[0], -1)
        return self.network(x)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_forecaster(X_train, y_train, X_test, y_test,
                     n_epochs: int = 300, lr: float = 1e-3,
                     batch_size: int = 64) -> TermStructureForecaster:

    model = TermStructureForecaster(
        lookback=X_train.shape[1],
        n_features=X_train.shape[2]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []

    print(f"Training forecaster for {n_epochs} epochs...")
    print(f"  Train samples: {len(X_train)} | Test samples: {len(X_test)}\n")

    for epoch in range(n_epochs):
        model.train()

        # Mini-batch training
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches

        # Test loss
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test).item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | Train: {train_loss:.6f} "
                  f"| Test: {test_loss:.6f}")

    return model, train_losses, test_losses


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_forecaster(model, X_test, y_test, v_min, v_max):
    """
    Evaluate forecast accuracy in original IV units (%).

    We denormalize predictions and ground truth back to original scale
    to report RMSE in interpretable units.
    """
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_test).numpy()

    y_test_norm = y_test.numpy()

    # Denormalize
    y_pred = y_pred_norm * (v_max - v_min) + v_min
    y_true = y_test_norm * (v_max - v_min) + v_min

    maturities = [30, 90, 180]
    print("\n--- Forecast Accuracy (RMSE in IV % points) ---")
    for i, dte in enumerate(maturities):
        rmse = np.sqrt(np.mean((y_pred[:, i] - y_true[:, i])**2))
        print(f"  {dte:3d}d IV : RMSE = {rmse:.4f} ({rmse:.2%})")

    overall_rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    print(f"  Overall  : RMSE = {overall_rmse:.4f} ({overall_rmse:.2%})")

    return y_pred, y_true


if __name__ == "__main__":
    os.makedirs("data/processed/plots", exist_ok=True)

    # --- Fetch data ---
    df = fetch_vix_term_structure()
    df.to_csv("data/processed/vix_term_structure.csv")

    # --- Build sequences ---
    LOOKBACK = 20
    X_tr, y_tr, X_te, y_te, v_min, v_max = build_sequences(df, lookback=LOOKBACK)

    # --- Train ---
    model, train_losses, test_losses = train_forecaster(
        X_tr, y_tr, X_te, y_te, n_epochs=300
    )
    torch.save(model.state_dict(), "data/processed/iv_forecaster.pt")
    print("\nModel saved to data/processed/iv_forecaster.pt")

    # --- Evaluate ---
    y_pred, y_true = evaluate_forecaster(model, X_te, y_te, v_min, v_max)

    # --- Plot 1: Training curves ---
    plt.figure(figsize=(10, 3))
    plt.plot(train_losses, label="Train", color="royalblue")
    plt.plot(test_losses, label="Test", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("IV Forecaster Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/processed/plots/forecaster_training.png", dpi=150)
    plt.close()

    # --- Plot 2: Predicted vs actual for each maturity ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    maturities = [30, 90, 180]
    colors = ["royalblue", "seagreen", "darkorange"]

    for ax, dte, color, i in zip(axes.ravel(), maturities, colors, range(3)):
        n_plot = min(200, len(y_true))   # plot last 200 days
        ax.plot(y_true[-n_plot:, i] * 100, label="Actual",
                color=color, alpha=0.7, linewidth=1.5)
        ax.plot(y_pred[-n_plot:, i] * 100, label="Predicted",
                color="black", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_title(f"{dte}-day IV", fontsize=11)
        ax.set_ylabel("Implied Volatility (%)")
        ax.set_xlabel("Trading days")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("IV Term Structure Forecast — Predicted vs Actual",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("data/processed/plots/forecast_vs_actual.png", dpi=150)
    plt.close()
    print("Plots saved.")