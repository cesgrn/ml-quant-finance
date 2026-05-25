import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# NETWORK ARCHITECTURE
# Inspired by Horvath et al. (2021):
# - 4 hidden layers, 30 neurons each
# - ELU activation (smooth, C-infinity — needed for arbitrage derivatives)
# - Input: [moneyness, T_years]
# - Output: single IV value
# ─────────────────────────────────────────────

class IVSurfaceNet(nn.Module):
    """
    Neural network for IV surface smoothing.

    Architecture from Horvath et al.:
    - 4 hidden layers x 30 neurons
    - ELU activation: smooth everywhere unlike ReLU
      Important: arbitrage penalties require computing derivatives
      of the network output, so the activation must be differentiable
    - Softplus output: ensures IV > 0 always
      (addresses the negative IV artifact in cubic interpolation)
    """

    def __init__(self, hidden_size: int = 30, n_layers: int = 4):
        super().__init__()

        layers = []
        # Input layer: 2 features → hidden
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.ELU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ELU())

        # Output layer: hidden → 1 (single IV value)
        layers.append(nn.Linear(hidden_size, 1))
        # Softplus ensures output > 0 (IV cannot be negative)
        # Softplus(x) = log(1 + exp(x)) — smooth approximation of ReLU
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_smoother(df: pd.DataFrame,
                   n_epochs: int = 500,
                   lr: float = 1e-3,
                   lambda_calendar: float = 10.0,
                   lambda_butterfly: float = 5.0) -> IVSurfaceNet:
    """
    Train the IV surface smoother with no-arbitrage penalties.

    Loss = MSE(predicted IV, observed IV)
           + lambda_calendar * calendar_penalty
           + lambda_butterfly * butterfly_penalty

    The MSE term fits the market data.
    The penalty terms enforce no-arbitrage constraints.
    Lambda values control the trade-off — inspired by Ackerer et al.

    Inputs are normalized to [-1, 1] following Horvath et al.
    """

    # --- Prepare data ---
    moneyness = df["moneyness"].values.astype(np.float32)
    T_years = (df["days_to_expiry"].values / 365.0).astype(np.float32)
    iv = df["iv_computed"].values.astype(np.float32)

    # Normalize inputs to [-1, 1] (Horvath et al. Section 3.2.2)
    m_min, m_max = moneyness.min(), moneyness.max()
    t_min, t_max = T_years.min(), T_years.max()

    def normalize_m(x):
        return 2 * (x - m_min) / (m_max - m_min) - 1

    def normalize_t(x):
        return 2 * (x - t_min) / (t_max - t_min) - 1

    m_norm = normalize_m(moneyness)
    t_norm = normalize_t(T_years)

    # Build input tensor: shape (N, 2)
    X = torch.tensor(np.stack([m_norm, t_norm], axis=1))
    y = torch.tensor(iv).unsqueeze(1)

    # Build grids for arbitrage penalty evaluation
    m_grid_raw = np.linspace(m_min, m_max, 20).astype(np.float32)
    t_grid_raw = np.linspace(t_min, t_max, 15).astype(np.float32)
    m_grid = torch.tensor(normalize_m(m_grid_raw))
    t_grid = torch.tensor(normalize_t(t_grid_raw))

    # --- Model and optimizer ---
    model = IVSurfaceNet(hidden_size=30, n_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Reduce learning rate when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5
    )

    # --- Training loop ---
    losses = []
    mse_losses = []

    print(f"Training IV smoother for {n_epochs} epochs...")
    print(f"  lambda_calendar: {lambda_calendar}")
    print(f"  lambda_butterfly: {lambda_butterfly}\n")

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass — predict IVs
        y_pred = model(X)

        # MSE loss — fit market data
        mse_loss = nn.MSELoss()(y_pred, y)

        # Calendar spread penalty
        cal_penalty = torch.tensor(0.0)
        for m in m_grid[::4]:   # sample every 4th moneyness for speed
            m_exp = m.expand(len(t_grid))
            inputs_pen = torch.stack([m_exp, t_grid], dim=1)
            ivs_pen = model(inputs_pen).squeeze()
            total_var = ivs_pen**2 * torch.tensor(t_grid_raw)
            for i in range(len(total_var) - 1):
                violation = torch.relu(total_var[i] - total_var[i+1])
                cal_penalty = cal_penalty + violation**2
        cal_penalty = cal_penalty / (len(m_grid) * len(t_grid))

        # Butterfly penalty
        but_penalty = torch.tensor(0.0)
        for t in t_grid[::3]:   # sample every 3rd maturity for speed
            t_exp = t.expand(len(m_grid))
            inputs_pen = torch.stack([m_grid, t_exp], dim=1)
            ivs_pen = model(inputs_pen).squeeze()
            if len(ivs_pen) >= 3:
                dm = m_grid[1] - m_grid[0]
                d2iv = (ivs_pen[2:] - 2*ivs_pen[1:-1] + ivs_pen[:-2]) / (dm**2)
                but_penalty = but_penalty + torch.relu(-d2iv).pow(2).mean()
        but_penalty = but_penalty / len(t_grid)

        # Total loss
        loss = mse_loss + lambda_calendar * cal_penalty + lambda_butterfly * but_penalty

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        losses.append(loss.item())
        mse_losses.append(mse_loss.item())

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.6f} "
                  f"| MSE: {mse_loss.item():.6f} "
                  f"| Cal: {cal_penalty.item():.6f} "
                  f"| But: {but_penalty.item():.6f}")

    return model, losses, mse_losses, (m_min, m_max, t_min, t_max)


def predict_smooth_surface(model, m_min, m_max, t_min, t_max,
                            n_moneyness=60, n_dte=25):
    """
    Generate smooth IV surface predictions on a regular grid.
    Returns arrays for plotting.
    """
    model.eval()

    m_raw = np.linspace(m_min, m_max, n_moneyness).astype(np.float32)
    t_raw = np.linspace(t_min, t_max, n_dte).astype(np.float32)

    m_norm = 2 * (m_raw - m_min) / (m_max - m_min) - 1
    t_norm = 2 * (t_raw - t_min) / (t_max - t_min) - 1

    M, T = np.meshgrid(m_norm, t_norm)
    inputs = torch.tensor(
        np.stack([M.ravel(), T.ravel()], axis=1).astype(np.float32)
    )

    with torch.no_grad():
        iv_pred = model(inputs).numpy().reshape(n_dte, n_moneyness)

    M_raw, T_raw = np.meshgrid(m_raw, t_raw * 365)
    return M_raw, T_raw, iv_pred


if __name__ == "__main__":
    os.makedirs("data/processed/plots", exist_ok=True)

    # Load data
    df = pd.read_csv("data/processed/iv_surface_raw.csv")
    print(f"Loaded {len(df)} options\n")

    # Train smoother
    model, losses, mse_losses, bounds = train_smoother(
        df, n_epochs=500, lr=1e-3,
        lambda_calendar=10.0, lambda_butterfly=5.0
    )

    # Save model
    torch.save(model.state_dict(), "data/processed/iv_smoother.pt")
    print("\nModel saved to data/processed/iv_smoother.pt")

    # Plot training curve
    plt.figure(figsize=(10, 3))
    plt.plot(losses, label="Total loss", color="royalblue", alpha=0.7)
    plt.plot(mse_losses, label="MSE loss", color="orange", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("IV Smoother Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/processed/plots/smoother_training.png", dpi=150)
    plt.close()

    # Generate smooth surface
    m_min, m_max, t_min, t_max = bounds
    M, T, IV = predict_smooth_surface(model, m_min, m_max, t_min, t_max)

    # Plot smoothed 3D surface
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(M, T, IV, cmap=cm.RdYlGn_r,
                           alpha=0.85, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.1, label="Implied Volatility")
    ax.set_xlabel("Moneyness (K/S)", fontsize=10, labelpad=10)
    ax.set_ylabel("Days to Expiry", fontsize=10, labelpad=10)
    ax.set_zlabel("Implied Volatility", fontsize=10, labelpad=10)
    ax.set_title("SPY IV Surface — Neural Network Smoothed", fontsize=13, pad=20)
    ax.zaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, pos: f"{val:.0%}")
    )
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig("data/processed/plots/smoothed_surface_3d.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Smoothed surface saved to data/processed/plots/smoothed_surface_3d.png")

    # Final MSE
    print(f"\nFinal MSE : {mse_losses[-1]:.6f}")
    print(f"Final RMSE: {np.sqrt(mse_losses[-1]):.4f} ({np.sqrt(mse_losses[-1]):.2%} IV units)")