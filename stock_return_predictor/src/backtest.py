import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Chargement ---
df = pd.read_csv("data/SPY_features.csv", index_col="Date", parse_dates=True)
df.columns = [col.split(",")[0].strip() for col in df.columns]

# --- Reproduire le split et les prédictions ---
from sklearn.ensemble import RandomForestClassifier

features = ["return_1d", "return_5d", "sma_ratio", "volatility_10", "rsi_14"]
X = df[features]
y = df["target"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Backtest sur la période test uniquement ---
test_df = df.iloc[split:].copy()
test_df["prediction"] = model.predict(X_test)

# Stratégie : long si prédit 1, flat si prédit 0
test_df["strategy_return"] = test_df["prediction"] * test_df["return_1d"]
test_df["buyhold_return"] = test_df["return_1d"]

# --- Cumulative returns ---
test_df["strategy_cum"] = (1 + test_df["strategy_return"]).cumprod()
test_df["buyhold_cum"] = (1 + test_df["buyhold_return"]).cumprod()

# --- Métriques ---
def sharpe(returns, freq=252):
    return np.sqrt(freq) * returns.mean() / returns.std()

def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

print("=== Stratégie ML ===")
print(f"Return total    : {test_df['strategy_cum'].iloc[-1]-1:.2%}")
print(f"Sharpe ratio    : {sharpe(test_df['strategy_return']):.2f}")
print(f"Max drawdown    : {max_drawdown(test_df['strategy_cum']):.2%}")

print("\n=== Buy & Hold ===")
print(f"Return total    : {test_df['buyhold_cum'].iloc[-1]-1:.2%}")
print(f"Sharpe ratio    : {sharpe(test_df['buyhold_return']):.2f}")
print(f"Max drawdown    : {max_drawdown(test_df['buyhold_cum']):.2%}")

# --- Plot ---
plt.figure(figsize=(12, 5))
plt.plot(test_df.index, test_df["strategy_cum"], label="Stratégie ML", color="royalblue")
plt.plot(test_df.index, test_df["buyhold_cum"], label="Buy & Hold", color="orange")
plt.title("Stratégie ML vs Buy & Hold — SPY")
plt.ylabel("Valeur du portefeuille (base 1)")
plt.legend()
plt.tight_layout()
plt.savefig("data/backtest.png")
print("\nGraphique sauvegardé dans data/backtest.png")