import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- Load ---
df = pd.read_csv("data/NVDA_features_v1.csv", index_col="Date", parse_dates=True)
df.columns = [col.split(",")[0].strip() for col in df.columns]

# Based on Gu, Kelly & Xiu (2020): momentum features are the strongest predictors
features = ["return_1d", "return_5d", "return_10d", "sma_ratio",
            "volatility_10", "volatility_20", "rsi_14",
            "volume_ratio", "vix", "vix_change", "vix_stressed"]
X = df[features]
y = df["target"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# --- Model ---
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# --- Confidence threshold ---
y_proba = model.predict_proba(X_test)[:, 1]
test_df = df.iloc[split:].copy()
test_df["proba"] = y_proba

print(f"Buy & Hold NVDA : {(1 + test_df['return_1d']).cumprod().iloc[-1]-1:.2%}\n")

for threshold in [0.50, 0.55, 0.60]:
    signal = (test_df["proba"] > threshold).astype(int)
    strategy_returns = signal * test_df["return_1d"]
    cum = (1 + strategy_returns).cumprod()

    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    n_trades = signal.sum()

    print(f"--- Threshold {threshold:.0%} ---")
    print(f"Trades        : {n_trades} / {len(test_df)} days")
    print(f"Total return  : {cum.iloc[-1]-1:.2%}")
    print(f"Sharpe ratio  : {sharpe:.2f}")
    print(f"Max drawdown  : {max_dd:.2%}\n")

# --- Feature importance ---
print("--- Top 5 features ---")
importance = pd.Series(model.feature_importances_, index=features)
print(importance.sort_values(ascending=False).head(5))
