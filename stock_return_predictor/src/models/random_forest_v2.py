import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Chargement ---
df = pd.read_csv("data/SPY_features_v2.csv", index_col="Date", parse_dates=True)
df.columns = [col.split(",")[0].strip() for col in df.columns]

# --- Features et target ---
features = ["return_1d", "return_5d", "return_10d", "sma_ratio", 
            "volatility_10", "volatility_20", "rsi_14", 
            "volume_ratio", "vix", "vix_change"]
X = df[features]
y = df["target"]

# --- Time-series split (jamais shuffler !) ---
split = int(len(df) * 0.8)   # 80% train, 20% test
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# --- Modèle ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Évaluation ---
y_pred = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))