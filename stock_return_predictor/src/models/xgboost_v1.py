import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/SPY_features_v2.csv", index_col="Date", parse_dates=True)
df.columns = [col.split(",")[0].strip() for col in df.columns]

features = ["return_1d", "return_5d", "return_10d", "sma_ratio",
            "volatility_10", "volatility_20", "rsi_14",
            "volume_ratio", "vix", "vix_change"]
X = df[features]
y = df["target"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = XGBClassifier(n_estimators=200, learning_rate=0.05, 
                      max_depth=4, random_state=42, eval_metric="logloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))