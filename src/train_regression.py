"""
Reads:  data/modeling/dataset.parquet
Writes: models/nextday_regressor.pkl
"""
import os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["smart_score","S_recency","S_events","S_breadth","S_volume","total","pos","neg"]
TARGET = "ret_fwd"

def evaluate(model, X, y, folds=5):
    if len(X) < folds + 5:
        folds = max(2, min(3, len(X)//3))
    tscv = TimeSeriesSplit(n_splits=folds)
    maes, r2s, dirs, cors = [], [], [], []
    for tr, va in tscv.split(X):
        Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        maes.append(mean_absolute_error(yva, pred))
        r2s.append(r2_score(yva, pred))
        dirs.append((np.sign(pred) == np.sign(yva)).mean())
        cors.append(pd.Series(pred).corr(yva, method="spearman"))
    return dict(mae=float(np.mean(maes)), r2=float(np.mean(r2s)),
                dir_acc=float(np.mean(dirs)), spearman=float(np.nanmean(cors)))

def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_parquet("data/modeling/dataset.parquet").sort_values(["date","ticker"])
    if df.empty:
        raise SystemExit("❌ dataset is empty. You need at least ~2 days of history.")

    X, y = df[FEATURES], df[TARGET]

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=4, n_jobs=-1, random_state=42)
    }

    scores = {name: evaluate(m, X, y) for name, m in models.items()}
    for name, s in scores.items():
        print(f"{name}: {s}")

    best_name = min(scores, key=lambda n: (scores[n]["mae"], -scores[n]["dir_acc"]))
    best_model = models[best_name].fit(X, y)
    joblib.dump(dict(model=best_model, features=FEATURES), "models/nextday_regressor.pkl")
    print(f"✅ Saved {best_name} → models/nextday_regressor.pkl")
    print(f"🏁 Best scores: {scores[best_name]}")

if __name__ == "__main__":
    main()
