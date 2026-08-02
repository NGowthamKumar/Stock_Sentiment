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
from sklearn.metrics import accuracy_score  
from xgboost import XGBClassifier           

FEATURES = ["smart_score","S_recency","S_events","S_breadth","S_volume","total","pos","neg","ret_lag1","ret_lag2", "fii_net","dii_net"]
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

def evaluate_xgb(model, X, y, folds=5):
    """Separate evaluator for XGBoost classifier — uses accuracy not MAE"""
    if len(X) < folds + 5:
        folds = max(2, min(3, len(X)//3))
    tscv = TimeSeriesSplit(n_splits=folds)
    accs, dirs, probs_list = [], [], []
    # Binary label: 1 if return positive, 0 if negative
    y_bin = (y > 0).astype(int)
    for tr, va in tscv.split(X):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr_bin, yva_bin = y_bin.iloc[tr], y_bin.iloc[va]
        model.fit(Xtr, ytr_bin)
        pred_bin = model.predict(Xva)
        pred_prob = model.predict_proba(Xva)[:, 1]  # probability of UP
        accs.append(accuracy_score(yva_bin, pred_bin))
        dirs.append(float(accuracy_score(yva_bin, pred_bin)))
    return dict(
        accuracy=float(np.mean(accs)),
        dir_acc=float(np.mean(dirs)),
        mae=0.0,
        r2=0.0,
        spearman=0.0
    )

def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_parquet("data/modeling/dataset.parquet").sort_values(["date","ticker"])
    if df.empty:
        raise SystemExit("dataset is empty. You need at least ~2 days of history.")

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
    # ---------------------------------------------------------
    # Save metrics for dashboard / Model Health tab
    # ---------------------------------------------------------
    metrics_path = "data/modeling/model_metrics.csv"
    run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # ── XGBoost Classifier (new, parallel) ──
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_scores = evaluate_xgb(xgb, X, y)
    print(f"XGBoost Classifier: {xgb_scores}")

    # Train final XGBoost on full data
    y_bin = (y > 0).astype(int)
    xgb.fit(X, y_bin)
    joblib.dump(dict(model=xgb, features=FEATURES), "models/xgb_classifier.pkl")
    print(f"Saved XGBoost Classifier → models/xgb_classifier.pkl")

    # ── Save metrics (regression models) ──
    metrics_path = "data/modeling/model_metrics.csv"
    run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")


    rows = []
    for name, s in scores.items():
        rows.append({
            "train_date": run_time,
            "model": name,
            "is_best": name == best_name,
            "mae": s["mae"],
            "r2": s["r2"],
            "direction_accuracy": s["dir_acc"],
            "spearman": s["spearman"],
            "rows": len(df)
        })
    rows.append({
        "train_date": run_time,
        "model": "XGBoost_Classifier",
        "is_best": False,
        "mae": 0.0,
        "r2": 0.0,
        "direction_accuracy": xgb_scores["accuracy"],
        "spearman": 0.0,
        "rows": len(df)
    })

    new_rows = pd.DataFrame(rows)

    if os.path.exists(metrics_path):
        new_rows.to_csv(metrics_path, mode='a', header=False, index=False)
    else:
        new_rows.to_csv(metrics_path, index=False)

    print(f"Saved model metrics → {metrics_path}")

    joblib.dump(dict(model=best_model, features=FEATURES), "models/nextday_regressor.pkl")
    print(f"Saved {best_name} → models/nextday_regressor.pkl")
    print(f"Best scores: {scores[best_name]}")

if __name__ == "__main__":
    main()
