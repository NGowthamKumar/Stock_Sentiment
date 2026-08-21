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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score  
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier      
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler    

FEATURES = ["smart_score","S_recency","S_events","S_breadth","S_volume","total","pos","neg","ret_lag1","ret_lag2", "fii_net","dii_net",        
            "vix_change","oil_change","usdinr_change","rsi","macd_diff","bb_pct","bb_width","price_vs_sma"]

TARGET = "ret_fwd_1d"
TARGET_1D = "ret_fwd_1d"
TARGET_3D = "ret_fwd_3d"

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

def evaluate_classifier(model, X, y, folds=5):
    """Separate evaluator for XGBoost classifier — uses accuracy not MAE"""
    if len(X) < folds + 5:
        folds = max(2, min(3, len(X)//3))
    tscv = TimeSeriesSplit(n_splits=folds)
    accs = []
    # Binary label: 1 if return positive, 0 if negative
    y_bin = (y > 0).astype(int)
    for tr, va in tscv.split(X):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr_bin, yva_bin = y_bin.iloc[tr], y_bin.iloc[va]
        model.fit(Xtr, ytr_bin)
        pred_bin = model.predict(Xva)
        accs.append(accuracy_score(yva_bin, pred_bin))
    return dict(
        accuracy=float(np.mean(accs)),
        dir_acc=float(np.mean(accs)),
        mae=0.0,
        r2=0.0,
        spearman=0.0
    )

def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_parquet("data/modeling/dataset.parquet").sort_values(["date","ticker"])
    if df.empty:
        raise SystemExit("dataset is empty. You need at least ~2 days of history.")

    X, y = df[FEATURES], df[TARGET_1D]
    y_bin = (y > 0).astype(int)
    reg_models = {
        "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=4, n_jobs=-1, random_state=42)
    }

    scores = {name: evaluate(m, X, y) for name, m in reg_models.items()}
    for name, s in scores.items():
        print(f"{name}: {s}")

    best_name = min(scores, key=lambda n: (scores[n]["mae"], -scores[n]["dir_acc"]))
    best_model = reg_models[best_name].fit(X, y)

    # ── XGBoost Classifier ──
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
    xgb_scores = evaluate_classifier(xgb, X, y)
    print(f"XGBoost Classifier: {xgb_scores}")
    xgb.fit(X, y_bin)

    joblib.dump(dict(model=xgb, features=FEATURES), "models/xgb_classifier.pkl")
    print(f"Saved XGBoost Classifier → models/xgb_classifier.pkl")

    # ── Voting Ensemble ──
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        n_jobs=-1, random_state=42
    )
    voting = VotingClassifier(
        estimators=[
            ("xgb",  XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   eval_metric="logloss", random_state=42, n_jobs=-1)),
            ("lgbm", lgbm),
            ("rf",   rf_clf)
        ],
        voting="soft"  # uses probabilities — more accurate than hard voting
    )
    
    ensemble_scores = evaluate_classifier(voting, X, y)
    print(f"Voting Ensemble: {ensemble_scores}")
    voting.fit(X, y_bin)
    joblib.dump(dict(model=voting, features=FEATURES), "models/voting_ensemble.pkl")
    print(f"Saved Voting Ensemble → models/voting_ensemble.pkl")

    # ---------------------------------------------------------
    # Define metrics path and run time (needed by 3-day block below)
    # ---------------------------------------------------------
    metrics_path = "data/modeling/model_metrics.csv"
    run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # ── 3-Day XGBoost Classifier ──
    if TARGET_3D in df.columns:
        y_3d = df[TARGET_3D]
        y_bin_3d = (y_3d > 0).astype(int)
        
        # Drop rows where 3d return is NaN (last 3 rows per ticker)
        mask_3d = y_3d.notna()
        X_3d = X[mask_3d]
        y_bin_3d = y_bin_3d[mask_3d]
        
        xgb_3d = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
        xgb_3d_scores = evaluate_classifier(xgb_3d, X_3d, y_3d[mask_3d])
        print(f"XGBoost 3-Day Classifier: {xgb_3d_scores}")
        xgb_3d.fit(X_3d, y_bin_3d)
        joblib.dump(dict(model=xgb_3d, features=FEATURES), "models/xgb_3d_classifier.pkl")
        print(f"Saved XGBoost 3-Day → models/xgb_3d_classifier.pkl")
        
        # ── 3-Day Voting Ensemble ──
        lgbm_3d = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        rf_3d = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=4,
            n_jobs=-1, random_state=42
        )
        voting_3d = VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(n_estimators=300, max_depth=4,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=42, n_jobs=-1)),
                ("lgbm", lgbm_3d),
                ("rf", rf_3d)
            ],
            voting="soft"
        )
        ensemble_3d_scores = evaluate_classifier(voting_3d, X_3d, y_3d[mask_3d])
        print(f"Voting Ensemble 3-Day: {ensemble_3d_scores}")
        voting_3d.fit(X_3d, y_bin_3d)
        joblib.dump(dict(model=voting_3d, features=FEATURES), "models/voting_3d_ensemble.pkl")
        print(f"Saved 3-Day Voting Ensemble → models/voting_3d_ensemble.pkl")
        
        # Save 3-day metrics
        rows_3d = [
            {
                "train_date": run_time, "model": "XGBoost_3Day_Classifier",
                "is_best": False, "mae": 0.0, "r2": 0.0,
                "direction_accuracy": xgb_3d_scores["accuracy"],
                "spearman": 0.0, "rows": len(X_3d)
            },
            {
                "train_date": run_time, "model": "Voting_3Day_Ensemble",
                "is_best": False, "mae": 0.0, "r2": 0.0,
                "direction_accuracy": ensemble_3d_scores["accuracy"],
                "spearman": 0.0, "rows": len(X_3d)
            }
        ]
        pd.DataFrame(rows_3d).to_csv(
            metrics_path, mode='a', header=False, index=False
        )
        print("Saved 3-day model metrics")
    else:
        print("Warning: ret_fwd_3d not in dataset — skipping 3-day models")

    # ---------------------------------------------------------
    # Save all metrics
    # ---------------------------------------------------------
    
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
    rows.append({
        "train_date": run_time,
        "model": "Voting_Ensemble",
        "is_best": False,
        "mae": 0.0, "r2": 0.0,
        "direction_accuracy": ensemble_scores["accuracy"],
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
