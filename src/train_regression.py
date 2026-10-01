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
from sklearn.dummy import DummyClassifier
from scipy import stats

FEATURES = ["smart_score","S_recency","S_recency_3d","S_events","S_breadth","S_volume","total","pos","neg","ret_lag1","ret_lag2", "fii_net","dii_net",        
            "vix_change","oil_change","usdinr_change","rsi","macd_diff","bb_pct","bb_width","price_vs_sma",
            "us_vix_change",      # US fear → IT sector pressure
            "nifty_ret_change",   # Market-wide momentum
            "nifty_it_change",    # IT sector momentum
            "nifty_bank_change",  # Banking sector momentum
            #"bond_yield_change",  # Interest rate sensitivity
            "us_10y_change",      # US 10yr yield, FII flow predictor
            "gold_change",        # Gold price
            ]
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

def evaluate_baselines(X, y, folds=5):
    """
    Evaluate naive baseline models to compare against ML models.
    Critical for proving our models add genuine predictive value.
    """
    y_bin = (y > 0).astype(int)
    tscv = TimeSeriesSplit(n_splits=folds)

    baselines = {
        "Always_UP":        DummyClassifier(strategy="constant", constant=1),
        "Always_DOWN":      DummyClassifier(strategy="constant", constant=0),
        "Most_Frequent":    DummyClassifier(strategy="most_frequent"),
        "Prior_Day_Direction": None,  # handled separately below
    }

    results = {}
    for name, clf in baselines.items():
        if clf is None:
            continue
        accs = []
        for tr, va in tscv.split(X):
            Xtr = X.iloc[tr]; Xva = X.iloc[va]
            ytr = y_bin.iloc[tr]; yva = y_bin.iloc[va]
            clf.fit(Xtr, ytr)
            accs.append(accuracy_score(yva, clf.predict(Xva)))
        results[name] = float(np.mean(accs))

    # Prior day direction baseline
    if "ret_lag1" in X.columns:
        accs = []
        for tr, va in tscv.split(X):
            Xva = X.iloc[va]
            yva = y_bin.iloc[va]
            prior_dir = (Xva["ret_lag1"] > 0).astype(int)
            accs.append(accuracy_score(yva, prior_dir))
        results["Prior_Day_Direction"] = float(np.mean(accs))

    return results


def compute_statistical_significance(dir_acc, n_samples, baseline=0.50):
    """
    Compute p-value and confidence interval for direction accuracy.
    Tests whether accuracy is significantly better than baseline (default 50%).
    """
    n_correct = int(dir_acc * n_samples)

    # One-sided binomial test: is accuracy > baseline?
    p_value = stats.binomtest(n_correct, n_samples, baseline,
                              alternative="greater").pvalue

    # 95% confidence interval (Wilson method — better for proportions)
    z = 1.96
    p = dir_acc
    n = n_samples
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denominator
    margin  = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
    ci_low  = round(centre - margin, 4)
    ci_high = round(centre + margin, 4)

    return {
        "p_value":    round(float(p_value), 6),
        "ci_low":     ci_low,
        "ci_high":    ci_high,
        "significant": p_value < 0.05,
        "n_samples":  n_samples,
        "n_correct":  n_correct,
    }

def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_parquet("data/modeling/dataset.parquet").sort_values(["date","ticker"])
    if df.empty:
        raise SystemExit("dataset is empty. You need at least ~2 days of history.")
    
    # Fill any NaN in features (e.g. new columns not in historical data)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    # Fill any NaN in features — handles new columns missing from historical data
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = df["S_recency"] if "recency" in feat else 0
        df[feat] = df[feat].fillna(df[feat].median())
        
    X, y = df[FEATURES], df[TARGET_1D]
    y_bin = (y > 0).astype(int)
    n_samples = len(y)

    # ── Baseline models ──
    print("\n── Baseline Models ──")
    baseline_scores = evaluate_baselines(X, y)
    for name, acc in baseline_scores.items():
        print(f"  {name}: {acc:.4f} ({acc*100:.2f}%)")
    reg_models = {
        "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=4, n_jobs=-1, random_state=42, 
        )
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
        n_jobs=-1,
        scale_pos_weight=1.0,
    )
    xgb_scores = evaluate_classifier(xgb, X, y)
    print(f"XGBoost Classifier: {xgb_scores}")
    
    # Compute class balance for XGBoost
    neg_count = (y_bin == 0).sum()
    pos_count = (y_bin == 1).sum()
    scale_pw = neg_count / max(pos_count, 1)
    xgb_balanced = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
        scale_pos_weight=scale_pw
    )
    xgb_balanced.fit(X, y_bin)
    joblib.dump(dict(model=xgb_balanced, features=FEATURES), "models/xgb_classifier.pkl")
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
        n_jobs=-1, random_state=42, class_weight="balanced" 
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

    # ── Statistical significance ──
    print("\n── Statistical Significance ──")
    sig_xgb = compute_statistical_significance(xgb_scores["accuracy"], n_samples)
    print(f"XGBoost vs 50% baseline:")
    print(f"  Accuracy: {xgb_scores['accuracy']*100:.2f}%")
    print(f"  p-value:  {sig_xgb['p_value']:.6f} {'✅ SIGNIFICANT' if sig_xgb['significant'] else '❌ NOT SIGNIFICANT'}")
    print(f"  95% CI:   [{sig_xgb['ci_low']*100:.2f}%, {sig_xgb['ci_high']*100:.2f}%]")
    print(f"  Edge vs Always-UP: {(xgb_scores['accuracy'] - baseline_scores.get('Always_UP', 0.5))*100:+.2f}%")

    ensemble_scores = evaluate_classifier(voting, X, y)
    print(f"Voting Ensemble: {ensemble_scores}")
    sig_ens = compute_statistical_significance(ensemble_scores["accuracy"], n_samples)
    print(f"Ensemble vs 50% baseline:")
    print(f"  Accuracy: {ensemble_scores['accuracy']*100:.2f}%")
    print(f"  p-value:  {sig_ens['p_value']:.6f} {'✅ SIGNIFICANT' if sig_ens['significant'] else '❌ NOT SIGNIFICANT'}")
    print(f"  95% CI:   [{sig_ens['ci_low']*100:.2f}%, {sig_ens['ci_high']*100:.2f}%]")
    print(f"  Edge vs Always-UP: {(ensemble_scores['accuracy'] - baseline_scores.get('Always_UP', 0.5))*100:+.2f}%")
    voting.fit(X, y_bin)
    
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
            n_jobs=-1, random_state=42, class_weight="balanced"
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

    # Save baseline scores
    for bname, bacc in baseline_scores.items():
        rows.append({
            "train_date": run_time,
            "model": f"Baseline_{bname}",
            "is_best": False,
            "mae": 0.0, "r2": 0.0,
            "direction_accuracy": bacc,
            "spearman": 0.0,
            "rows": len(df)
        })

    # Save statistical significance
    for model_name, sig in [
        ("XGBoost_Classifier", sig_xgb),
        ("Voting_Ensemble",    sig_ens),
    ]:
        rows.append({
            "train_date": run_time,
            "model": f"Significance_{model_name}",
            "is_best": False,
            "mae": sig["p_value"],
            "r2": sig["ci_low"],
            "direction_accuracy": sig["ci_high"],
            "spearman": 1.0 if sig["significant"] else 0.0,
            "rows": sig["n_samples"]
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
