"""
Reads:  data/stock_sentiment_summary.csv
        models/nextday_regressor.pkl
        models/xgb_classifier.pkl
Writes: data/predictions_nextday.csv
"""
import joblib
import os
import pandas as pd
from src.price_labels import fetch_prices, add_forward_return, add_technical_indicators
import json

def build_features(latest):
    """Shared feature building for both models"""
    features_needed = ["ret_lag1", "ret_lag2", "fii_net", "dii_net"]

    if "ret_lag1" not in latest.columns or "rsi" not in latest.columns:
        try:
            tickers = latest["ticker"].dropna().unique().tolist()
            end   = pd.Timestamp.now().strftime("%Y-%m-%d")
            start = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            
            prices = fetch_prices(tickers, start, end)
            prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
            prices = add_forward_return(prices, horizon_days=1)
            prices = add_technical_indicators(prices)
            prices = prices.sort_values(["ticker","date"])
            
            # Lag features
            prices["ret_lag1"] = prices.groupby("ticker")["ret_fwd"].shift(1)
            prices["ret_lag2"] = prices.groupby("ticker")["ret_fwd"].shift(2)
            
            # Take most recent row per ticker for all features
            today = prices.groupby("ticker").tail(1)[[
                "ticker","ret_lag1","ret_lag2",
                "rsi","macd_diff","bb_pct","bb_width","price_vs_sma"
            ]]
            latest = latest.merge(today, on="ticker", how="left")
            print(f"Fetched price features for {len(today)} tickers")
        except Exception as e:
            print(f"Warning: price feature fetch failed: {e}")
            for col in ["ret_lag1","ret_lag2","rsi","macd_diff","bb_pct","bb_width","price_vs_sma"]:
                if col not in latest.columns:
                    latest[col] = 0

    if "fii_net" not in latest.columns:
        fii_dii_path = os.path.join(os.path.dirname(__file__), "../data/fii_dii_history.csv")
        if os.path.exists(fii_dii_path):
            fii_dii = pd.read_csv(fii_dii_path, parse_dates=["date"])
            today = fii_dii.iloc[-1]
            latest["fii_net"] = min(max(float(today["fii_net"]) / 1000, -10), 10)
            latest["dii_net"] = min(max(float(today["dii_net"]) / 1000, -10), 10)
        else:
            latest["fii_net"] = 0
            latest["dii_net"] = 0
    # Fetch macro indicators if model needs them
    if "india_vix" not in latest.columns:
        try:
            import yfinance as yf
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            start = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            
            macro_map = {
                "^INDIAVIX": "india_vix",
                "BZ=F":      "crude_oil",
                "USDINR=X":  "usd_inr"
            }
            for ticker, col in macro_map.items():
                data = yf.download(ticker, start=start, end=end,
                                  progress=False, auto_adjust=True)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    latest_val = float(data["Close"].iloc[-1].iloc[0] if hasattr(data["Close"].iloc[-1], 'iloc') else data["Close"].iloc[-1])
                    prev_val = float(data["Close"].iloc[-2].iloc[0] if hasattr(data["Close"].iloc[-2], 'iloc') else data["Close"].iloc[-2]) if len(data) > 1 else latest_val
                    latest[col] = latest_val
                    change_col = {"india_vix": "vix_change",
                                  "crude_oil": "oil_change",
                                  "usd_inr":   "usdinr_change"}[col]
                    latest[change_col] = (latest_val - prev_val) / prev_val * 100
                else:
                    latest[col] = 0
                    change_col = {"india_vix": "vix_change",
                                  "crude_oil": "oil_change",
                                  "usd_inr":   "usdinr_change"}[col]
                    latest[change_col] = 0
        except Exception as e:
            print(f"Warning: macro fetch failed: {e}")
            for col in ["india_vix","crude_oil","usd_inr",
                        "vix_change","oil_change","usdinr_change"]:
                if col not in latest.columns:
                    latest[col] = 0
    return latest

def get_signal_label(prob):
    """Convert XGBoost probability to human readable signal"""
    if prob >= 0.72:
        return "🟢 STRONG BULLISH"
    elif prob >= 0.6:
        return "🟡 MILD BULLISH"
    elif prob >= 0.42:
        return "⚪ NEUTRAL"
    elif prob >= 0.30:
        return "🟠 MILD BEARISH"
    else:
        return "🔴 STRONG BEARISH"

def get_confidence(prob):
    """Convert probability to confidence level"""
    distance = abs(prob - 0.5)
    if distance >= 0.15:
        return "High Confidence"
    elif distance >= 0.08:
        return "Medium Confidence"
    else:
        return "Low Confidence"

def save_signal_history(latest: pd.DataFrame, out: pd.DataFrame, 
                        ens_signals: pd.DataFrame) -> None:
    """Save today's predictions to signal history for accuracy tracking"""
    history_path = "data/signal_history.csv"
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    rows = []
    
    for _, row in latest.iterrows():
        ticker = row.get("ticker")
        if not ticker:
            continue
        
        smart_score = float(row.get("smart_score", 0))
        s_recency   = float(row.get("S_recency", 0))
        
        # Signal 1 — SmartScore signal
        # Bullish if SmartScore > 65 AND S_recency > 70
        ss_bullish = 1 if (smart_score > 65 and s_recency > 70) else 0
        
        # Signal 2 — XGBoost signal
        xgb_prob = None
        xgb_bullish = None
        if ens_signals is not None and not ens_signals.empty:
            ens_row = ens_signals[ens_signals["ticker"] == ticker]
            if not ens_row.empty:
                xgb_prob = float(ens_row.iloc[0].get("ensemble_probability", 50))
                xgb_bullish = 1 if xgb_prob > 55 else 0
        
        # Signal 3 — Combined (both agree)
        combined_bullish = None
        if ss_bullish is not None and xgb_bullish is not None:
            combined_bullish = 1 if (ss_bullish == 1 and xgb_bullish == 1) else 0
        
        # Regression predicted return
        reg_row = out[out["ticker"] == ticker]
        pred_ret = float(reg_row.iloc[0]["pred_ret_1d_pct"]) if not reg_row.empty else None
        
        rows.append({
            "pred_date":        today,
            "ticker":           ticker,
            "smart_score":      round(smart_score, 2),
            "s_recency":        round(s_recency, 2),
            "ss_signal":        ss_bullish,       # SmartScore signal
            "xgb_prob":         round(xgb_prob, 2) if xgb_prob else None,
            "xgb_signal":       xgb_bullish,      # XGBoost signal
            "combined_signal":  combined_bullish,  # Both agree
            "pred_ret_pct":     round(pred_ret, 4) if pred_ret else None,
            "actual_ret_pct":   None,              # filled next day
            "actual_dir":       None,              # filled next day
            "ss_correct":       None,              # filled next day
            "xgb_correct":      None,              # filled next day
            "combined_correct": None,              # filled next day
        })
    
    if not rows:
        return
    
    new_df = pd.DataFrame(rows)
    
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        # Avoid duplicating same date + ticker
        existing_keys = set(zip(existing["pred_date"], existing["ticker"]))
        new_rows = new_df[~new_df.apply(
            lambda r: (r["pred_date"], r["ticker"]) in existing_keys, axis=1
        )]
        if not new_rows.empty:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined.to_csv(history_path, index=False)
            print(f"Appended {len(new_rows)} rows to signal history")
    else:
        new_df.to_csv(history_path, index=False)
        print(f"Created signal history with {len(new_df)} rows → {history_path}")


def main():
    latest = pd.read_csv("data/stock_sentiment_summary.csv")
    latest = build_features(latest)

    # ── Existing regression model (unchanged) ──
    bundle = joblib.load("models/nextday_regressor.pkl")
    model = bundle["model"]
    features = bundle["features"]

    X = latest[["ticker", *features]].dropna()
    if X.empty:
        raise SystemExit("No rows with full features in latest snapshot.")

    preds = model.predict(X[features])
    out = X[["ticker"]].copy()
    out["pred_ret_1d_pct"] = preds
    out = out.sort_values("pred_ret_1d_pct", ascending=False)
    out.to_csv("data/predictions_nextday.csv", index=False)
    print("Wrote predictions → data/predictions_nextday.csv")
    print(out.head(10))

    # ── XGBoost Classifier signals ──
    xgb_path = "models/xgb_classifier.pkl"
    if os.path.exists(xgb_path):
        xgb_bundle = joblib.load(xgb_path)
        xgb_model = xgb_bundle["model"]
        xgb_features = xgb_bundle["features"]

        X_xgb = latest[["ticker", *xgb_features]].dropna()
        probs = xgb_model.predict_proba(X_xgb[xgb_features])[:, 1]

        signals = X_xgb[["ticker"]].copy()
        signals["up_probability"] = (probs * 100).round(1)
        signals["signal"] = [get_signal_label(p) for p in probs]
        signals["confidence"] = [get_confidence(p) for p in probs]
        signals = signals.sort_values("up_probability", ascending=False)

        signals.to_csv("data/xgb_signals.csv", index=False)
        print("\nXGBoost Signals:")
        print(signals.head(10))
        print("Wrote XGBoost signals → data/xgb_signals.csv")

    # ── Voting Ensemble signals (new) ──
    ensemble_path = "models/voting_ensemble.pkl"
    if os.path.exists(ensemble_path):
        ens_bundle = joblib.load(ensemble_path)
        ens_model = ens_bundle["model"]
        ens_features = ens_bundle["features"]

        X_ens = latest[["ticker", *ens_features]].dropna()
        ens_probs = ens_model.predict_proba(X_ens[ens_features])[:, 1]

        # Individual model probabilities for transparency
        xgb_probs  = ens_model.estimators_[0].predict_proba(X_ens[ens_features])[:, 1]
        lgbm_probs = ens_model.estimators_[1].predict_proba(X_ens[ens_features])[:, 1]
        rf_probs   = ens_model.estimators_[2].predict_proba(X_ens[ens_features])[:, 1]

        # Count how many models agree
        def models_agree(xgb_p, lgbm_p, rf_p):
            votes_up = sum([xgb_p > 0.5, lgbm_p > 0.5, rf_p > 0.5])
            return votes_up

        ens_signals = X_ens[["ticker"]].copy()
        ens_signals["ensemble_probability"] = (ens_probs * 100).round(1)
        ens_signals["xgb_prob"]   = (xgb_probs  * 100).round(1)
        ens_signals["lgbm_prob"]  = (lgbm_probs * 100).round(1)
        ens_signals["rf_prob"]    = (rf_probs   * 100).round(1)
        ens_signals["models_agree"] = [
            models_agree(x, l, r)
            for x, l, r in zip(xgb_probs, lgbm_probs, rf_probs)
        ]

        def ensemble_signal(row):
            prob = row["ensemble_probability"]
            agree = row["models_agree"]
            if prob >= 68 and agree == 3:
                return "🟢 STRONG BUY — All 3 agree"
            elif prob >= 58 and agree >= 2:
                return "🟡 MILD BUY — 2-3 models agree"
            elif prob <= 32 and agree == 0:
                return "🔴 STRONG AVOID — All 3 agree"
            elif prob <= 42 and agree <= 1:
                return "🟠 MILD AVOID — 2-3 models agree"
            else:
                return "⚪ NEUTRAL — Models uncertain"

        ens_signals["signal"] = ens_signals.apply(ensemble_signal, axis=1)
        ens_signals = ens_signals.sort_values("ensemble_probability", ascending=False)
        ens_signals.to_csv("data/ensemble_signals.csv", index=False)
        print("\nVoting Ensemble Signals:")
        print(ens_signals[["ticker","ensemble_probability","models_agree","signal"]].head(10))
        print("Wrote ensemble signals → data/ensemble_signals.csv")
        # ── Save signal history ──
        try:
            ens_signals = None
            if os.path.exists("data/ensemble_signals.csv"):
                ens_signals = pd.read_csv("data/ensemble_signals.csv")
            save_signal_history(latest, out, ens_signals)
        except Exception as e:
            print(f"Warning: signal history save failed: {e}")

    # ── 3-Day XGBoost signals ──
    xgb_3d_path = "models/xgb_3d_classifier.pkl"
    ens_3d_path = "models/voting_3d_ensemble.pkl"

    if os.path.exists(xgb_3d_path) and os.path.exists(ens_3d_path):
        xgb_3d_bundle = joblib.load(xgb_3d_path)
        ens_3d_bundle  = joblib.load(ens_3d_path)

        X_3d = latest[["ticker", *xgb_3d_bundle["features"]]].dropna()

        # XGBoost 3-day
        probs_3d = xgb_3d_bundle["model"].predict_proba(
            X_3d[xgb_3d_bundle["features"]]
        )[:, 1]

        # Ensemble 3-day
        ens_3d_probs = ens_3d_bundle["model"].predict_proba(
            X_3d[ens_3d_bundle["features"]]
        )[:, 1]

        signals_3d = X_3d[["ticker"]].copy()
        signals_3d["xgb_3d_prob"]      = (probs_3d * 100).round(1)
        signals_3d["ensemble_3d_prob"] = (ens_3d_probs * 100).round(1)
        signals_3d["xgb_3d_signal"]    = signals_3d["xgb_3d_prob"].apply(
            get_signal_label
        )

        # Combined 1d + 3d agreement
        if os.path.exists("data/ensemble_signals.csv"):
            ens_1d = pd.read_csv("data/ensemble_signals.csv")[
                ["ticker","ensemble_probability"]
            ].rename(columns={"ensemble_probability": "ens_1d_prob"})
            signals_3d = signals_3d.merge(ens_1d, on="ticker", how="left")
            signals_3d["both_agree"] = (
                (signals_3d["ens_1d_prob"] > 55) &
                (signals_3d["ensemble_3d_prob"] > 55)
            )
            signals_3d["combined_signal"] = signals_3d.apply(
                lambda r: "🟢 STRONG — Both 1d & 3d agree" if r["both_agree"]
                else "🟡 1d only bullish" if r["ens_1d_prob"] > 55
                else "🔵 3d only bullish" if r["ensemble_3d_prob"] > 55
                else "⚪ NEUTRAL",
                axis=1
            )

        signals_3d = signals_3d.sort_values("ensemble_3d_prob", ascending=False)
        signals_3d.to_csv("data/signals_3d.csv", index=False)
        print("\n3-Day Signals (top 10):")
        print(signals_3d[["ticker","xgb_3d_prob","ensemble_3d_prob",
                           "combined_signal"]].head(10))
        print("Wrote 3-day signals → data/signals_3d.csv")

if __name__ == "__main__":
    main()