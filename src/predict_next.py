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
    if "india_vix" not in latest.columns or "us_10y_change" not in latest.columns:
        try:
            import yfinance as yf
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            start = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            
            macro_map = {
                "^INDIAVIX": "india_vix",
                "BZ=F":      "crude_oil",
                "USDINR=X":  "usd_inr",
                "^VIX":      "us_vix",
                "^NSEI":     "nifty_ret",
                "^CNXIT":    "nifty_it",
                "^NSEBANK":  "nifty_bank",
                "^TNX":      "us_10y_yield",   
                "GC=F":      "gold_price", 
            }
            
            change_map = {
                "india_vix":  "vix_change",
                "crude_oil":  "oil_change",
                "usd_inr":    "usdinr_change",
                "us_vix":     "us_vix_change",
                "nifty_ret":  "nifty_ret_change",
                "nifty_it":   "nifty_it_change",
                "nifty_bank": "nifty_bank_change",
                "us_10y_yield": "us_10y_change",    
                "gold_price":   "gold_change",
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
                    change_col = change_map[col]  
                    latest[change_col] = (latest_val - prev_val) / prev_val * 100
                else:
                    latest[col] = 0
                    latest[change_map[col]] = 0   
        except Exception as e:
            print(f"Warning: macro fetch failed: {e}")
            for col in ["india_vix","crude_oil","usd_inr","us_vix",
                        "nifty_ret","nifty_it","nifty_bank",
                        "us_10y_yield","gold_price", 
                        "vix_change","oil_change","usdinr_change",
                        "us_vix_change","nifty_ret_change",
                        "nifty_it_change","nifty_bank_change",
                        "us_10y_change","gold_change"]:
                if col not in latest.columns:
                    latest[col] = 0
    # Weekend fix: fill all change columns with 0 if NaN
    change_cols = ["vix_change","oil_change","usdinr_change","us_vix_change",
                    "nifty_ret_change","nifty_it_change","nifty_bank_change",
                    "us_10y_change","gold_change"]
    for col in change_cols:
        if col in latest.columns:
            latest[col] = latest[col].fillna(0)
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
                        ens_signals: pd.DataFrame,
                        signals_3d: pd.DataFrame = None) -> None:
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
        
        # Signal 4 — 3-day signals
        xgb_3d_prob = None
        xgb_3d_bullish = None
        ens_3d_prob = None
        ens_3d_bullish = None
        if signals_3d is not None and not signals_3d.empty:
            row_3d = signals_3d[signals_3d["ticker"] == ticker]
            if not row_3d.empty:
                xgb_3d_prob    = float(row_3d.iloc[0].get("xgb_3d_prob", 50))
                ens_3d_prob    = float(row_3d.iloc[0].get("ensemble_3d_prob", 50))
                xgb_3d_bullish = 1 if xgb_3d_prob > 55 else 0
                ens_3d_bullish = 1 if ens_3d_prob > 55 else 0
        
        # Regression predicted return
        reg_row = out[out["ticker"] == ticker]
        pred_ret = float(reg_row.iloc[0]["pred_ret_1d_pct"]) if not reg_row.empty else None
        
        rows.append({
            "pred_date":        today,
            "ticker":           ticker,
            "smart_score":      round(smart_score, 2),
            "s_recency":        round(s_recency, 2),
            "ss_signal":        ss_bullish,
            "xgb_prob":         round(xgb_prob, 2) if xgb_prob else None,
            "xgb_signal":       xgb_bullish,
            "combined_signal":  combined_bullish,
            "pred_ret_pct":     round(pred_ret, 4) if pred_ret else None,
            "xgb_3d_prob":      round(xgb_3d_prob, 2) if xgb_3d_prob else None,
            "ens_3d_prob":      round(ens_3d_prob, 2) if ens_3d_prob else None,
            "xgb_3d_signal":    xgb_3d_bullish,
            "ens_3d_signal":    ens_3d_bullish,
            "actual_ret_pct":   None,
            "actual_dir":       None,
            "actual_dir_3d":    None,
            "ss_correct":       None,
            "xgb_correct":      None,
            "combined_correct": None,
            "xgb_3d_correct":   None,
            "ens_3d_correct":   None,
        })
    
    if not rows:
        return
    
    new_df = pd.DataFrame(rows)
    
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        # Avoid duplicating same date + ticker
        if "pred_date" in existing.columns:
            existing_keys = set(zip(existing["pred_date"], existing["ticker"]))
        else:
            existing_keys = set()
        new_rows = new_df[~new_df.apply(
            lambda r: (r["pred_date"], r["ticker"]) in existing_keys, axis=1
        )]
        if not new_rows.empty:
            combined = pd.concat([existing, new_rows.dropna(how='all')], ignore_index=True)
            combined.to_csv(history_path, index=False)
            print(f"Appended {len(new_rows)} rows to signal history")
    else:
        new_df.to_csv(history_path, index=False)
        print(f"Created signal history with {len(new_df)} rows → {history_path}")

def compute_macro_risk(latest: pd.DataFrame) -> dict:
    """
    Compute macro risk score (0-100) from current market conditions
    Higher score = more macro risk = suppress weak bullish signals
    """
    risk_score = 0
    reasons = []

    # US VIX — global fear gauge
    us_vix = latest["us_vix"].median() if "us_vix" in latest.columns else 20
    if us_vix > 25:
        risk_score += 30
        reasons.append(f"US VIX high ({us_vix:.1f})")
    elif us_vix > 20:
        risk_score += 15
        reasons.append(f"US VIX elevated ({us_vix:.1f})")

    # India VIX
    india_vix = latest["india_vix"].median() if "india_vix" in latest.columns else 15
    if india_vix > 20:
        risk_score += 20
        reasons.append(f"India VIX high ({india_vix:.1f})")
    elif india_vix > 15:
        risk_score += 10
        reasons.append(f"India VIX elevated ({india_vix:.1f})")

    # Oil price change — rising oil bad for India
    oil_change = latest["oil_change"].median() if "oil_change" in latest.columns else 0
    if oil_change > 3:
        risk_score += 20
        reasons.append(f"Crude oil surging (+{oil_change:.1f}%)")
    elif oil_change > 1.5:
        risk_score += 10
        reasons.append(f"Crude oil rising (+{oil_change:.1f}%)")

    # US 10yr yield rising = FII selling India
    us_10y = latest["us_10y_change"].median() if "us_10y_change" in latest.columns else 0
    if us_10y > 0.5:
        risk_score += 15
        reasons.append(f"US yield rising (+{us_10y:.2f}%)")

    # FII net selling
    fii_net = latest["fii_net"].median() if "fii_net" in latest.columns else 0
    if fii_net < -5:       # -5000 crores normalized
        risk_score += 25
        reasons.append(f"Heavy FII selling ({fii_net*1000:.0f}cr)")
    elif fii_net < -2:
        risk_score += 12
        reasons.append(f"FII selling ({fii_net*1000:.0f}cr)")

    # S&P 500 falling
    sp500 = latest["sp500_change"].median() if "sp500_change" in latest.columns else 0
    if sp500 < -1.5:
        risk_score += 15
        reasons.append(f"S&P 500 falling ({sp500:.1f}%)")

    return {
        "risk_score": min(risk_score, 100),
        "risk_level": "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 25 else "LOW",
        "reasons": reasons
    }


def apply_macro_filter(signals_df: pd.DataFrame,
                       macro_risk: dict,
                       prob_col: str = "ensemble_probability",
                       signal_col: str = "signal") -> pd.DataFrame:
    """
    Suppress weak bullish signals when macro environment is hostile
    Only suppresses signals below 65% — strong signals remain unchanged
    """
    risk_score = macro_risk["risk_score"]
    reasons = macro_risk["reasons"]

    if risk_score < 25:
        return signals_df  # LOW risk — no suppression

    signals_df = signals_df.copy()

    if risk_score >= 50:
        # HIGH macro risk — suppress all weak bullish signals
        threshold = 65
        warning = f"MACRO RISK ({risk_score}/100)"
        reason_str = " | ".join(reasons[:2])
        mask = signals_df[prob_col] < threshold
        signals_df.loc[mask, signal_col] = f"{warning} — {reason_str}"
        print(f"\n MACRO RISK FILTER ACTIVE (score={risk_score}/100)")
        print(f"   Reasons: {', '.join(reasons)}")
        print(f"   Suppressed {mask.sum()} signals below {threshold}%")

    elif risk_score >= 25:
        # MEDIUM macro risk — suppress only very weak signals
        threshold = 58
        warning = "MACRO CAUTION"
        reason_str = " | ".join(reasons[:1])
        mask = signals_df[prob_col] < threshold
        signals_df.loc[mask, signal_col] = f"{warning} — {reason_str}"
        print(f"\n MACRO CAUTION (score={risk_score}/100)")
        print(f"   Reason: {', '.join(reasons)}")
        print(f"   Suppressed {mask.sum()} signals below {threshold}%")

    return signals_df

def main():
    latest = pd.read_csv("data/stock_sentiment_summary.csv")
    latest = build_features(latest)

    # ── Existing regression model (unchanged) ──
    bundle = joblib.load("models/nextday_regressor.pkl")
    model = bundle["model"]
    features = bundle["features"]

    # Fill NaN features with 0 - handles weekend/holiday missing macro data
    for feat in features:
        if feat not in latest.columns:
            latest[feat] = 0
        else:
            latest[feat] = latest[feat].fillna(0)  #  fill in latest directly

    X = latest[["ticker", *features]].copy()
    X = X.dropna(subset=["ticker"])
    if X.empty:
        raise SystemExit("No rows with full features in latest snapshot.")

    preds = model.predict(X[features])
    out = X[["ticker"]].copy()
    out["pred_ret_1d_pct"] = preds
    out = out.sort_values("pred_ret_1d_pct", ascending=False)
    out.to_csv("data/predictions_nextday.csv", index=False)
    print("Wrote predictions → data/predictions_nextday.csv")
    print(out.head(10))

    # ── Compute macro risk for signal filtering ──
    macro_risk = compute_macro_risk(latest)
    print(f"\nMacro Risk: {macro_risk['risk_level']} ({macro_risk['risk_score']}/100)")
    if macro_risk["reasons"]:
        print(f"  Drivers: {', '.join(macro_risk['reasons'])}")

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
        signals = apply_macro_filter(
            signals, macro_risk,
            prob_col="up_probability",
            signal_col="signal"
        )
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
        ens_signals = apply_macro_filter(
            ens_signals, macro_risk,
            prob_col="ensemble_probability",
            signal_col="signal"
        )
        ens_signals = ens_signals.sort_values("ensemble_probability", ascending=False)
        ens_signals.to_csv("data/ensemble_signals.csv", index=False)
        print("\nVoting Ensemble Signals:")
        print(ens_signals[["ticker","ensemble_probability","models_agree","signal"]].head(10))
        print("Wrote ensemble signals → data/ensemble_signals.csv")

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
        # ── Save signal history (after ALL signals computed) ──
    try:
        ens_loaded = None
        sig3d_loaded = None
        if os.path.exists("data/ensemble_signals.csv"):
            ens_loaded = pd.read_csv("data/ensemble_signals.csv")
        if os.path.exists("data/signals_3d.csv"):
            sig3d_loaded = pd.read_csv("data/signals_3d.csv")
        save_signal_history(latest, out, ens_loaded, sig3d_loaded)
    except Exception as e:
        print(f"Warning: signal history save failed: {e}")

if __name__ == "__main__":
    main()