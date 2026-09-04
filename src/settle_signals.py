# src/settle_signals.py
"""
Settles yesterday's predictions with actual market returns.
Reads:  data/signal_history.csv
Writes: data/signal_history.csv (with actual outcomes filled in)
        data/stock_accuracy.csv (per-stock accuracy summary)
"""
import os
import pandas as pd
import yfinance as yf

def fetch_actual_returns(tickers: list, date: str, horizon: int = 1) -> dict:
    """Fetch actual return for a given prediction date and horizon (1 or 3 days)"""
    try:
        start = (pd.Timestamp(date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end   = (pd.Timestamp(date) + pd.Timedelta(days=horizon+5)).strftime("%Y-%m-%d")
        data = yf.download(
            tickers, start=start, end=end,
            progress=False, auto_adjust=True, group_by="ticker"
        )
        
        returns = {}
        if isinstance(data.columns, pd.MultiIndex):
            for tk in tickers:
                try:
                    close = data[tk]["Close"].dropna()
                    if len(close) < 2:
                        continue
                    dates = close.index.strftime("%Y-%m-%d").tolist()
                    if date in dates:
                        idx = dates.index(date)
                        if idx + horizon < len(close):
                            ret = (close.iloc[idx+horizon] / close.iloc[idx] - 1) * 100
                            returns[tk] = {
                                "actual_ret_pct": round(float(ret), 4),
                                "actual_dir":     1 if ret > 0 else 0
                            }
                except:
                    pass
        elif len(tickers) == 1:
            close = data["Close"].dropna()
            dates = close.index.strftime("%Y-%m-%d").tolist()
            if date in dates:
                idx = dates.index(date)
                if idx + horizon < len(close):
                    ret = (close.iloc[idx+horizon] / close.iloc[idx] - 1) * 100
                    returns[tickers[0]] = {
                        "actual_ret_pct": round(float(ret), 4),
                        "actual_dir":     1 if ret > 0 else 0
                    }
        return returns
    except Exception as e:
        print(f"Warning: fetch failed for {date}: {e}")
        return {}

def compute_stock_accuracy(history: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock accuracy for all 3 signal types"""
    settled = history[history["actual_dir"].notna()].copy()
    
    if settled.empty:
        print("No settled predictions yet — check back tomorrow")
        return pd.DataFrame()
    
    results = []
    for ticker, grp in settled.groupby("ticker"):
        grp = grp.sort_values("pred_date")
        n = len(grp)
        if n < 3:
            continue
        
        def acc(col, n_days=None):
            s = grp[col].dropna()
            if n_days:
                s = s.tail(n_days)
            return round(float(s.mean()) * 100, 1) if len(s) > 0 else None
        
        def trust(a):
            if a is None: return "N/A"
            if a >= 60: return "✅ TRUST"
            if a >= 53: return "🟡 MODERATE"
            return "❌ WEAK"
        
        ss_acc   = acc("ss_correct")
        xgb_acc  = acc("xgb_correct")
        com_acc  = acc("combined_correct")
        xgb_3d_acc = acc("xgb_3d_correct")
        ens_3d_acc = acc("ens_3d_correct")

        results.append({
            "ticker":          ticker,
            "total_signals":   n,
            # SmartScore signal
            "ss_acc_overall":  ss_acc,
            "ss_acc_10d":      acc("ss_correct", 10),
            "ss_trust":        trust(ss_acc),
            # XGBoost 1d signal
            "xgb_acc_overall": xgb_acc,
            "xgb_acc_10d":     acc("xgb_correct", 10),
            "xgb_trust":       trust(xgb_acc),
            # Combined signal
            "combined_acc":    com_acc,
            "combined_trust":  trust(com_acc),
            # 3d signal accuracy
            "xgb_3d_acc":      xgb_3d_acc,
            "ens_3d_acc":      ens_3d_acc,
            # Best signal
            "best_signal":     max(
                [("SmartScore", ss_acc or 0),
                 ("XGBoost", xgb_acc or 0),
                 ("Combined", com_acc or 0)],
                key=lambda x: x[1]
            )[0],
        })
    
    df_results = pd.DataFrame(results)
    if df_results.empty:
        return df_results
    # Sort by whatever accuracy column exists
    for sort_col in ["combined_acc", "ss_acc_overall", "xgb_acc_overall"]:
        if sort_col in df_results.columns:
            return df_results.sort_values(sort_col, ascending=False, na_position="last")
    return df_results

def compute_streaks(history: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive positive day streaks and correct prediction streaks per stock"""
    settled = history[history["actual_dir"].notna()].copy()
    if settled.empty:
        return pd.DataFrame()
    
    settled["actual_dir"] = settled["actual_dir"].astype(int)
    settled = settled.sort_values(["ticker", "pred_date"])
    
    results = []
    for ticker, grp in settled.groupby("ticker"):
        grp = grp.sort_values("pred_date")
        
        # Current consecutive positive return streak
        pos_streak = 0
        for v in grp["actual_dir"].values:
            if v == 1:
                pos_streak += 1
            else:
                pos_streak = 0
        
        # Current consecutive correct prediction streak (XGBoost)
        correct_streak = 0
        for v in grp["xgb_correct"].dropna().astype(int).values:
            if v == 1:
                correct_streak += 1
            else:
                correct_streak = 0
        
        # Current consecutive correct prediction streak (SmartScore)
        ss_correct_streak = 0
        for v in grp["ss_correct"].dropna().astype(int).values:
            if v == 1:
                ss_correct_streak += 1
            else:
                ss_correct_streak = 0
        
        # Positive days last 10
        pos_days_10d = int(grp["actual_dir"].tail(10).sum())
        
        # Win rate last 10 days
        xgb_wr = grp["xgb_correct"].tail(10).mean()
        win_rate_10d = round(float(xgb_wr) * 100, 1) if pd.notna(xgb_wr) else None
        
        results.append({
            "ticker":            ticker,
            "pos_day_streak":    pos_streak,
            "xgb_correct_streak": correct_streak,
            "ss_correct_streak": ss_correct_streak,
            "pos_days_10d":      pos_days_10d,
            "win_rate_10d":      win_rate_10d,
            "total_settled":     len(grp),
        })
    
    return pd.DataFrame(results).sort_values(
        "pos_day_streak", ascending=False, na_position="last"
    )


def main():
    history_path  = "data/signal_history.csv"
    accuracy_path = "data/stock_accuracy.csv"
    
    if not os.path.exists(history_path):
        print("No signal history found. Run predict_next.py first.")
        return
    
    history = pd.read_csv(history_path)
    print(f"Loaded {len(history)} signal history rows")
    
    # Find unsettled rows
    unsettled = history[history["actual_dir"].isna()].copy()
    print(f"Unsettled predictions: {len(unsettled)}")
    
    if not unsettled.empty:
        # Only settle past dates (not today)
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        unsettled = unsettled[unsettled["pred_date"] < today]
        
        dates = unsettled["pred_date"].unique()
        print(f"Settling {len(dates)} dates...")
        
        for date in sorted(dates):
            tickers = unsettled[unsettled["pred_date"] == date]["ticker"].unique().tolist()
            print(f"  {date}: {len(tickers)} tickers...")
            
            actuals = fetch_actual_returns(tickers, date)
            if not actuals:
                continue
            
            for idx, row in history[history["pred_date"] == date].iterrows():
                ticker = row["ticker"]
                if ticker not in actuals:
                    continue
                
                actual = actuals[ticker]
                actual_dir = actual["actual_dir"]
                actual_ret = actual["actual_ret_pct"]
                
                history.loc[idx, "actual_ret_pct"] = actual_ret
                history.loc[idx, "actual_dir"]     = actual_dir
                
                # Settle each signal
                if pd.notna(row["ss_signal"]):
                    history.loc[idx, "ss_correct"] = (
                        1 if int(row["ss_signal"]) == actual_dir else 0
                    )
                if pd.notna(row["xgb_signal"]):
                    history.loc[idx, "xgb_correct"] = (
                        1 if int(row["xgb_signal"]) == actual_dir else 0
                    )
                if pd.notna(row["combined_signal"]):
                    history.loc[idx, "combined_correct"] = (
                        1 if int(row["combined_signal"]) == actual_dir else 0
                    )
        
        history.to_csv(history_path, index=False)
        print(f"Updated signal history → {history_path}")
        # ── Settle 3-day signals ──
        if "xgb_3d_signal" in history.columns:
            cutoff_3d = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            unsettled_3d = history[
                history["xgb_3d_correct"].isna() &
                history["xgb_3d_signal"].notna() &
                (history["pred_date"] < cutoff_3d)
            ].copy()

            dates_3d = unsettled_3d["pred_date"].unique()
            if len(dates_3d) > 0:
                print(f"\nSettling 3-day signals for {len(dates_3d)} dates...")
                for date in sorted(dates_3d):
                    tickers_3d = unsettled_3d[
                        unsettled_3d["pred_date"] == date
                    ]["ticker"].unique().tolist()

                    actuals_3d = fetch_actual_returns(tickers_3d, date, horizon=3)
                    if not actuals_3d:
                        continue

                    for idx, row in history[history["pred_date"] == date].iterrows():
                        ticker = row["ticker"]
                        if ticker not in actuals_3d:
                            continue
                        actual_dir_3d = actuals_3d[ticker]["actual_dir"]
                        history.loc[idx, "actual_dir_3d"] = actual_dir_3d

                        if pd.notna(row.get("xgb_3d_signal")):
                            history.loc[idx, "xgb_3d_correct"] = (
                                1 if int(row["xgb_3d_signal"]) == actual_dir_3d else 0
                            )
                        if pd.notna(row.get("ens_3d_signal")):
                            history.loc[idx, "ens_3d_correct"] = (
                                1 if int(row["ens_3d_signal"]) == actual_dir_3d else 0
                            )

                history.to_csv(history_path, index=False)
                print("Updated 3d signal history")

    # Compute accuracy
    acc_df = compute_stock_accuracy(history)
    if not acc_df.empty:
        acc_df.to_csv(accuracy_path, index=False)
        print(f"\nSaved stock accuracy → {accuracy_path}")
        print("\nTop stocks by combined signal accuracy:")
        cols = ["ticker","total_signals","ss_acc_overall","xgb_acc_overall",
                "combined_acc","best_signal"]
        cols = [c for c in cols if c in acc_df.columns]
        print(acc_df[cols].head(15).to_string(index=False))
    else:
        print("Not enough settled data yet — check back tomorrow")

    # Compute and save streaks
    streak_df = compute_streaks(history)
    if not streak_df.empty:
        streak_df.to_csv("data/stock_streaks.csv", index=False)
        print("\nTop stocks by positive day streak:")
        print(streak_df[["ticker","pos_day_streak","xgb_correct_streak",
                          "pos_days_10d","win_rate_10d"]].head(10).to_string(index=False))
    else:
        print("Not enough data for streaks yet")

if __name__ == "__main__":
    main()