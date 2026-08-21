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

def fetch_actual_returns(tickers: list, date: str) -> dict:
    """Fetch actual next-day return for a given prediction date"""
    try:
        start = (pd.Timestamp(date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end   = (pd.Timestamp(date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        
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
                        if idx + 1 < len(close):
                            ret = (close.iloc[idx+1] / close.iloc[idx] - 1) * 100
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
                if idx + 1 < len(close):
                    ret = (close.iloc[idx+1] / close.iloc[idx] - 1) * 100
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
        
        ss_acc  = acc("ss_correct")
        xgb_acc = acc("xgb_correct")
        com_acc = acc("combined_correct")
        
        results.append({
            "ticker":          ticker,
            "total_signals":   n,
            # SmartScore signal
            "ss_acc_overall":  ss_acc,
            "ss_acc_10d":      acc("ss_correct", 10),
            "ss_trust":        trust(ss_acc),
            # XGBoost signal
            "xgb_acc_overall": xgb_acc,
            "xgb_acc_10d":     acc("xgb_correct", 10),
            "xgb_trust":       trust(xgb_acc),
            # Combined signal
            "combined_acc":    com_acc,
            "combined_trust":  trust(com_acc),
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

if __name__ == "__main__":
    main()