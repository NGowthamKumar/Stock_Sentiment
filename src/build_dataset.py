# src/build_dataset.py
"""
Reads:  data/history/stock_sentiment_summary_history.csv
Pulls:  yfinance prices
Writes: data/modeling/dataset.parquet
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from src.price_labels import fetch_prices, add_forward_return

def main():
    hist_path = "data/history/stock_sentiment_summary_history.csv"
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"History not found: {hist_path}. Run aggregate first for several days.")

    h = pd.read_csv(hist_path, parse_dates=["date"])
    feats = h[[
        "date","ticker","smart_score","S_recency","S_events","S_breadth","S_volume","total","pos","neg"
    ]].copy()

    # Shift features by 1 day to avoid leakage (predict t+1 with features at t)
    feats = feats.sort_values(["ticker","date"])
    feats[["smart_score","S_recency","S_events","S_breadth","S_volume","total","pos","neg"]] = \
        feats.groupby("ticker")[["smart_score","S_recency","S_events","S_breadth","S_volume","total","pos","neg"]].shift(1)

    first = feats["date"].min().date()
    last = feats["date"].max().date()
    start = (first - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end   = (last + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    tickers = sorted(feats["ticker"].dropna().unique().tolist())
    prices = fetch_prices(tickers, start, end)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices = add_forward_return(prices, horizon_days=1)

    df = feats.merge(prices, on=["date","ticker"], how="inner")

    # Drop rows with missing shifted features or label
    df = df.dropna(subset=["smart_score","ret_fwd"]).copy()

    os.makedirs("data/modeling", exist_ok=True)
    out = "data/modeling/dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"✅ Built dataset with {len(df)} rows → {out}")

if __name__ == "__main__":
    main()
