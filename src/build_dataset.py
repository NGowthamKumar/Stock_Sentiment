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


def fetch_macro_indicators(start: str, end: str) -> pd.DataFrame:
    """Fetch global macro indicators from yfinance as daily features."""
    import yfinance as yf
    
    macro_tickers = {
        "^INDIAVIX": "india_vix",      # India fear index
        "CL=F":      "crude_oil",      # Crude oil price
        "USDINR=X":  "usd_inr",        # USD/INR exchange rate
    }
    
    frames = []
    for ticker, col_name in macro_tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, 
                           progress=False, auto_adjust=True)
            if df.empty:
                print(f"Warning: no data for {ticker}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Close"]].rename(columns={"Close": col_name})
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df = df.reset_index()
            frames.append(df)
        except Exception as e:
            print(f"Warning: failed to fetch {ticker}: {e}")
    
    if not frames:
        return pd.DataFrame()
    
    # Merge all macro indicators on date
    macro = frames[0]
    for f in frames[1:]:
        macro = macro.merge(f, on="date", how="outer")
    
    # Add daily % change for each indicator
    macro = macro.sort_values("date")
    macro["vix_change"] = macro["india_vix"].pct_change(fill_method=None) * 100
    macro["oil_change"] = macro["crude_oil"].pct_change(fill_method=None) * 100
    macro["usdinr_change"] = macro["usd_inr"].pct_change() * 100
    
    # Forward fill missing values (weekends/holidays)
    macro = macro.ffill()
    
    return macro


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
    start = (first - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end   = (last + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    tickers = sorted(feats["ticker"].dropna().unique().tolist())
    prices = fetch_prices(tickers, start, end)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices = add_forward_return(prices, horizon_days=1)
    prices = prices.sort_values(["ticker","date"])
    prices["ret_lag1"] = prices.groupby("ticker")["ret_fwd"].shift(1)
    prices["ret_lag2"] = prices.groupby("ticker")["ret_fwd"].shift(2)

    # Clip extreme outliers from corporate actions (splits, demergers, bonus issues)
    for col in ["ret_fwd", "ret_lag1", "ret_lag2"]:
        prices[col] = prices[col].clip(lower=-15, upper=15)

    df = feats.merge(prices, on=["date","ticker"], how="inner")

    # FII/DII flow features
    fii_dii_path = "data/fii_dii_history.csv"
    if os.path.exists(fii_dii_path):
        fii_dii = pd.read_csv(fii_dii_path, parse_dates=["date"])
        fii_dii["date"] = pd.to_datetime(fii_dii["date"]).dt.tz_localize(None)
        df = df.merge(fii_dii[["date","fii_net","dii_net"]], on="date", how="left")
        df["fii_net"] = df["fii_net"].fillna(0)
        df["dii_net"] = df["dii_net"].fillna(0)
        print(f"Merged FII/DII features → {df[['fii_net','dii_net']].notna().sum().to_dict()}")
    else:
        df["fii_net"] = 0
        df["dii_net"] = 0
        print("FII/DII history not found, defaulting to 0")

    # Drop rows with missing shifted features or label
    df = df.dropna(subset=["smart_score","ret_fwd","ret_lag1"]).copy()

    # ── Macro indicators ──
    macro = fetch_macro_indicators(start, end)
    if not macro.empty:
        macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
        df = df.merge(macro, on="date", how="left")
        # Forward fill macro for any missing dates
        macro_cols = ["india_vix","crude_oil","usd_inr",
                      "vix_change","oil_change","usdinr_change"]
        df[macro_cols] = df[macro_cols].ffill().fillna(0)
        print(f"Merged macro indicators → {macro_cols}")
    else:
        print("Warning: macro indicators unavailable, defaulting to 0")
        for col in ["india_vix","crude_oil","usd_inr",
                    "vix_change","oil_change","usdinr_change"]:
            df[col] = 0

    os.makedirs("data/modeling", exist_ok=True)
    out = "data/modeling/dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"Built dataset with {len(df)} rows → {out}")

if __name__ == "__main__":
    main()
