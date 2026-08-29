# src/build_dataset.py
"""
Reads:  data/history/stock_sentiment_summary_history.csv
Pulls:  yfinance prices
Writes: data/modeling/dataset.parquet
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from src.price_labels import fetch_prices, add_forward_return, add_technical_indicators


def fetch_macro_indicators(start: str, end: str) -> pd.DataFrame:
    """Fetch global macro indicators from yfinance as daily features."""
    import yfinance as yf
    
    macro_tickers = {
        "^INDIAVIX": "india_vix",      # India fear index
        "BZ=F":      "crude_oil",      # Brent crude oil
        "USDINR=X":  "usd_inr",        # USD/INR exchange rate
        "^VIX":      "us_vix",         # US fear index (captures Nvidia/global tech fear)
        "^NSEI":     "nifty_ret",      # Nifty 50 index (market-wide momentum)
        "^CNXIT":    "nifty_it",       # Nifty IT index (IT sector momentum)
        "^NSEBANK":  "nifty_bank",     # Nifty Bank index (banking sector)
        # "^INBMK":   "bond_yield",      # India 10-year bond yield
        "^TNX":      "us_10y_yield",   # US 10-year Treasury yield, FII signal
        "GC=F":      "gold_price",     # Gold futures, risk-off signal
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
    macro["vix_change"]       = macro["india_vix"].pct_change(fill_method=None) * 100
    macro["oil_change"]       = macro["crude_oil"].pct_change(fill_method=None) * 100
    macro["usdinr_change"]    = macro["usd_inr"].pct_change(fill_method=None) * 100
    if "us_vix" in macro.columns:
        macro["us_vix_change"]    = macro["us_vix"].pct_change(fill_method=None) * 100
    if "nifty_ret" in macro.columns:
        macro["nifty_ret_change"] = macro["nifty_ret"].pct_change(fill_method=None) * 100
    if "nifty_it" in macro.columns:
        macro["nifty_it_change"]  = macro["nifty_it"].pct_change(fill_method=None) * 100
    if "nifty_bank" in macro.columns:
        macro["nifty_bank_change"]= macro["nifty_bank"].pct_change(fill_method=None) * 100
    if "us_10y_yield" in macro.columns:
        macro["us_10y_change"] = macro["us_10y_yield"].pct_change(fill_method=None) * 100
    if "gold_price" in macro.columns:
        macro["gold_change"] = macro["gold_price"].pct_change(fill_method=None) * 100
    
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

    # Actual modeling period
    data_start = pd.Timestamp(first)
    data_end = pd.Timestamp(last)

    # Extra historical warm-up period for technical indicators
    # SMA20 / Bollinger Bands need previous trading-day data.
    price_start = (data_start - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    price_end = (data_end + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    tickers = sorted(feats["ticker"].dropna().unique().tolist())

    print(f"Downloading price history: {price_start} → {price_end}")
    print(f"Modeling period: {data_start.date()} → {data_end.date()}")

    prices = fetch_prices(tickers, price_start, price_end)

    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)

    # 1-day forward return (existing)
    prices = add_forward_return(prices, horizon_days=1)
    prices = prices.rename(columns={"ret_fwd": "ret_fwd_1d"})

    # 3-day forward return (new)
    prices_3d = prices.copy()
    prices_3d["close_next_3d"] = prices_3d.groupby("ticker")["close"].shift(-3)
    prices_3d["ret_fwd_3d"] = (prices_3d["close_next_3d"] / prices_3d["close"] - 1.0) * 100
    prices["ret_fwd_3d"] = prices_3d["ret_fwd_3d"]

    # Use 1-day as primary target (keep ret_fwd for compatibility)
    prices["ret_fwd"] = prices["ret_fwd_1d"]

    # Technical indicators
    prices = add_technical_indicators(prices)
    prices = prices.sort_values(["ticker", "date"])
    prices["ret_lag1"] = prices.groupby("ticker")["ret_fwd_1d"].shift(1)
    prices["ret_lag2"] = prices.groupby("ticker")["ret_fwd_1d"].shift(2)

    # Now keep only the actual modeling period.
    # The warm-up data was used only for calculating indicators/lags.
    prices = prices[
        (prices["date"] >= data_start) &
        (prices["date"] <= data_end)
    ].copy()
    # Clip extreme outliers — including 3-day return
    for col in ["ret_fwd", "ret_fwd_1d", "ret_lag1", "ret_lag2", "ret_fwd_3d"]:
        if col in prices.columns:
            prices[col] = prices[col].clip(lower=-15, upper=15)

    df = feats.merge(prices, on=["date","ticker"], how="inner")

    # FII/DII flow features
    fii_dii_path = "data/fii_dii_history.csv"
    if os.path.exists(fii_dii_path):
        fii_dii = pd.read_csv(fii_dii_path, parse_dates=["date"])
        fii_dii["date"] = pd.to_datetime(fii_dii["date"]).dt.tz_localize(None)
        # Normalize crores - thousands of crores to reduce scale
        fii_dii["fii_net"] = fii_dii["fii_net"] / 1000
        fii_dii["dii_net"] = fii_dii["dii_net"] / 1000
        df = df.merge(fii_dii[["date","fii_net","dii_net"]], on="date", how="left")
        # Forward fill — use last known value instead of 0
        df = df.sort_values(["ticker","date"])
        df["fii_net"] = df.groupby("ticker")["fii_net"].ffill().fillna(0)
        df["dii_net"] = df.groupby("ticker")["dii_net"].ffill().fillna(0)
        # Clip extremes
        df["fii_net"] = df["fii_net"].clip(lower=-10, upper=10)
        df["dii_net"] = df["dii_net"].clip(lower=-10, upper=10)
        print(f"Merged FII/DII features → {df[['fii_net','dii_net']].notna().sum().to_dict()}")
    else:
        df["fii_net"] = 0
        df["dii_net"] = 0
        print("FII/DII history not found, defaulting to 0")

    df = df.dropna(subset=["smart_score","ret_fwd_1d","ret_lag1"]).copy()

    # ── Macro indicators ──
    macro = fetch_macro_indicators(price_start, price_end)
    if not macro.empty:
        macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
        df = df.merge(macro, on="date", how="left")
        macro_cols = ["india_vix","crude_oil","usd_inr",
                      "vix_change","oil_change","usdinr_change"]
        df[macro_cols] = df[macro_cols].ffill().fillna(0)
        
        # Clip extreme macro changes AFTER merging into df
        df["vix_change"]    = df["vix_change"].clip(lower=-15, upper=15)
        df["oil_change"]    = df["oil_change"].clip(lower=-10, upper=10)
        df["usdinr_change"] = df["usdinr_change"].clip(lower=-3,  upper=3)
        df["crude_oil"]     = df["crude_oil"].clip(lower=50, upper=110)
        df["usd_inr"]       = df["usd_inr"].clip(lower=80, upper=95)
        # New sector indices — only clip if they exist
        for col, lo, hi in [
            ("us_vix_change",     -20, 20),
            ("nifty_ret_change",   -5,  5),
            ("nifty_it_change",    -5,  5),
            ("nifty_bank_change",  -5,  5),
            ("bond_yield_change",  -2,  2),
            ("us_10y_change",      -1,  1),   
            ("gold_change",        -5,  5),
        ]:
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)
        
        existing_macro = [c for c in df.columns if any(x in c for x in 
                        ['vix','oil','usd','nifty','us_10y','gold','sp500'])]
        print(f"Merged macro indicators → {existing_macro}")
    else:
        print("Warning: macro indicators unavailable, defaulting to 0")
        for col in ["india_vix","crude_oil","usd_inr","us_vix","nifty_ret",
                    "nifty_it","nifty_bank","bond_yield",
                    "vix_change","oil_change","usdinr_change","us_vix_change",
                    "nifty_ret_change","nifty_it_change","nifty_bank_change",
                    "bond_yield_change","us_10y_yield","us_10y_change",  
                    "gold_price","gold_change"]:
            df[col] = 0

    os.makedirs("data/modeling", exist_ok=True)
    out = "data/modeling/dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"Built dataset with {len(df)} rows → {out}")

if __name__ == "__main__":
    main()
