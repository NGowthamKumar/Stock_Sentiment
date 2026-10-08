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
        "date","ticker","smart_score","S_recency","S_recency_3d","S_events","S_breadth","S_volume","total","pos","neg"
    ]].copy()

    # Shift features by 1 day to avoid leakage (predict t+1 with features at t)
    feats = feats.sort_values(["ticker","date"])
    feats[["smart_score","S_recency","S_recency_3d","S_events","S_breadth","S_volume","total","pos","neg"]] = \
        feats.groupby("ticker")[["smart_score","S_recency","S_recency_3d","S_events","S_breadth","S_volume","total","pos","neg"]].shift(1)

    # ── Sentiment velocity — how fast is SmartScore changing ──
    feats = feats.sort_values(["ticker","date"])
    feats["smartscore_3d_ago"] = feats.groupby("ticker")["smart_score"].shift(3)
    feats["smartscore_velocity_3d"] = (
        feats["smart_score"] - feats["smartscore_3d_ago"]
    ).clip(-30, 30).fillna(0)
    
    # Fill NaN for S_recency_3d — new column, missing in older history rows
    feats["S_recency_3d"] = feats["S_recency_3d"].fillna(feats["S_recency"])
    
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

    # ── PCR (Put/Call Ratio) features ──
    pcr_path = "data/pcr_history.csv"
    if os.path.exists(pcr_path):
        try:
            pcr = pd.read_csv(pcr_path, parse_dates=["date"])
            pcr["date"] = pd.to_datetime(pcr["date"]).dt.tz_localize(None)
            pcr = pcr.sort_values("date")

            # Rolling features
            pcr["pcr_oi_5d_avg"]  = pcr["pcr_oi"].rolling(5,  min_periods=1).mean()
            pcr["pcr_oi_20d_avg"] = pcr["pcr_oi"].rolling(20, min_periods=1).mean()
            pcr["pcr_change"]     = pcr["pcr_oi"].diff()  # rising vs falling

            # Z-score: how extreme is today's PCR vs recent history?
            pcr["pcr_zscore"] = (
                (pcr["pcr_oi"] - pcr["pcr_oi"].rolling(20, min_periods=5).mean()) /
                pcr["pcr_oi"].rolling(20, min_periods=5).std().clip(lower=0.01)
            )

            # Regime signal: -1 bearish, 0 neutral, +1 bullish
            pcr["pcr_regime"] = 0
            pcr.loc[pcr["pcr_oi"] >= 1.2, "pcr_regime"] = -1  # high hedging = bearish
            pcr.loc[pcr["pcr_oi"] <= 0.7, "pcr_regime"] =  1  # low hedging = contrarian bullish

            pcr_cols = ["date","pcr_oi","pcr_vol","pcr_change","pcr_zscore",
                        "pcr_oi_5d_avg","pcr_oi_20d_avg","pcr_regime"]
            pcr_cols = [c for c in pcr_cols if c in pcr.columns]

            df = df.merge(pcr[pcr_cols], on="date", how="left")

            # Forward fill (PCR is daily — same value for all tickers on same day)
            for col in ["pcr_oi","pcr_vol","pcr_change","pcr_zscore",
                        "pcr_oi_5d_avg","pcr_oi_20d_avg","pcr_regime"]:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(
                        1.0 if col in ["pcr_oi","pcr_vol","pcr_oi_5d_avg","pcr_oi_20d_avg"] else 0
                    )

            # Clip extremes
            if "pcr_oi" in df.columns:
                df["pcr_oi"]     = df["pcr_oi"].clip(0.3, 3.0)
                df["pcr_zscore"] = df["pcr_zscore"].clip(-3, 3)
                df["pcr_change"] = df["pcr_change"].clip(-0.5, 0.5)

            print(f"Merged PCR features → pcr_oi range: {df['pcr_oi'].min():.2f}–{df['pcr_oi'].max():.2f}")
        except Exception as e:
            print(f"Warning: PCR merge failed: {e}")
            for col in ["pcr_oi","pcr_vol","pcr_change","pcr_zscore",
                        "pcr_oi_5d_avg","pcr_oi_20d_avg","pcr_regime"]:
                df[col] = 1.0 if "oi" in col or "vol" in col or "avg" in col else 0.0
    else:
        print("PCR history not found — defaulting to neutral (1.0)")
        df["pcr_oi"]       = 1.0
        df["pcr_vol"]      = 1.0
        df["pcr_change"]   = 0.0
        df["pcr_zscore"]   = 0.0
        df["pcr_oi_5d_avg"] = 1.0
        df["pcr_oi_20d_avg"]= 1.0
        df["pcr_regime"]   = 0.0

    # ── DXY + Yield Spread (from fetch_dxy.py) ──
    dxy_path = "data/dxy_history.csv"
    if os.path.exists(dxy_path):
        dxy_df = pd.read_csv(dxy_path, parse_dates=["date"])
        dxy_df["date"] = pd.to_datetime(dxy_df["date"]).dt.tz_localize(None)
        df = df.merge(dxy_df[["date","dxy_change","us_yield_spread"]], on="date", how="left")
        df = df.sort_values(["ticker","date"])
        df["dxy_change"]      = df["dxy_change"].ffill().fillna(0).clip(-3, 3)
        df["us_yield_spread"] = df["us_yield_spread"].ffill().fillna(0).clip(-3, 4)
        print(f"Merged DXY + yield spread → dxy_change range: "
              f"{df['dxy_change'].min():.3f} to {df['dxy_change'].max():.3f}")
    else:
        df["dxy_change"]      = 0.0
        df["us_yield_spread"] = 0.0
        print("DXY history not found — run: python src/fetch_dxy.py")

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
            ("dxy_change",         -3,  3),
            ("us_yield_spread",   -3,   4),   # spread in % points; inversion floor ~-3
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
                    "gold_price","gold_change",
                    "dxy_change","us_yield_spread"]:
            df[col] = 0

    os.makedirs("data/modeling", exist_ok=True)
    out = "data/modeling/dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"Built dataset with {len(df)} rows → {out}")

if __name__ == "__main__":
    main()