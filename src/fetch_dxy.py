"""
Fetches DXY (US Dollar Index) and US yield spread (3m10s) from yfinance.
Writes: data/dxy_history.csv

Run standalone:
    python src/fetch_dxy.py

Called automatically by build_dataset.py before macro merge.

Tickers tried in order (first that returns data wins):
  DXY  → "EURUSD=X" inverted (EUR is 57% of DXY, inverse is a strong proxy)
  3m   → "^IRX" (US 3-month T-bill yield)
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def fetch_dxy(start: str, end: str) -> pd.Series:
    """
    Returns daily DXY % change.
    Strategy: DXY = inverse of EUR/USD (EUR = 57.6% weight in DXY).
    When USD strengthens, EUR/USD falls — so DXY change ≈ -EURUSD % change.
    """
    candidates = [
        ("EURUSD=X",  "eurusd",  -1.0),   # invert: DXY moves opposite to EUR/USD
        ("GBPUSD=X",  "gbpusd",  -0.8),   # GBP = 11.9% of DXY, partial proxy
    ]
    for ticker, name, sign in candidates:
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                continue
            df.index = pd.to_datetime(df.index).tz_localize(None)
            pct = df["Close"].pct_change(fill_method=None) * 100 * sign
            pct.name = "dxy_change"
            print(f"  DXY proxy: {ticker} (sign={sign:+.1f}) — {len(pct.dropna())} rows")
            return pct
        except Exception as e:
            print(f"  Warning: {ticker} failed: {e}")
    print("  Warning: all DXY proxies failed — returning zeros")
    return pd.Series(dtype=float, name="dxy_change")


def fetch_yield_spread(start: str, end: str) -> pd.Series:
    """
    Returns US 3m10s yield spread (10Y - 3M).
    Inverted spread = risk-off = FII EM outflows.
    """
    try:
        df_10y = yf.download("^TNX", start=start, end=end,
                              progress=False, auto_adjust=True)
        if isinstance(df_10y.columns, pd.MultiIndex):
            df_10y.columns = df_10y.columns.get_level_values(0)
        df_10y.index = pd.to_datetime(df_10y.index).tz_localize(None)

        df_3m = yf.download("^IRX", start=start, end=end,
                             progress=False, auto_adjust=True)
        if isinstance(df_3m.columns, pd.MultiIndex):
            df_3m.columns = df_3m.columns.get_level_values(0)
        df_3m.index = pd.to_datetime(df_3m.index).tz_localize(None)

        if not df_10y.empty and not df_3m.empty:
            spread = (df_10y["Close"] - df_3m["Close"]).rename("us_yield_spread")
            print(f"  Yield spread (10Y-3M): {len(spread.dropna())} rows, "
                  f"range {spread.min():.2f}% to {spread.max():.2f}%")
            return spread

        # Fallback: only 10Y available — use 10Y level as spread proxy
        if not df_10y.empty:
            spread = df_10y["Close"].rename("us_yield_spread")
            print(f"  Yield spread fallback (10Y level only): {len(spread.dropna())} rows")
            return spread

    except Exception as e:
        print(f"  Warning: yield spread fetch failed: {e}")

    print("  Warning: yield spread unavailable — returning zeros")
    return pd.Series(dtype=float, name="us_yield_spread")


def main():
    os.makedirs("data", exist_ok=True)
    out_path = "data/dxy_history.csv"

    # Determine fetch window
    end   = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start = "2024-01-01"  # generous lookback for rolling features

    # Extend if existing file already has history
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, parse_dates=["date"])
        if not existing.empty:
            last_date = existing["date"].max()
            start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
            print(f"  Extending existing file from {last_date.date()}")

    print(f"\n── Fetching DXY + Yield Spread ({start} → {end}) ──")

    dxy    = fetch_dxy(start, end)
    spread = fetch_yield_spread(start, end)

    # Combine into one DataFrame
    combined = pd.DataFrame({"dxy_change": dxy, "us_yield_spread": spread})
    combined.index.name = "date"
    combined = combined.reset_index()
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.dropna(subset=["dxy_change"]).sort_values("date")

    # Merge with existing if present
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, parse_dates=["date"])
        combined = pd.concat([existing, combined]).drop_duplicates("date").sort_values("date")

    combined.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(combined)} rows, "
          f"{combined['date'].min().date()} → {combined['date'].max().date()})")
    print(f"  dxy_change range:     {combined['dxy_change'].min():.3f}% "
          f"to {combined['dxy_change'].max():.3f}%")
    if "us_yield_spread" in combined.columns:
        print(f"  us_yield_spread range: {combined['us_yield_spread'].min():.3f}% "
              f"to {combined['us_yield_spread'].max():.3f}%")


if __name__ == "__main__":
    main()