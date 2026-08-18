# src/price_labels.py
import pandas as pd
import yfinance as yf

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily Adjusted Close; returns long df: date, ticker, close."""
    if not tickers:
        return pd.DataFrame(columns=["date","ticker","close"])
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, group_by="ticker")
    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for tk in tickers:
            try:
                sub = data[tk]["Close"].dropna().rename("close").reset_index()
                sub["ticker"] = tk
                sub.rename(columns={"Date":"date"}, inplace=True)
                rows.append(sub[["date","ticker","close"]])
            except KeyError:
                pass
    else:
        sub = data["Close"].dropna().rename("close").reset_index()
        sub["ticker"] = tickers[0]
        sub.rename(columns={"Date":"date"}, inplace=True)
        rows.append(sub[["date","ticker","close"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date","ticker","close"])

def add_forward_return(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Compute forward return (t→t+1) per ticker."""
    d = df.sort_values(["ticker","date"]).copy()
    d["close_next"] = d.groupby("ticker")["close"].shift(-horizon_days)
    d["ret_fwd"] = (d["close_next"] / d["close"] - 1.0) * 100.0  # in %
    return d.drop(columns=["close_next"])

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, MACD, and Bollinger Band technical indicators per ticker.
    Requires 'close' column. Adds:
        rsi          — Relative Strength Index (14-day)
        macd         — MACD line
        macd_signal  — MACD signal line
        macd_diff    — MACD histogram (macd - signal)
        bb_upper     — Bollinger Band upper
        bb_lower     — Bollinger Band lower
        bb_pct       — Where price sits within bands (0=lower, 1=upper)
        bb_width     — Band width (volatility measure)
        volume_sma20 — 20-day SMA of close (momentum proxy)
    """
    result = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        close = grp["close"]

        # ── RSI (14-day) ──
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, float("nan"))
        grp["rsi"] = 100 - (100 / (1 + rs))

        # ── MACD (12, 26, 9) ──
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        grp["macd"]        = ema12 - ema26
        grp["macd_signal"] = grp["macd"].ewm(span=9, adjust=False).mean()
        grp["macd_diff"]   = grp["macd"] - grp["macd_signal"]
        # Normalize MACD by price level (comparable across stocks)
        grp["macd_diff"]   = grp["macd_diff"] / close.rolling(20).mean() * 100

        # ── Bollinger Bands (20-day, 2 std) ──
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std()
        grp["bb_upper"] = sma20 + 2 * std20
        grp["bb_lower"] = sma20 - 2 * std20
        band_width      = grp["bb_upper"] - grp["bb_lower"]
        grp["bb_pct"]   = (close - grp["bb_lower"]) / band_width.replace(0, float("nan"))
        grp["bb_width"] = band_width / sma20  # normalized width

        # ── Price momentum proxy ──
        grp["close_sma20"] = sma20
        grp["price_vs_sma"] = (close - sma20) / sma20 * 100  # % above/below 20d SMA

        # ── Clip extremes AFTER all computations ──
        grp["macd_diff"]    = grp["macd_diff"].clip(-5, 5)
        grp["bb_pct"]       = grp["bb_pct"].clip(-0.5, 1.5)
        grp["price_vs_sma"] = grp["price_vs_sma"].clip(-20, 20)
        
        result.append(grp)

    return pd.concat(result, ignore_index=True)