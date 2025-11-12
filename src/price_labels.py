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
