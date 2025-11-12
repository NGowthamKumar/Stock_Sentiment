# src/backfill_history.py
"""
Builds multi-day history of SmartScore snapshots using existing processed news.
Reads: data/processed_sentiment.csv
Writes: data/history/stock_sentiment_summary_history.csv (overwrites)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.event_classifier import add_event_cols

# Same tunables as aggregate_sentiment.py
HALF_LIFE_HOURS = 36
WINDOW_DAYS = 10
WEIGHTS = dict(recency=0.45, events=0.25, breadth=0.20, volume=0.10)
EVENT_WEIGHTS = {
    "EARNINGS": 0.7, "GUIDANCE": 0.6, "MA": 0.6, "LITIGATION": -0.7,
    "REGULATORY": -0.5, "MGMT_CHANGE": 0.3, "ORDER_WIN": 0.5,
    "PRODUCT_LAUNCH": 0.3, "MACRO": 0.2, "OTHER": 0.0,
}

def ewma_weight(hours_ago: float, half_life_h: float) -> float:
    return 0.5 ** (hours_ago / max(half_life_h, 1e-6))

def minmax(series: pd.Series) -> pd.Series:
    if series.empty: return series
    lo, hi = series.min(), series.max()
    if hi == lo: return pd.Series([0.5]*len(series), index=series.index)
    return (series - lo) / (hi - lo)

def compute_snapshot(asof_dt: pd.Timestamp, df: pd.DataFrame) -> pd.DataFrame:
    # filter window up to as-of time
    cutoff = asof_dt - pd.Timedelta(days=WINDOW_DAYS)
    g = df[(df["published_utc"] >= cutoff) & (df["published_utc"] <= asof_dt)]
    if g.empty:
        return pd.DataFrame(columns=["date","ticker","S_recency","S_breadth","S_volume","S_events",
                                     "pos","neg","total","smart_score"])
    hours_ago = (asof_dt - g["published_utc"]).dt.total_seconds() / 3600.0
    g = g.assign(w=hours_ago.map(lambda h: ewma_weight(h, HALF_LIFE_HOURS)))
    g = g.assign(w_total=g["w"])

    rows = []
    for tk, sub in g.groupby("ticker"):
        total = len(sub)
        pos = int((sub["label"] == "positive").sum())
        neg = int((sub["label"] == "negative").sum())

        w = sub["w_total"].values
        s = sub["ensemble"].values
        ewma = float(np.average(s, weights=w)) if w.sum() > 0 else float(np.mean(s))
        S_recency = float(np.clip((ewma + 1) / 2 * 100, 0, 100))
        S_breadth_raw = (pos - neg) / max(1, total)
        S_volume_raw = float(np.log1p(total))
        evw = [EVENT_WEIGHTS.get(e, 0.0) for e in sub["event_type"]]
        ev_mean = float(np.mean(evw)) if evw else 0.0
        S_events = 100.0 * float(np.clip(0.5 + 0.5 * (ev_mean * np.sign(ewma)), 0, 1))
        rows.append(dict(
            date=str(asof_dt.date()), ticker=tk,
            S_recency=S_recency, S_breadth_raw=S_breadth_raw, S_volume_raw=S_volume_raw,
            S_events=S_events, pos=pos, neg=neg, total=int(total)
        ))
    out = pd.DataFrame(rows)
    if out.empty: return out
    out["S_breadth"] = minmax(out["S_breadth_raw"]) * 100
    out["S_volume"]  = minmax(out["S_volume_raw"])  * 100
    out["smart_score"] = (
        WEIGHTS["recency"] * out["S_recency"] +
        WEIGHTS["events"]  * out["S_events"]  +
        WEIGHTS["breadth"] * out["S_breadth"] +
        WEIGHTS["volume"]  * out["S_volume"]
    ).round(2)
    return out[["date","ticker","S_recency","S_breadth","S_volume","S_events",
                "pos","neg","total","smart_score"]]

def main(days: int = 180):
    df = pd.read_csv("data/processed_sentiment.csv")
    if "published_utc" not in df.columns:
        if "published" in df.columns:
            df["published_utc"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        else:
            raise SystemExit("processed_sentiment.csv missing published_utc.")
    df["published_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["ticker","ensemble","published_utc"]).copy()
    df = add_event_cols(df)

    end = pd.Timestamp(datetime.now(timezone.utc).date())
    dates = pd.date_range(end - pd.Timedelta(days=days-1), end, freq="D")
    snaps = []
    for d in dates:
        snap = compute_snapshot(d.tz_localize("UTC"), df)
        if not snap.empty:
            snaps.append(snap)
    if not snaps:
        raise SystemExit("No snapshots computed. Check your processed_sentiment.csv.")
    hist = pd.concat(snaps, ignore_index=True)
    os.makedirs("data/history", exist_ok=True)
    out = "data/history/stock_sentiment_summary_history.csv"
    hist.to_csv(out, index=False)
    print(f"✅ Backfilled {hist['date'].nunique()} days → {out} (rows={len(hist)})")

if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    main(days)
