# src/aggregate_sentiment.py
"""
Reads:  data/processed_sentiment.csv
Writes: data/stock_sentiment_summary.csv (today snapshot)
        data/history/stock_sentiment_summary_history.csv (append with date)

SmartScore v2 = 0.45*Recency + 0.25*Events + 0.20*Breadth + 0.10*Volume
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.event_classifier import add_event_cols

HALF_LIFE_1D = 8    # 8h half-life, for 1-day prediction (fresh news only)
HALF_LIFE_3D = 36   # 36h half-life, for 3-day prediction (trend signal)
WINDOW_DAYS  = 10
WEIGHTS = dict(recency=0.45, events=0.25, breadth=0.20, volume=0.10)

# Event weights in [-1, 1]
EVENT_WEIGHTS = {
    "EARNINGS": 0.7,
    "GUIDANCE": 0.6,
    "MA": 0.6,
    "LITIGATION": -0.7,
    "REGULATORY": -0.5,
    "MGMT_CHANGE": 0.3,
    "ORDER_WIN": 0.5,
    "PRODUCT_LAUNCH": 0.3,
    "MACRO": 0.2,
    "OTHER": 0.0,
}

def ewma_weight(hours_ago: float, half_life_h: float) -> float:
    return 0.5 ** (hours_ago / max(half_life_h, 1e-6))

def minmax(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - lo) / (hi - lo)

# Cross-sectional Z-score normalization (replaced Min-Max)
# Prevents one viral stock from crushing all other stocks' S_volume scores to near zero
def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Z-score across all tickers — immune to outliers"""
    if series.empty or series.std() == 0:
        return pd.Series([50.0] * len(series), index=series.index)
    z = (series - series.mean()) / series.std()
    z = z.clip(-3, 3)              # cap extreme outliers at ±3σ
    return (z + 3) / 6 * 100       # rescale to 0-100

def main():
    now = datetime.now(timezone.utc)
    today = now.date()
    df = pd.read_csv("data/processed_sentiment.csv")

    # ensure published_utc exists and is datetime UTC
    if "published_utc" not in df.columns and "published" in df.columns:
        df["published_utc"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    else:
        df["published_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)

    # ensure ensemble exists (fallback to 'compound' from old runs)
    if "ensemble" not in df.columns and "compound" in df.columns:
        df["ensemble"] = df["compound"]
        
    df = df.dropna(subset=["ticker", "ensemble", "published_utc"]).copy()
    if df.empty:
        print("No rows with ticker+published_utc+ensemble.")
        return

    # classify events
    df = add_event_cols(df)

    # filter by window
    cutoff = now - pd.Timedelta(days=WINDOW_DAYS)
    df = df[df["published_utc"] >= cutoff]
    if df.empty:
        print("No rows in time window.")
        return

    # recency weights
    hours_ago = (now - df["published_utc"]).dt.total_seconds() / 3600.0
    df["w"] = hours_ago.map(lambda h: ewma_weight(h, HALF_LIFE_3D))

    # ── Add tier weighting ──
    # Tier 1 (0-48h) = full weight, Tier 2 (2-30d) = half weight
    def get_tier_weight(published_utc):
        try:
            age_hours = (now - published_utc).total_seconds() / 3600
            if age_hours <= 48:   return 1.0   # LIVE
            elif age_hours <= 720: return 0.5  # RECENT
            else:                  return 0.1  # OLD (shouldn't exist after fetch filter)
        except:
            return 0.5

    df["tier_weight"] = df["published_utc"].apply(get_tier_weight)

    # Use source_weight if available
    if "source_weight" in df.columns:
        df["source_weight"] = df["source_weight"].fillna(0.5)
    else:
        df["source_weight"] = 0.75  # default medium weight
    
    # Use model_confidence if available
    if "model_confidence" in df.columns:
        df["model_confidence"] = df["model_confidence"].fillna(0.5)
    else:
        df["model_confidence"] = 0.5

    # w_total computed per-ticker inside loop using dual half-lives
    # keeping df["w"] for reference only

    # Combined weight: EWMA × tier × source quality × NLP confidence
    '''df["w_total"] = (
        df["w"] * 
        df["tier_weight"] * 
        df["source_weight"] *      #  source quality weight
        df["model_confidence"]     #  NLP confidence weight
    )'''

    out_rows = []
    for tk, g in df.groupby("ticker"):
        total = len(g)
        pos = (g["label"] == "positive").sum()
        neg = (g["label"] == "negative").sum()

        s = g["ensemble"].values
        tier_w = g["tier_weight"].values
        src_w  = g["source_weight"].values
        conf_w = g["model_confidence"].values

        # Dual EWMA — different half-lives for 1d vs 3d
        hours_ago_vals = (now - g["published_utc"]).dt.total_seconds().values / 3600.0
        w_1d = np.array([ewma_weight(h, HALF_LIFE_1D) for h in hours_ago_vals])
        w_3d = np.array([ewma_weight(h, HALF_LIFE_3D) for h in hours_ago_vals])

        # Combined weights with tier/source/confidence
        w_total_1d = w_1d * tier_w * src_w * conf_w
        w_total_3d = w_3d * tier_w * src_w * conf_w

        ewma_1d = float(np.average(s, weights=w_total_1d)) if w_total_1d.sum() > 0 else float(np.mean(s))
        ewma_3d = float(np.average(s, weights=w_total_3d)) if w_total_3d.sum() > 0 else float(np.mean(s))

        # S_recency split by horizon
        S_recency_1d = float(np.clip((ewma_1d + 1) / 2 * 100, 0, 100))
        S_recency_3d = float(np.clip((ewma_3d + 1) / 2 * 100, 0, 100))

        # Keep combined for SmartScore (use 1d for primary signal)
        ewma = ewma_1d
        S_recency = S_recency_1d
        S_breadth_raw = (pos - neg) / max(1, total)
        S_volume_raw = np.log1p(total)

        # Event score: average event weight * sign(ewma) mapped to 0..100
        ev_weights = [EVENT_WEIGHTS.get(e, 0.0) for e in g["event_type"]]
        # Use max magnitude event (not mean) — one major event dominates
        ev_mean = float(max(ev_weights, key=abs)) if ev_weights else 0.0
        S_events = 100.0 * np.clip(0.5 + 0.5 * (ev_mean * np.sign(ewma)), 0, 1)

        out_rows.append(dict(
            date=str(today),
            ticker=tk,
            S_recency=S_recency,          # 8h half-life (1d signal)
            S_recency_3d=S_recency_3d,    # 36h half-life (3d signal)
            S_breadth_raw=S_breadth_raw,
            S_volume_raw=S_volume_raw,
            S_events=S_events,
            pos=int(pos), neg=int(neg), total=int(total),
        ))

    out = pd.DataFrame(out_rows)
    if out.empty:
        print("No aggregates.")
        return

    out["S_breadth"] = cross_sectional_zscore(out["S_breadth_raw"])
    out["S_volume"]  = cross_sectional_zscore(out["S_volume_raw"])

    out["smart_score"] = (
        WEIGHTS["recency"] * out["S_recency"]
        + WEIGHTS["events"]  * out["S_events"]
        + WEIGHTS["breadth"] * out["S_breadth"]
        + WEIGHTS["volume"]  * out["S_volume"]
    ).round(2)

    out = out.sort_values("smart_score", ascending=False)

    # Save today snapshot
    snap_path = "data/stock_sentiment_summary.csv"
    out.to_csv(snap_path, index=False)
    print(f"Summary → {snap_path} ({len(out)} tickers)")

    # Append to history
    hist_dir = "data/history"
    os.makedirs(hist_dir, exist_ok=True)
    hist_path = os.path.join(hist_dir, "stock_sentiment_summary_history.csv")
    if os.path.exists(hist_path):
        prev = pd.read_csv(hist_path)
        combined = pd.concat([prev, out], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date","ticker"], keep="last")
        combined.to_csv(hist_path, index=False)
    else:
        out.to_csv(hist_path, index=False)
    print(f"Appended snapshot → {hist_path}")

if __name__ == "__main__":
    main()
