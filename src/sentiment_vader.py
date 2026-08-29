# src/sentiment_vader.py
"""
Reads:  data/raw_news.csv
Writes: data/processed_sentiment.csv

Adds VADER + (optional) FinBERT sentiment, an ensemble score in [-1, 1],
model disagreement/confidence, and a 3-class label.

Design notes:
- FinBERT is optional; if download/import fails, we fall back to VADER-only.
- FinBERT is run in CPU batches for speed in CI (GitHub Actions has no GPU).
- We infer the correct 'pos'/'neg' indices from model.config.id2label.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FINBERT_OK = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from torch.nn.functional import softmax
    FINBERT_OK = True
except Exception:
    FINBERT_OK = False


FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")  # override if needed
MAX_LEN = 400
BATCH_SIZE = 32


def _finbert_pos_minus_neg(texts: list[str]) -> np.ndarray:
    """
    Returns FinBERT sentiment as (P(pos) - P(neg)) in [-1, 1] for each text.
    Uses CPU and processes in batches.
    """
    device = torch.device("cpu")  # Actions runners don't provide GPU
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME).to(device)
    model.eval()

    # work out label indices robustly
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    try:
        pos_idx = next(i for i, l in id2label.items() if "pos" in l)
        neg_idx = next(i for i, l in id2label.items() if "neg" in l)
    except StopIteration:
        # common order for some finbert variants is [neg, neu, pos]
        pos_idx, neg_idx = 2, 0

    out = np.zeros(len(texts), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = [str(t or "")[:MAX_LEN] for t in texts[i : i + BATCH_SIZE]]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            logits = model(**toks).logits  # [B, C]
            probs = softmax(logits, dim=-1).cpu().numpy()
            out[i : i + BATCH_SIZE] = probs[:, pos_idx] - probs[:, neg_idx]

    return out

# Price movement keyword patterns
NEGATIVE_PRICE_PATTERNS = [
    r"fall[s]?\s+\d+", r"drop[s]?\s+\d+", r"decline[s]?\s+\d+",
    r"down\s+\d+\s*%", r"lose[s]?\s+\d+", r"slip[s]?\s+\d+",
    r"plunge[s]?", r"crash[es]?", r"tumble[s]?",
    r"52.week\s+low", r"all.time\s+low", r"multi.year\s+low",
    r"hits?\s+low", r"touches?\s+low",
    r"worst\s+performer", r"top\s+loser",
    r"downgrad", r"cut\s+target", r"sell\s+rating",
    r"penalty", r"fraud", r"scam", r"lawsuit", r"ban",
    r"profit\s+fall", r"revenue\s+drop", r"loss\s+widen",
]

POSITIVE_PRICE_PATTERNS = [
    r"rise[s]?\s+\d+", r"gain[s]?\s+\d+", r"surge[s]?\s+\d+",
    r"up\s+\d+\s*%", r"jump[s]?\s+\d+", r"rally", r"rallies",
    r"52.week\s+high", r"all.time\s+high", r"record\s+high",
    r"hits?\s+high", r"touches?\s+high",
    r"best\s+performer", r"top\s+gainer",
    r"upgrad", r"raise[s]?\s+target", r"buy\s+rating",
    r"beats?\s+estimate", r"profit\s+rise", r"revenue\s+grow",
    r"strong\s+results", r"record\s+profit",
]

import re as _re

def detect_price_movement(title: str) -> float | None:
    """
    Returns -0.6 if headline signals price drop
    Returns +0.6 if headline signals price rise
    Returns None if no clear price movement signal
    """
    t = title.lower()
    for pattern in NEGATIVE_PRICE_PATTERNS:
        if _re.search(pattern, t):
            return -0.6
    for pattern in POSITIVE_PRICE_PATTERNS:
        if _re.search(pattern, t):
            return 0.6
    return None

def main():
    path = "data/raw_news.csv"
    if not os.path.exists(path):
        print("data/raw_news.csv not found. Run src.fetch_news first.")
        return

    df = pd.read_csv(path)
    if df.empty:
        print("data/raw_news.csv is empty.")
        return

    # ----- VADER -----
    vader = SentimentIntensityAnalyzer()
    df["vader"] = df["title"].astype(str).map(lambda x: vader.polarity_scores(x)["compound"]).astype(float)

    # ----- FinBERT (optional) -----
    if FINBERT_OK:
        try:
            fb = _finbert_pos_minus_neg(df["title"].astype(str).tolist())
            df["finbert"] = fb.astype(float)
        except Exception as e:
            print(f"FinBERT failed: {e} — continuing with VADER only.")
            df["finbert"] = np.nan
    else:
        df["finbert"] = np.nan

    # ----- Ensemble -----
    # If FinBERT exists for a row: 0.7*FinBERT + 0.3*VADER; else just VADER
    has_fb = df["finbert"].notna()
    df["ensemble"] = df["vader"].astype(float)
    df.loc[has_fb, "ensemble"] = 0.7 * df.loc[has_fb, "finbert"] + 0.3 * df.loc[has_fb, "vader"]

    # ----- Price movement override -----
    # Headlines with clear price direction override ensemble score
    # "HDFC Bank falls 2% to 52-week low" → force negative regardless of FinBERT
    price_signals = df["title"].astype(str).map(detect_price_movement)
    has_price_signal = price_signals.notna()
    
    # Blend: 60% price signal + 40% ensemble (don't completely ignore NLP)
    df.loc[has_price_signal, "ensemble"] = (
        0.6 * price_signals[has_price_signal] + 
        0.4 * df.loc[has_price_signal, "ensemble"]
    )
    
    overridden = has_price_signal.sum()
    print(f"Price movement override applied to {overridden} headlines")

    # Diagnostics
    diff = (df["finbert"].fillna(df["vader"]) - df["vader"]).abs().clip(0, 1)
    df["model_disagreement"] = diff
    df["model_confidence"] = (1.0 - diff).clip(0.0, 1.0)

    # Discrete label
    df["label"] = pd.cut(
        df["ensemble"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    )

    # Persist
    out = "data/processed_sentiment.csv"
    df.to_csv(out, index=False)
    print(f"Sentiment done for {len(df)} rows → {out}")
    print(df[["vader", "finbert", "ensemble", "model_confidence", "label"]].head(8))


if __name__ == "__main__":
    main()
