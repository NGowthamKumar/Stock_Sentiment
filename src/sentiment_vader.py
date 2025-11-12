# src/sentiment_vader.py
"""
Reads:  data/raw_news.csv
Writes: data/processed_sentiment.csv

Adds VADER + optional FinBERT sentiment, an ensemble score in [-1,1],
model disagreement/confidence, and a 3-class label.
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Try FinBERT; falls back to VADER-only if unavailable
FINBERT_OK = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_OK = True
except Exception:
    FINBERT_OK = False


def finbert_scores(texts: list[str]) -> np.ndarray:
    """Return FinBERT sentiment as pos-neg in [-1,1]."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    outs = []
    with torch.no_grad():
        for t in texts:
            tok = tokenizer((t or "")[:400], return_tensors="pt", truncation=True)
            logits = model(**tok).logits[0].numpy()
            # order: [neg, neu, pos]
            probs = np.exp(logits) / np.exp(logits).sum()
            outs.append(float(probs[2] - probs[0]))
    return np.array(outs, dtype=float)


def main():
    df = pd.read_csv("data/raw_news.csv")
    if df.empty:
        print("❌ data/raw_news.csv is empty.")
        return

    # VADER
    v = SentimentIntensityAnalyzer()
    df["vader"] = df["title"].astype(str).map(lambda x: v.polarity_scores(x)["compound"])

    # FinBERT (optional)
    if FINBERT_OK:
        try:
            df["finbert"] = finbert_scores(df["title"].astype(str).tolist())
        except Exception as e:
            print(f"⚠️ FinBERT failed ({e}); using VADER only.")
            df["finbert"] = np.nan
    else:
        df["finbert"] = np.nan

    # Ensemble (FinBERT 0.7, VADER 0.3 if available)
    use_fb = df["finbert"].notna().any()
    w_f, w_v = (0.7, 0.3) if use_fb else (0.0, 1.0)
    df["ensemble"] = np.where(
        df["finbert"].notna(),
        w_f * df["finbert"] + w_v * df["vader"],
        df["vader"],
    )

    # Diagnostics
    df["model_disagreement"] = (df["finbert"] - df["vader"]).abs().fillna(0.0)
    df["model_confidence"] = 1.0 - df["model_disagreement"].clip(0, 1)

    # Labels
    df["label"] = pd.cut(
        df["ensemble"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    )

    df.to_csv("data/processed_sentiment.csv", index=False)
    print(f"✅ Sentiment done for {len(df)} rows → data/processed_sentiment.csv")
    print(df[["vader", "finbert", "ensemble", "model_confidence", "label"]].head(8))


if __name__ == "__main__":
    main()
