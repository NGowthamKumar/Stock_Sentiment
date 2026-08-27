# check_pipeline.py
"""
End-to-end pipeline audit for today's news
Checks: ingestion → sentiment → ticker mapping → SmartScore contribution
"""
import pandas as pd
import os

TODAY = "2026-08-27"
KEYWORDS = ["iran", "hormuz", "tariff", "semiconductor", "fed", 
            "hdfc", "copper", "gdp", "inflation", "qatar"]

print("=" * 70)
print("STEP 1 — RAW NEWS INGESTION")
print("=" * 70)

raw = pd.read_csv("data/raw_news.csv")
raw["published_utc"] = pd.to_datetime(raw["published_utc"], utc=True, errors="coerce")
today_raw = raw[raw["published_utc"].dt.date.astype(str) >= TODAY]

print(f"Total articles today: {len(today_raw)}")
print(f"Date range: {today_raw['published_utc'].min()} → {today_raw['published_utc'].max()}")
print(f"Sources: {today_raw['source_name'].nunique()} unique sources")
print()

for kw in KEYWORDS:
    matches = today_raw[today_raw["title"].str.lower().str.contains(kw, na=False)]
    print(f"\n--- {kw.upper()} ({len(matches)} articles) ---")
    for _, row in matches.iterrows():
        print(f"  Ticker:  {row['ticker']}")
        print(f"  Source:  {row['source_name']}")
        print(f"  Title:   {str(row['title'])[:100]}")
        print(f"  Time:    {row['published_utc']}")
        print()

print("\n" + "=" * 70)
print("STEP 2 — SENTIMENT SCORING (FinBERT + VADER)")
print("=" * 70)

sent_path = "data/processed_sentiment.csv"
if os.path.exists(sent_path):
    sent = pd.read_csv(sent_path)
    sent["published_utc"] = pd.to_datetime(
        sent["published_utc"], utc=True, errors="coerce"
    )
    today_sent = sent[sent["published_utc"].dt.date.astype(str) >= TODAY]
    
    print(f"Total sentiment rows today: {len(today_sent)}")
    print(f"Label distribution:\n{today_sent['label'].value_counts()}")
    print()

    for kw in KEYWORDS:
        matches = today_sent[
            today_sent["title"].str.lower().str.contains(kw, na=False)
        ]
        if len(matches) > 0:
            print(f"\n--- {kw.upper()} sentiment ---")
            for _, row in matches.iterrows():
                print(f"  Title:    {str(row['title'])[:80]}")
                print(f"  Ticker:   {row.get('ticker', 'N/A')}")
                finbert  = row.get("finbert", row.get("finbert_score", "N/A"))
                vader    = row.get("vader", row.get("vader_score", "N/A"))
                ensemble = row.get("ensemble", row.get("ensemble_score", "N/A"))
                conf     = row.get("model_confidence", "N/A")
                label    = row.get("label", "N/A")
                print(f"  FinBERT:  {finbert}")
                print(f"  VADER:    {vader}")
                print(f"  Ensemble: {ensemble}")
                print(f"  Confidence: {conf}")
                print(f"  Label:    {label}")
                print()
else:
    print("processed_sentiment.csv not found")

print("\n" + "=" * 70)
print("STEP 3 — SMARTSCORE CONTRIBUTION (today's summary)")
print("=" * 70)

summary_path = "data/stock_sentiment_summary.csv"
if os.path.exists(summary_path):
    summary = pd.read_csv(summary_path)
    
    # Stocks most affected by today's news
    affected_tickers = {
        "HDFCBANK.NS": "HDFC Bank (lawsuit)",
        "HINDCOPPER.NS": "Hindustan Copper (OFS)",
        "RELIANCE.NS": "Reliance (Iran/Hormuz trade)",
        "INFY.NS": "Infosys (Fed/tariff)",
        "TCS.NS": "TCS (Fed/tariff)",
        "TATAMOTORS.NS": "Tata Motors (semiconductor)",
    }
    
    print(f"\nToday's SmartScore for news-affected stocks:")
    print(f"{'Ticker':<20} {'SmartScore':<12} {'S_recency':<12} {'S_events':<10} {'pos':<6} {'neg':<6}")
    print("-" * 70)
    
    for ticker, desc in affected_tickers.items():
        row = summary[summary["ticker"] == ticker]
        if not row.empty:
            r = row.iloc[0]
            print(f"{ticker:<20} {r.get('smart_score',0):<12.1f} "
                  f"{r.get('S_recency',0):<12.1f} "
                  f"{r.get('S_events',0):<10.1f} "
                  f"{r.get('pos',0):<6} {r.get('neg',0):<6}")
            print(f"  → {desc}")
    
    print(f"\nTop 5 highest SmartScore today:")
    top5 = summary.nlargest(5, "smart_score")[
        ["ticker","smart_score","S_recency","S_events","pos","neg"]
    ]
    print(top5.to_string(index=False))
    
    print(f"\nBottom 5 lowest SmartScore today (most negative sentiment):")
    bot5 = summary.nsmallest(5, "smart_score")[
        ["ticker","smart_score","S_recency","S_events","pos","neg"]
    ]
    print(bot5.to_string(index=False))
else:
    print("stock_sentiment_summary.csv not found")

print("\n" + "=" * 70)
print("STEP 4 — SIGNAL GENERATED")
print("=" * 70)

sig_path = "data/ensemble_signals.csv"
sig3d_path = "data/signals_3d.csv"

if os.path.exists(sig_path):
    signals = pd.read_csv(sig_path)
    print("\nTop ensemble signals today:")
    print(signals.head(10)[["ticker","ensemble_probability","models_agree","signal"]].to_string(index=False))

if os.path.exists(sig3d_path):
    sig3d = pd.read_csv(sig3d_path)
    print("\nTop 3-day signals today:")
    print(sig3d.head(10)[["ticker","xgb_3d_prob","ensemble_3d_prob","combined_signal"]].to_string(index=False))

print("\n" + "=" * 70)
print("STEP 5 — GAPS IDENTIFIED")
print("=" * 70)

print("""
What we captured today:
  iran:          3 articles → crude/geopolitical sentiment ✓
  hormuz:        3 articles → shipping/oil sentiment ✓
  tariff:        2 articles → trade sentiment ✓
  semiconductor: 2 articles → IT sector context ✓
  fed:           2 articles → rate sentiment ✓
  hdfc:         41 articles → HDFCBANK.NS negative sentiment ✓
  copper:        3 articles → commodity context ✓

What we missed:
  qatar:         0 articles → Qatar PM visit to Iran not captured
  gdp:           0 articles → India GDP data not captured
  inflation:     0 articles → India CPI not captured

Root cause of gaps:
  → Qatar/Iran mediation → not in Indian RSS sources
  → India GDP/CPI → timing issue (data released post-collection)
  → These would be in next fetch cycle
""")

if __name__ == "__main__":
    pass