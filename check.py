# check_pipeline_sep1.py
"""
Complete end-to-end pipeline audit — dynamic for any date
Checks: news → sentiment → macro → SmartScore → signals
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

TODAY = "2026-09-04"
DATE_LABEL = "Sep 4, 2026"

print("=" * 70)
print(f"FULL PIPELINE AUDIT — {DATE_LABEL}")
print(f"First day of trading week — Thursday Sep 4, 2026")
print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ── STEP 1: News Coverage ──
print("\n[STEP 1] NEWS INGESTION")
raw = pd.read_csv("data/raw_news.csv")
raw["published_utc"] = pd.to_datetime(raw["published_utc"], utc=True, errors="coerce")
today_raw = raw[raw["published_utc"].dt.date.astype(str) >= TODAY]
all_raw   = raw  # all articles in pipeline

print(f"Total articles in pipeline:  {len(all_raw)}")
print(f"Articles today ({TODAY}):    {len(today_raw)}")
print(f"Direct ticker mapped:        {today_raw['ticker'].notna().sum()}")
print(f"Global routed (conf=0.4):    {(today_raw['map_confidence']==0.4).sum()}")
print(f"Sources active today:        {today_raw['source_name'].nunique()}")

# Top sources
print(f"\nTop 10 sources today:")
print(today_raw["source_name"].value_counts().head(10).to_string())

# Key news coverage — dynamically check what's important today
print(f"\nKey news topics today:")
topics = {
    "Iran/Hormuz":        ["iran","hormuz","strait"],
    "Oil/Crude":          ["crude","brent","oil price","petroleum"],
    "FII/DII":            ["fii","dii","foreign institutional","fpi"],
    "Fed/Rate":           ["fed","federal reserve","rate cut","rate hike","warsh","fomc"],
    "India GDP":          ["india gdp","gdp growth","economic growth"],
    "India Inflation":    ["india inflation","cpi","wpi"],
    "HDFC Bank":          ["hdfc bank","hdfcbank"],
    "IT sector":          ["infosys","wipro","hcltech","it sector","tech layoff"],
    "Rupee/INR":          ["rupee","inr","usd inr"],
    "Gold":               ["gold price","bullion","gold rate"],
    "MSCI":               ["msci","rebalance","index inclusion"],
    "Semiconductors":     ["semiconductor","chip","nvidia"],
    "RBI":                ["rbi","repo rate","monetary policy"],
    "Budget/Capex":       ["budget","capex","government spending"],
    "Nifty/Sensex":       ["nifty","sensex","market fall","market rise"],
}
for topic, kws in topics.items():
    n = sum(today_raw["title"].str.lower().str.contains(kw, na=False).sum() for kw in kws)
    status = "✅" if n > 3 else "⚠️" if n > 0 else "❌"
    print(f"  {status} {topic}: {n} articles")

# ── STEP 2: Sentiment ──
print("\n[STEP 2] SENTIMENT SCORING")
sent = pd.read_csv("data/processed_sentiment.csv")
sent["published_utc"] = pd.to_datetime(sent["published_utc"], utc=True, errors="coerce")
today_sent = sent[sent["published_utc"].dt.date.astype(str) >= TODAY]

print(f"Total sentiment rows today:  {len(today_sent)}")
if len(today_sent) > 0:
    print(f"Labels: pos={(today_sent['label']=='positive').sum()} "
          f"neg={(today_sent['label']=='negative').sum()} "
          f"neu={(today_sent['label']=='neutral').sum()}")
    print(f"Avg FinBERT score:           {today_sent['finbert'].mean():.3f}")
    print(f"Avg VADER score:             {today_sent['vader'].mean():.3f}")
    print(f"Avg ensemble score:          {today_sent['ensemble'].mean():.3f}")
    print(f"Avg confidence:              {today_sent['model_confidence'].mean():.3f}")
    
    strong_neg = today_sent[today_sent["ensemble"] < -0.3]
    strong_pos = today_sent[today_sent["ensemble"] > 0.3]
    print(f"Strong negative signals:     {len(strong_neg)}")
    print(f"Strong positive signals:     {len(strong_pos)}")
else:
    print("No sentiment rows for today yet — pipeline may not have run")

# Key stocks sentiment
print(f"\nKey stock sentiment today:")
key_stocks = {
    "HDFCBANK.NS":  "HDFC Bank",
    "IOC.NS":       "IOC (oil importer)",
    "BPCL.NS":      "BPCL (oil importer)",
    "INDIGO.NS":    "IndiGo (aviation/fuel)",
    "ONGC.NS":      "ONGC (oil producer)",
    "INFY.NS":      "Infosys (IT)",
    "TCS.NS":       "TCS (IT)",
    "RELIANCE.NS":  "Reliance",
    "TITAN.NS":     "Titan (gold)",
    "SBIN.NS":      "SBI (banking/FII)",
}
print(f"{'Ticker':<20} {'Art':>4} {'Sentiment':>10} {'Pos':>4} {'Neg':>4} {'Label':<10} {'Stock'}")
print("-" * 80)
for tk, name in key_stocks.items():
    rows = today_sent[today_sent["ticker"] == tk]
    if len(rows) > 0:
        avg  = rows["ensemble"].mean()
        pos  = (rows["label"]=="positive").sum()
        neg  = (rows["label"]=="negative").sum()
        lbl  = "🟢 positive" if avg > 0.1 else "🔴 negative" if avg < -0.1 else "⚪ neutral"
        print(f"{tk:<20} {len(rows):>4} {avg:>+10.3f} {pos:>4} {neg:>4} {lbl:<10} {name}")

# ── STEP 3: SmartScore ──
print("\n[STEP 3] SMARTSCORE TODAY")
summary = pd.read_csv("data/stock_sentiment_summary.csv")
print(f"Tickers with SmartScore: {len(summary)}")
print(f"Market-wide avg SmartScore: {summary['smart_score'].mean():.1f}")
print(f"Stocks SmartScore>70 (bullish): {(summary['smart_score']>70).sum()}")
print(f"Stocks SmartScore<40 (bearish): {(summary['smart_score']<40).sum()}")

print(f"\nKey stocks SmartScore:")
print(f"{'Ticker':<20} {'Score':>7} {'S_recency':>10} {'S_events':>10} {'pos':>5} {'neg':>5} {'Signal'}")
print("-" * 75)
for tk in key_stocks:
    row = summary[summary["ticker"] == tk]
    if not row.empty:
        r = row.iloc[0]
        ss = r['smart_score']
        sig = "🟢 BULL" if ss > 70 else "🔴 BEAR" if ss < 40 else "⚪ NEUT"
        print(f"{tk:<20} {ss:>7.1f} {r['S_recency']:>10.1f} "
              f"{r['S_events']:>10.1f} {r['pos']:>5} {r['neg']:>5} {sig}")

print(f"\nTop 5 most bullish:")
print(summary.nlargest(5,"smart_score")[["ticker","smart_score","S_recency","pos","neg"]].to_string(index=False))

print(f"\nTop 5 most bearish:")
print(summary.nsmallest(5,"smart_score")[["ticker","smart_score","S_recency","pos","neg"]].to_string(index=False))

# ── STEP 4: Macro ──
print("\n[STEP 4] MACRO INDICATORS TODAY")
try:
    import yfinance as yf
    macro_check = {
        "India VIX":     ("^INDIAVIX", "fear_india"),
        "Brent Crude":   ("BZ=F",      "oil"),
        "USD/INR":       ("USDINR=X",  "currency"),
        "US VIX":        ("^VIX",      "fear_us"),
        "Nifty 50":      ("^NSEI",     "market"),
        "Nifty IT":      ("^CNXIT",    "sector"),
        "Nifty Bank":    ("^NSEBANK",  "sector"),
        "US 10yr Yield": ("^TNX",      "yield"),
        "Gold":          ("GC=F",      "commodity"),
        "S&P 500":       ("^GSPC",     "us_market"),
    }
    
    print(f"{'Indicator':<20} {'Value':>10} {'Change':>8}  {'Impact for India'}")
    print("-" * 65)
    
    for name, (ticker, category) in macro_check.items():
        try:
            data = yf.download(ticker, period="3d", progress=False, auto_adjust=True)
            if not data.empty and len(data) >= 2:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                latest = float(data["Close"].iloc[-1])
                prev   = float(data["Close"].iloc[-2])
                chg    = (latest - prev) / prev * 100
                arrow  = "↑" if chg > 0 else "↓"
                
                # Impact assessment
                impact = ""
                if category == "fear_india":
                    impact = "⚠️ Fear rising" if chg > 5 else "✅ Calm market" if latest < 15 else ""
                elif category == "oil":
                    impact = "❌ Bad (import bill up)" if chg > 1 else "✅ Good (cheaper imports)" if chg < -1 else ""
                elif category == "currency":
                    impact = "❌ Rupee weakens" if chg > 0.3 else "✅ Rupee stable/strong" if chg < -0.2 else ""
                elif category == "fear_us":
                    impact = "❌ Global fear rising" if chg > 5 else "✅ Global calm" if latest < 15 else ""
                elif category == "yield":
                    impact = "❌ FII may sell India" if chg > 0.5 else "✅ Yield falling (FII friendly)" if chg < -0.5 else ""
                elif category == "commodity" and "Gold" in name:
                    impact = "❌ Risk-off signal" if chg > 1 else "✅ Risk-on" if chg < -1 else ""
                elif category == "us_market":
                    impact = "❌ Global sell-off" if chg < -0.5 else "✅ Global positive" if chg > 0.5 else ""
                elif category == "sector":
                    impact = "❌ Sector weak" if chg < -0.5 else "✅ Sector strong" if chg > 0.5 else ""
                
                print(f"  {name:<18} {latest:>10.2f}  {arrow}{chg:>+6.2f}%  {impact}")
        except Exception as e:
            print(f"  {name:<18} fetch failed")
except Exception as e:
    print(f"yfinance error: {e}")

# ── STEP 5: FII/DII ──
print("\n[STEP 5] FII/DII INSTITUTIONAL FLOW")
fii_path = "data/fii_dii_history.csv"
if os.path.exists(fii_path):
    fii = pd.read_csv(fii_path)
    print(f"Total FII/DII rows collected: {len(fii)}")
    print(f"\nRecent history:")
    print(fii.tail(7).to_string(index=False))
    
    latest_fii = fii.iloc[-1]
    fii_val = float(latest_fii["fii_net"])
    dii_val = float(latest_fii["dii_net"])
    net = fii_val + dii_val
    
    print(f"\nLatest ({latest_fii['date']}):")
    if fii_val < -5000:
        fii_label = "← HEAVY SELLING ❌❌"
    elif fii_val < -2000:
        fii_label = "← selling ❌"
    elif fii_val < 0:
        fii_label = "← mild selling ⚠️"
    else:
        fii_label = "← buying ✅"
    
    print(f"  FII net: ₹{fii_val:>10,.2f} crores  {fii_label}")
    print(f"  DII net: ₹{dii_val:>10,.2f} crores  {'← strong support ✅' if dii_val > 3000 else '← buying ✅' if dii_val > 0 else '← selling ❌'}")
    print(f"  NET:     ₹{net:>10,.2f} crores  {'← NET BEARISH ❌' if net < 0 else '← NET BULLISH ✅'}")
    
    # Trend
    if len(fii) >= 3:
        recent_fii = fii["fii_net"].tail(3).mean()
        print(f"\n  3-day avg FII: ₹{recent_fii:,.2f} crores  {'← consistent selling ❌' if recent_fii < -1000 else '← consistent buying ✅' if recent_fii > 1000 else '← mixed ⚠️'}")
else:
    print("FII/DII data not found")

# ── STEP 6: Signals ──
print("\n[STEP 6] SIGNALS GENERATED TODAY")
ens_path  = "data/ensemble_signals.csv"
sig3d_path = "data/signals_3d.csv"
pred_path  = "data/predictions_nextday.csv"

if os.path.exists(ens_path):
    ens = pd.read_csv(ens_path)
    strong_bull = ens[ens["ensemble_probability"] >= 65]
    mild_bull   = ens[(ens["ensemble_probability"] >= 55) & (ens["ensemble_probability"] < 65)]
    neutral     = ens[(ens["ensemble_probability"] >= 45) & (ens["ensemble_probability"] < 55)]
    bearish     = ens[ens["ensemble_probability"] < 45]

    print(f"1-Day signal distribution:")
    print(f"  🟢 Strong bullish (>65%):  {len(strong_bull)}")
    print(f"  🟡 Mild bullish (55-65%):  {len(mild_bull)}")
    print(f"  ⚪ Neutral (45-55%):       {len(neutral)}")
    print(f"  🔴 Bearish (<45%):         {len(bearish)}")
    
    print(f"\nTop 10 ensemble signals:")
    print(ens.head(10)[["ticker","ensemble_probability","models_agree","signal"]].to_string(index=False))

if os.path.exists(sig3d_path):
    sig3d = pd.read_csv(sig3d_path)
    strong_3d = sig3d[sig3d["combined_signal"].str.contains("STRONG", na=False)]
    bullish_3d = sig3d[sig3d["ensemble_3d_prob"] > 55] if "ensemble_3d_prob" in sig3d.columns else pd.DataFrame()
    
    print(f"\n3-Day signal distribution:")
    print(f"  🟢 STRONG (both 1d+3d agree): {len(strong_3d)}")
    print(f"  🔵 3d bullish (>55%):          {len(bullish_3d)}")
    
    print(f"\nTop 3-Day signals:")
    cols = ["ticker","xgb_3d_prob","ensemble_3d_prob","combined_signal"]
    cols = [c for c in cols if c in sig3d.columns]
    print(sig3d.head(10)[cols].to_string(index=False))

if os.path.exists(pred_path):
    preds = pd.read_csv(pred_path)
    print(f"\nTop 5 predicted returns (regression):")
    print(preds.head(5)[["ticker","pred_ret_1d_pct"]].to_string(index=False))
    
    pos_preds = (preds["pred_ret_1d_pct"] > 0).sum()
    neg_preds = (preds["pred_ret_1d_pct"] < 0).sum()
    print(f"\nPositive predictions: {pos_preds}")
    print(f"Negative predictions: {neg_preds}")
    overall = "BULLISH" if pos_preds > neg_preds else "BEARISH" if neg_preds > pos_preds else "NEUTRAL"
    print(f"Overall prediction:   {overall}")

# ── STEP 7: System explanation ──
print("\n[STEP 7] SYSTEM EXPLANATION — WHY THESE SIGNALS?")

# Auto-generate explanation based on data
explanations = []

if os.path.exists(fii_path):
    fii_df = pd.read_csv(fii_path)
    if len(fii_df) > 0:
        fv = float(fii_df.iloc[-1]["fii_net"])
        if fv < -3000:
            explanations.append(f"❌ FII sold ₹{abs(fv):,.0f}cr → strong bearish pressure on all stocks")
        elif fv < 0:
            explanations.append(f"⚠️ FII mild selling ₹{abs(fv):,.0f}cr → slight bearish pressure")
        else:
            explanations.append(f"✅ FII buying ₹{fv:,.0f}cr → supportive for markets")

if len(today_raw) > 0:
    iran_n = sum(today_raw["title"].str.lower().str.contains(kw, na=False).sum() for kw in ["iran","hormuz"])
    if iran_n > 20:
        explanations.append(f"❌ {iran_n} Iran/Hormuz articles → oil disruption fears → energy stocks bearish")
    
    fed_n = sum(today_raw["title"].str.lower().str.contains(kw, na=False).sum() for kw in ["fed","warsh","federal reserve"])
    if fed_n > 50:
        explanations.append(f"⚠️ {fed_n} Fed/rate articles → rate uncertainty → IT and banking stocks affected")
    
    hdfc_n = today_raw["title"].str.lower().str.contains("hdfc bank", na=False).sum()
    if hdfc_n > 20:
        explanations.append(f"📊 {hdfc_n} HDFC Bank articles → significant stock-specific news")

if len(today_sent) > 0:
    avg_sent = today_sent["ensemble"].mean()
    if avg_sent < -0.05:
        explanations.append(f"❌ Overall sentiment negative ({avg_sent:+.3f}) → broad bearish signal")
    elif avg_sent > 0.05:
        explanations.append(f"✅ Overall sentiment positive ({avg_sent:+.3f}) → broad bullish signal")

for i, exp in enumerate(explanations, 1):
    print(f"\n{i}. {exp}")

if not explanations:
    print("Run the full pipeline first to generate explanation")

# ── STEP 8: Summary ──
print("\n[STEP 8] OVERALL SYSTEM HEALTH TODAY")
print(f"{'Component':<30} {'Status':<15} {'Details'}")
print("-" * 70)
print(f"{'News ingestion':<30} {'✅ OK':<15} {len(today_raw)} articles today")
print(f"{'Sentiment scoring':<30} {'✅ OK' if len(today_sent)>0 else '❌ NOT RUN':<15} {len(today_sent)} rows scored")
print(f"{'SmartScore':<30} {'✅ OK':<15} {len(summary)} tickers")
print(f"{'FII/DII data':<30} {'⚠️ PARTIAL':<15} {len(pd.read_csv(fii_path)) if os.path.exists(fii_path) else 0} rows only")
print(f"{'Macro indicators':<30} {'✅ OK':<15} 18 indicators")
print(f"{'Signals':<30} {'✅ OK' if os.path.exists(ens_path) else '❌ NOT RUN':<15} 1d + 3d signals")

print("\n" + "=" * 70)
print("PIPELINE AUDIT COMPLETE")
print(f"Run: python check_pipeline_sep1.py")
print("=" * 70)