# dashboard/app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Indian Stock Sentiment Analyser", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- helpers ----------
@st.cache_data
def load_csv_with_mtime(path, mtime, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()

def load_csv(path, parse_dates=None):
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(full_path):
        return pd.DataFrame()
    mtime = os.path.getmtime(full_path)
    return load_csv_with_mtime(full_path, mtime, parse_dates)

def fmt_dt(ts=None):
    return datetime.now().strftime("%Y-%m-%d %H:%M")

# ---------- helper functions for Market Intelligence Brief ----------

@st.cache_data(ttl=1800)
def get_macro_snapshot():
    try:
        import yfinance as yf
        macro_tickers = {
            "BZ=F":      ("crude_oil",  "Brent Crude"),
            "USDINR=X":  ("usd_inr",    "USD/INR"),
            "^VIX":      ("us_vix",     "US VIX"),
            "^INDIAVIX": ("india_vix",  "India VIX"),
            "GC=F":      ("gold",       "Gold"),
            "^TNX":      ("us_10y",     "US 10yr Yield"),
            "^GSPC":     ("sp500",      "S&P 500"),
            "^CNXIT":    ("nifty_it",   "Nifty IT"),
            "^NSEBANK":  ("nifty_bank", "Nifty Bank"),
            "^NSEI":     ("nifty50",    "Nifty 50"),
        }
        result = {}
        for ticker, (key, label) in macro_tickers.items():
            try:
                data = yf.download(ticker, period="3d", progress=False, auto_adjust=True)
                if not data.empty and len(data) >= 2:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    latest = float(data["Close"].iloc[-1])
                    prev   = float(data["Close"].iloc[-2])
                    chg    = (latest - prev) / prev * 100
                    result[key] = {"label": label, "value": latest, "change": chg}
            except:
                pass
        return result
    except:
        return {}

def get_top_signals(accuracy_df, streaks_df, ens_signals_df, summary_df, top_n=3):
    if accuracy_df.empty or streaks_df.empty or ens_signals_df.empty:
        return pd.DataFrame()
    try:
        merged = accuracy_df.merge(streaks_df, on="ticker", how="inner")
        merged = merged.merge(
            ens_signals_df[["ticker","ensemble_probability","signal"]],
            on="ticker", how="inner"
        )
        merged = merged.merge(
            summary_df[["ticker","smart_score"]],
            on="ticker", how="left"
        )
        merged = merged[merged["total_signals"] >= 3]
        merged["trust_score"] = (
            merged["combined_acc"].fillna(50) * 0.40 +
            merged["win_rate_10d"].fillna(50) * 0.35 +
            merged["ensemble_probability"].fillna(50) * 0.25
        )
        merged = merged[merged["ensemble_probability"] >= 45]
        return merged.nlargest(top_n, "trust_score")
    except:
        return pd.DataFrame()

def get_avoid_list(summary_df, ens_signals_df, top_n=4):
    if summary_df.empty:
        return pd.DataFrame()
    try:
        avoid = summary_df[summary_df["smart_score"] < 35].copy()
        if not ens_signals_df.empty and "ticker" in ens_signals_df.columns:
            avoid = avoid.merge(
                ens_signals_df[["ticker","ensemble_probability"]],
                on="ticker", how="left"
            )
        return avoid.nsmallest(top_n, "smart_score")
    except:
        return summary_df.nsmallest(top_n, "smart_score")

def get_avoid_reason(ticker, score):
    reasons = {
        "COFORGE":    "Chairman resigned — governance risk",
        "INDIGO":     "Crude $100+ — aviation fuel costs surging",
        "IOC":        "Oil importer — crude prices elevated",
        "BPCL":       "Oil importer — crude prices elevated",
        "SPICEJET":   "Delisted / financial stress",
        "YESBANK":    "Persistent negative sentiment",
        "PERSISTENT": "IT services — sector under pressure",
        "HINDPETRO":  "Oil refiner — crude price pressure",
        "IRCTC":      "Negative sentiment — recent selloff",
        "MARUTI":     "Auto sector — cost pressures",
        "BANKBARODA": "Banking — rate hike sensitivity",
        "IRCON":      "Infrastructure — negative sentiment",
        "BANSALWIRE": "Industrial wire — weak demand signals",
        "HAVELLS":    "Capital goods — sector pressure",
    }
    clean = ticker.replace(".NS","")
    if clean in reasons:
        return reasons[clean]
    if score < 25:
        return f"Heavy negative sentiment ({score:.0f}/100)"
    return f"Weak signals ({score:.0f}/100 SmartScore)"

def get_sector_heatmap(macro):
    sectors = []
    it_chg   = macro.get("nifty_it",  {}).get("change", 0)
    bnk_chg  = macro.get("nifty_bank",{}).get("change", 0)
    oil_chg  = macro.get("crude_oil", {}).get("change", 0)
    vix      = macro.get("india_vix", {}).get("value",  15)
    us_vix   = macro.get("us_vix",    {}).get("value",  15)

    if it_chg > 0.5:
        sectors.append(("IT", "🟢", f"+{it_chg:.1f}%", "Recovering — oversold bounce"))
    elif it_chg < -1:
        sectors.append(("IT", "🔴", f"{it_chg:.1f}%", "Rate hike fears + global selloff"))
    else:
        sectors.append(("IT", "🟡", f"{it_chg:.1f}%", "Mixed — await macro clarity"))

    if bnk_chg > 0.5:
        sectors.append(("Banking", "🟢", f"+{bnk_chg:.1f}%", "DII buying + rate hold hopes"))
    elif bnk_chg < -0.5:
        sectors.append(("Banking", "🔴", f"{bnk_chg:.1f}%", "FII selling + rate hike fears"))
    else:
        sectors.append(("Banking", "🟡", f"{bnk_chg:.1f}%", "Volatile — FII vs DII"))

    if oil_chg > 2:
        sectors.append(("FMCG", "🟢", "Safe Haven", f"Crude +{oil_chg:.1f}% → defensive rotation"))
        sectors.append(("Energy Import", "🔴", "Under Pressure", f"Crude +{oil_chg:.1f}% → IOC/BPCL suffer"))
        sectors.append(("Energy Prod.", "🟢", "Benefits", f"Crude +{oil_chg:.1f}% → ONGC gains"))
    elif oil_chg < -2:
        sectors.append(("FMCG", "🟡", "Stable", "No crude pressure today"))
        sectors.append(("Energy Import", "🟢", "Relief", f"Crude {oil_chg:.1f}% → IOC/BPCL relief"))
        sectors.append(("Energy Prod.", "🔴", "Pressure", f"Crude {oil_chg:.1f}% → ONGC revenue falls"))
    else:
        sectors.append(("FMCG", "🟡", "Stable", "Defensive — holds in volatility"))
        sectors.append(("Energy", "🟡", "Mixed", "Crude stable — no strong signal"))

    if us_vix > 20 or vix > 18:
        sectors.append(("Pharma", "🟢", "Safe Haven", "High VIX → defensive rotation"))
    else:
        sectors.append(("Pharma", "🟡", "Neutral", "No major catalyst"))

    sectors.append(("Defence", "🟢", "Positive", "Geopolitical tension → defence spending theme"))

    return sectors

def get_tomorrow_outlook(macro, fii_df, news_counts):
    events = []
    if not fii_df.empty and len(fii_df) >= 3:
        fii_3d = fii_df["fii_net"].tail(3).mean()
        if fii_3d < -500:
            events.append(("⚠️", f"FII selling trend (-₹{abs(fii_3d):,.0f}cr 3-day avg)", "Watch for reversal at key support levels"))
        elif fii_3d > 500:
            events.append(("✅", f"FII buying trend (+₹{fii_3d:,.0f}cr 3-day avg)", "Supportive for next session"))

    crude_chg = macro.get("crude_oil", {}).get("change", 0)
    if crude_chg > 3:
        events.append(("🔴", f"Crude elevated (+{crude_chg:.1f}%)", "Energy importers remain at risk"))
    elif crude_chg < -2:
        events.append(("🟢", f"Crude falling ({crude_chg:.1f}%)", "Relief rally possible for importers"))

    us_vix_chg = macro.get("us_vix", {}).get("change", 0)
    if us_vix_chg < -5:
        events.append(("✅", f"Global fear dropping (VIX {us_vix_chg:.1f}%)", "Risk-on — equity positive"))
    elif us_vix_chg > 5:
        events.append(("⚠️", f"Global fear rising (VIX +{us_vix_chg:.1f}%)", "Defensive positioning recommended"))

    fed_count = news_counts.get("Fed/Rate", 0)
    rbi_count = news_counts.get("RBI", 0)
    if fed_count > 50:
        events.append(("🏛️", "Fed rate decision imminent (high article count)", "Expect elevated market volatility"))
    elif fed_count > 20:
        events.append(("🏛️", "Fed meeting approaching", "Watch for rate decision impact on FII flows"))
    if rbi_count > 100:
        events.append(("🏛️", "RBI policy in focus", "Banking stocks sensitive to outcome"))

    return events

def classify_market_regime(macro, fii_df):
    """Classify current market regime based on macro + FII data"""
    fii_5d_avg = 0
    if not fii_df.empty and len(fii_df) >= 3:
        fii_5d_avg = fii_df["fii_net"].tail(5).mean()

    nifty_chg  = macro.get("nifty50",   {}).get("change", 0)
    us_10y_val = macro.get("us_10y",    {}).get("value",  4.0)
    india_vix  = macro.get("india_vix", {}).get("value",  15)
    us_vix     = macro.get("us_vix",    {}).get("value",  15)

    if fii_5d_avg < -3000 or us_10y_val > 4.8 or us_vix > 25:
        return "RISK-OFF", "🔴", "Trust 3d signals more. Avoid weak 1d signals."
    elif fii_5d_avg > 1000 and nifty_chg > 0 and us_vix < 18:
        return "RISK-ON", "🟢", "Both 1d and 3d signals reliable."
    elif india_vix > 18 or us_vix > 22:
        return "HIGH VOLATILITY", "🟠", "Signals less reliable. Use conviction matrix only."
    else:
        return "NEUTRAL", "🟡", "Moderate signal reliability. Filter by conviction."

def detect_unusual_activity(summary_df, hist_df):
    """Detect stocks with unusual sentiment or volume activity"""
    alerts = []
    if summary_df.empty or hist_df.empty:
        return alerts
    try:
        if "date" not in hist_df.columns:
            return alerts
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        cutoff = hist_df["date"].max() - pd.Timedelta(days=30)
        recent_hist = hist_df[hist_df["date"] >= cutoff]
        for _, row in summary_df.iterrows():
            tk = row["ticker"]
            tk_hist = recent_hist[recent_hist["ticker"] == tk]
            if len(tk_hist) < 5:
                continue
            avg_score = tk_hist["smart_score"].mean()
            avg_arts  = tk_hist["total"].mean() if "total" in tk_hist.columns else 0
            today_score = row.get("smart_score", avg_score)
            today_arts  = row.get("total", 0)
            if today_score > avg_score + 20:
                alerts.append(("🚨", tk.replace(".NS",""), f"Sentiment surge +{today_score-avg_score:.0f} vs 30d avg ({avg_score:.0f}→{today_score:.0f})"))
            elif today_score < avg_score - 20:
                alerts.append(("🚨", tk.replace(".NS",""), f"Sentiment crash -{avg_score-today_score:.0f} vs 30d avg ({avg_score:.0f}→{today_score:.0f})"))
            if avg_arts > 2 and today_arts > avg_arts * 2.5:
                alerts.append(("📢", tk.replace(".NS",""), f"News volume {today_arts/avg_arts:.1f}x normal ({int(today_arts)} vs avg {int(avg_arts)})"))
    except Exception as e:
        pass
    return alerts[:6]

def get_fundamental_signal(ticker_ns):
    """Get key fundamentals from yfinance for display"""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker_ns).info
        pe      = info.get("trailingPE")
        fpe     = info.get("forwardPE")
        eps_g   = info.get("earningsGrowth")
        de      = info.get("debtToEquity")
        roe     = info.get("returnOnEquity")
        target  = info.get("targetMeanPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        rec     = info.get("recommendationKey","").upper()
        n_anal  = info.get("numberOfAnalystOpinions", 0)
        hi52    = info.get("fiftyTwoWeekHigh", 0)
        lo52    = info.get("fiftyTwoWeekLow",  0)
        pos52   = round((current - lo52) / max(hi52 - lo52, 1) * 100, 1) if current and hi52 > lo52 else None
        score   = 0
        if pe and pe < 15:   score += 2
        elif pe and pe > 40: score -= 2
        if eps_g and eps_g > 0.10: score += 2
        elif eps_g and eps_g < 0:  score -= 2
        if de and de < 0.5:  score += 1
        elif de and de > 2:  score -= 2
        if roe and roe > 0.15: score += 1
        if pos52 and pos52 < 30: score += 1
        elif pos52 and pos52 > 85: score -= 1
        if score >= 4:   fsig, fcolor = "STRONG 🟢", "success"
        elif score >= 2: fsig, fcolor = "MODERATE 🟡", "warning"
        elif score >= 0: fsig, fcolor = "NEUTRAL ⚪", "info"
        else:            fsig, fcolor = "WEAK 🔴", "error"
        upside = round((target - current) / current * 100, 1) if target and current else None
        return {
            "pe": pe, "forward_pe": fpe, "eps_growth": eps_g,
            "debt_equity": de, "roe": roe,
            "target": target, "current": current, "upside": upside,
            "recommendation": rec, "n_analysts": n_anal,
            "pos_52w": pos52, "signal": fsig, "color": fcolor, "score": score
        }
    except Exception as e:
        return None

# ---------- load data ----------
summary     = load_csv("data/stock_sentiment_summary.csv")
hist        = load_csv("data/history/stock_sentiment_summary_history.csv", parse_dates=["date"])
preds       = load_csv("data/predictions_nextday.csv")
signals     = load_csv("data/xgb_signals.csv")

st.title("Indian Stock Sentiment Analyser")
st.caption(f"Last updated: {fmt_dt()}  •  Data: Google News, BS, ET Markets, Mint, BusinessLine")

if summary.empty:
    st.warning("No summary found. Run the pipeline first.")
    st.stop()

if "date" in hist.columns:
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")

# -------------------------------- TABS --------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Market Brief", "Trade Ideas", "Signals", "Stock Accuracy", "Stock Intelligence", "Model Health", "Tech & Workflow"]
)

# ============================================================
# TAB 1 — MARKET INTELLIGENCE BRIEF  (drop-in replacement)
# ============================================================
# Paste this block in place of the existing `with tab1:` block
# in dashboard/app.py. All helper functions above it are unchanged.
# ============================================================

with tab1:

    # ── Load data ──
    ens_signals = load_csv("data/ensemble_signals.csv")
    streaks     = load_csv("data/stock_streaks.csv")
    accuracy    = load_csv("data/stock_accuracy.csv")
    fii_dii     = load_csv("data/fii_dii_history.csv")
    metrics_df  = load_csv("data/modeling/model_metrics.csv")
    pcr_df      = load_csv("data/pcr_history.csv")

    with st.spinner("Fetching live macro data…"):
        macro = get_macro_snapshot()

    # ── Compute scalar values ──
    crude_chg  = macro.get("crude_oil", {}).get("change", 0)
    us_vix_val = macro.get("us_vix",    {}).get("value",  15)
    us_vix_chg = macro.get("us_vix",    {}).get("change", 0)
    us_10y_chg = macro.get("us_10y",    {}).get("change", 0)
    us_10y_val = macro.get("us_10y",    {}).get("value",  4.0)
    fii_latest = 0; dii_latest = 0
    if not fii_dii.empty:
        fii_latest = float(fii_dii.iloc[-1]["fii_net"])
        dii_latest = float(fii_dii.iloc[-1]["dii_net"])

    pcr_latest = 1.0; pcr_vol_latest = 1.0
    pcr_change = 0.0; pcr_date = "N/A"
    pcr_zscore = 0.0; pcr_5d_avg = 1.0; pcr_regime = "N/A"
    if not pcr_df.empty:
        pcr_df["date"] = pd.to_datetime(pcr_df["date"])
        pcr_row        = pcr_df.sort_values("date").iloc[-1]
        pcr_latest     = float(pcr_row.get("pcr_oi",      1.0))
        pcr_vol_latest = float(pcr_row.get("pcr_vol",     1.0))
        pcr_change     = float(pcr_row.get("pcr_change",  0.0))
        pcr_date       = str(pcr_row["date"].date())
        pcr_zscore     = float(pcr_row.get("pcr_zscore",  0.0))
        pcr_5d_avg     = float(pcr_row.get("pcr_oi_5d_avg", pcr_latest))
        pcr_regime_raw = str(pcr_row.get("pcr_regime",    ""))
        pcr_regime     = "STRESS" if pcr_latest >= 1.2 else "NEUTRAL" if pcr_latest >= 0.8 else "BULLISH"

    # ── News counts ──
    raw_news_path = os.path.join(BASE_DIR, "data/raw_news.csv")
    news_counts   = {}
    news_source   = "estimated"
    if os.path.exists(raw_news_path):
        try:
            raw_news = load_csv("data/raw_news.csv")
            if not raw_news.empty and "published_utc" in raw_news.columns:
                raw_news["published_utc"] = pd.to_datetime(raw_news["published_utc"], utc=True, errors="coerce")
                cutoff     = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
                today_news = raw_news[raw_news["published_utc"] >= cutoff]
                if len(today_news) < 50:
                    cutoff     = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
                    today_news = raw_news[raw_news["published_utc"] >= cutoff]
                topics = {
                    "Oil/Crude":      ["crude","brent","oil price"],
                    "RBI":            ["rbi","repo rate","monetary policy"],
                    "Iran/Hormuz":    ["iran","hormuz","strait"],
                    "Fed/Rate":       ["fed","federal reserve","rate hike","fomc"],
                    "IT sector":      ["infosys","wipro","hcltech","it sector"],
                    "Gold":           ["gold price","bullion","gold etf"],
                    "Nifty/Sensex":   ["nifty","sensex"],
                    "FII/DII":        ["fii","dii","foreign institutional"],
                }
                for topic, kws in topics.items():
                    count = sum(today_news["title"].str.lower().str.contains(kw, na=False).sum() for kw in kws)
                    if count > 0:
                        news_counts[topic] = count
                if news_counts:
                    news_source = "live"
        except:
            pass
    if not news_counts and not summary.empty:
        try:
            oil_stocks  = ["IOC.NS","BPCL.NS","ONGC.NS","INDIGO.NS","HINDPETRO.NS"]
            it_stocks   = ["INFY.NS","TCS.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS"]
            bank_stocks = ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS"]
            def avg_articles(tickers):
                rows = summary[summary["ticker"].isin(tickers)]
                return int(rows["total"].sum()) if not rows.empty else 0
            oc = avg_articles(oil_stocks); ic = avg_articles(it_stocks)
            bc = avg_articles(bank_stocks)
            if oc > 0: news_counts["Oil/Crude"] = oc
            if ic > 0: news_counts["IT sector"]  = ic
            if bc > 0: news_counts["Banking"]    = bc
        except:
            pass

    # ── Headwind / risk score ──
    risk_score = 0; risk_reasons = []
    if us_vix_val > 25:      risk_score += 30; risk_reasons.append(f"US VIX high ({us_vix_val:.1f})")
    elif us_vix_val > 20:    risk_score += 15; risk_reasons.append(f"US VIX elevated ({us_vix_val:.1f})")
    if crude_chg > 3:        risk_score += 25; risk_reasons.append(f"Crude surging (+{crude_chg:.1f}%)")
    elif crude_chg > 1.5:    risk_score += 12; risk_reasons.append(f"Crude rising (+{crude_chg:.1f}%)")
    if us_10y_chg > 0.5:     risk_score += 15; risk_reasons.append(f"US yield rising (+{us_10y_chg:.2f}%)")
    if us_10y_val > 4.8:     risk_score += 10; risk_reasons.append(f"US 10yr near 5% ({us_10y_val:.2f}%)")
    if fii_latest < -5000:   risk_score += 25; risk_reasons.append("Heavy FII selling")
    elif fii_latest < -2000: risk_score += 12; risk_reasons.append("FII selling")
    elif fii_latest < -500:  risk_score +=  8; risk_reasons.append("FII mild selling")
    if us_vix_chg > 5:       risk_score += 10; risk_reasons.append("Global fear rising")
    fed_count = news_counts.get("Fed/Rate", 0)
    if fed_count > 50:       risk_score += 20; risk_reasons.append("Fed decision imminent")
    elif fed_count > 20:     risk_score += 10; risk_reasons.append("Fed meeting approaching")
    rbi_count = news_counts.get("RBI", 0)
    if rbi_count > 100:      risk_score += 10; risk_reasons.append("RBI policy focus")
    if pcr_latest >= 1.3:    risk_score += 15; risk_reasons.append(f"High PCR ({pcr_latest:.2f})")
    elif pcr_latest >= 1.1:  risk_score +=  8; risk_reasons.append(f"Elevated PCR ({pcr_latest:.2f})")
    elif pcr_latest <= 0.6:  risk_score +=  5; risk_reasons.append(f"Low PCR ({pcr_latest:.2f})")
    risk_score = min(risk_score, 100)

    risk_color  = "#dc2626" if risk_score >= 50 else "#d97706" if risk_score >= 25 else "#16a34a"
    risk_label  = "HIGH" if risk_score >= 50 else "MODERATE" if risk_score >= 25 else "LOW"
    risk_badge  = "badge-red" if risk_score >= 50 else "badge-amber" if risk_score >= 25 else "badge-green"
    score_pct   = risk_score  # for arrow position

    # ── Market regime ──
    regime_label, regime_icon, regime_advice = classify_market_regime(macro, fii_dii)

    # ── Unusual activity ──
    alerts = detect_unusual_activity(summary, hist)

    # ── Top signals (XGBoost ensemble) ──
    top_signals = get_top_signals(accuracy, streaks, ens_signals, summary, top_n=5)
    # Also get raw ens_signals sorted by probability for display
    ens_top5 = pd.DataFrame()
    if not ens_signals.empty and "ensemble_probability" in ens_signals.columns:
        ens_top5 = (ens_signals[ens_signals["ensemble_probability"] >= 65]
                    .sort_values("ensemble_probability", ascending=False)
                    .head(5))

    # ── Sectors ──
    sectors = get_sector_heatmap(macro)

    # ── Model accuracy ──
    xgb_acc = ens_acc = None
    xgb_rows_count = ens_rows_count = 0
    if not metrics_df.empty:
        metrics_df["train_date"] = pd.to_datetime(metrics_df["train_date"], errors="coerce")
        xgb_r = metrics_df[metrics_df["model"] == "XGBoost_Classifier"].sort_values("train_date")
        ens_r = metrics_df[metrics_df["model"] == "Voting_Ensemble"].sort_values("train_date")
        if not xgb_r.empty:
            xgb_acc = xgb_r.iloc[-1]["direction_accuracy"] * 100
            xgb_rows_count = int(xgb_r.iloc[-1]["rows"])
        if not ens_r.empty:
            ens_acc = ens_r.iloc[-1]["direction_accuracy"] * 100
            ens_rows_count = int(ens_r.iloc[-1]["rows"])

    # ── Suppress counts ──
    xgb_suppressed = 0; ens_suppressed = 0
    if not ens_signals.empty and "ensemble_probability" in ens_signals.columns:
        xgb_suppressed = int((ens_signals.get("up_probability", ens_signals["ensemble_probability"]) < 65).sum()) if "up_probability" in ens_signals.columns else 0
        ens_suppressed = int((ens_signals["ensemble_probability"] < 65).sum())

    # ── Avoid list ──
    avoid_df = get_avoid_list(summary, ens_signals, top_n=4)

    # ── Tomorrow outlook ──
    tomorrow = get_tomorrow_outlook(macro, fii_dii, news_counts)

    # ── Geo signals ──
    iran_count  = news_counts.get("Iran/Hormuz", 0)
    iran_level  = "HIGH" if iran_count > 50 else "MEDIUM" if iran_count > 20 else "LOW"
    iran_dot    = "dot-red" if iran_count > 50 else "dot-amber" if iran_count > 20 else "dot-green"
    fed_level   = "HIGH" if fed_count > 50 else "MEDIUM" if fed_count > 20 else "LOW"
    fed_dot     = "dot-red" if fed_count > 50 else "dot-amber" if fed_count > 20 else "dot-green"
    u10v        = macro.get("us_10y", {}).get("value", 4.5)
    jap_level   = "HIGH" if u10v > 4.9 else "MEDIUM" if u10v > 4.7 else "LOW"
    jap_dot     = "dot-red" if u10v > 4.9 else "dot-amber" if u10v > 4.7 else "dot-green"
    gchg        = macro.get("gold", {}).get("change", 0)
    gold_level  = "HIGH" if gchg > 1.5 else "MEDIUM" if gchg > 0 else "LOW"
    gold_dot    = "dot-red" if gchg > 1.5 else "dot-amber" if gchg > 0 else "dot-green"

    # ─────────────────────────────────────────────────────────
    # Helper: build a dot-row HTML snippet
    # ─────────────────────────────────────────────────────────
    def dot_row(dot_cls, label, val, chg, chg_cls="flat"):
        return f"""
        <div class="stat-row">
          <div class="dot {dot_cls}"></div>
          <div class="stat-label">{label}</div>
          <div class="stat-val">{val}</div>
          <div class="stat-chg {chg_cls}">{chg}</div>
        </div>"""

    def badge(text, cls):
        return f'<span class="badge {cls}">{text}</span>'

    def card_head(title, badge_html):
        return f"""<div class="card-head"><h2>{title}</h2>{badge_html}</div>"""

    # ─────────────────────────────────────────────────────────
    # CSS (injected once)
    # ─────────────────────────────────────────────────────────
    st.html("""
    <style>
    /* ── Token system ── */
    :root {
      --bg:#f7f8fa; --surface:#ffffff; --surface2:#f0f2f5; --border:#e4e7ec;
      --fg:#111827; --fg2:#4b5563; --fg3:#9ca3af;
      --green:#16a34a; --green-bg:#dcfce7;
      --red:#dc2626;   --red-bg:#fee2e2;
      --amber:#d97706; --amber-bg:#fef3c7;
      --accent:#2563eb; --blue-bg:#dbeafe;
      --radius:10px;
      --font-mono:'JetBrains Mono','Courier New',monospace;
      --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
    }
    @media(prefers-color-scheme:dark){
      :root{
        --bg:#0f1117; --surface:#1a1d27; --surface2:#22263a; --border:#2d3148;
        --fg:#f1f5f9; --fg2:#94a3b8; --fg3:#475569;
        --green:#22c55e; --green-bg:#14532d55;
        --red:#f87171;   --red-bg:#7f1d1d55;
        --amber:#fbbf24; --amber-bg:#78350f55;
        --blue-bg:#1e3a8a55;
      }
    }
    .mib *{box-sizing:border-box;margin:0;padding:0}
    .mib{font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif;font-size:15px;line-height:1.5;color:var(--fg)}

    /* grid */
    .row3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:14px}
    .row2{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:14px}
    @media(max-width:900px){.row3{grid-template-columns:1fr 1fr}.row2{grid-template-columns:1fr}}
    @media(max-width:600px){.row3{grid-template-columns:1fr}}

    /* card */
    .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
    .card-head{display:flex;align-items:center;justify-content:space-between;padding:10px 14px 8px;border-bottom:1px solid var(--border)}
    .card-head h2{font-size:12px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;color:var(--fg3)}
    .card-body{padding:12px 14px}

    /* badge */
    .badge{font-size:12px;font-weight:600;padding:2px 8px;border-radius:20px}
    .badge-green{background:var(--green-bg);color:var(--green)}
    .badge-red{background:var(--red-bg);color:var(--red)}
    .badge-amber{background:var(--amber-bg);color:var(--amber)}
    .badge-blue{background:var(--blue-bg);color:var(--accent)}

    /* stat-row */
    .stat-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border);font-size:14px}
    .stat-row:last-child{border-bottom:none}
    .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
    .dot-green{background:var(--green)}.dot-red{background:var(--red)}.dot-amber{background:var(--amber)}.dot-gray{background:var(--fg3)}
    .stat-label{flex:1;color:var(--fg2)}
    .stat-val{font-family:var(--font-mono);font-size:13px;font-weight:500;color:var(--fg)}
    .stat-chg{font-family:var(--font-mono);font-size:12px;min-width:52px;text-align:right}
    .up{color:var(--green)}.down{color:var(--red)}.flat{color:var(--fg3)}

    /* score hero */
    .score-hero{text-align:center;padding:14px 14px 10px}
    .score-big{font-size:58px;font-weight:700;font-family:var(--font-mono);line-height:1;margin-bottom:4px}
    .score-label{font-size:12px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px}
    .score-bar-wrap{background:var(--surface2);border-radius:4px;height:6px;margin:8px 0 4px;overflow:hidden}
    .score-bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#16a34a 0%,#d97706 50%,#dc2626 100%);width:100%}
    .score-pointer{position:relative;height:12px;margin-bottom:8px}
    .score-arrow{position:absolute;width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-bottom:9px solid var(--fg);transform:translateX(-50%)}
    .score-sub-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
    .score-sub{background:var(--surface2);border-radius:7px;padding:8px 10px;text-align:center}
    .score-sub .val{font-family:var(--font-mono);font-size:18px;font-weight:600}
    .score-sub .lbl{font-size:12px;color:var(--fg3);margin-top:2px}

    /* signal rows */
    .signal-row{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--border)}
    .signal-row:last-child{border-bottom:none}
    .signal-rank{font-size:12px;font-weight:700;color:var(--fg3);font-family:var(--font-mono);width:18px;text-align:center}
    .signal-ticker{font-family:var(--font-mono);font-size:14px;font-weight:600;flex:1}
    .signal-conf-bar{width:70px;height:5px;background:var(--surface2);border-radius:3px;overflow:hidden}
    .signal-conf-fill{height:100%;border-radius:3px;background:var(--green)}
    .signal-pct{font-family:var(--font-mono);font-size:13px;font-weight:600;color:var(--green);min-width:42px;text-align:right}
    .signal-dir{font-size:12px;font-weight:600;padding:2px 7px;border-radius:4px}
    .dir-up{background:var(--green-bg);color:var(--green)}
    .dir-down{background:var(--red-bg);color:var(--red)}

    /* ua rows */
    .ua-row{padding:7px 0;border-bottom:1px solid var(--border)}
    .ua-row:last-child{border-bottom:none}
    .ua-top{display:flex;align-items:center;gap:8px;margin-bottom:3px}
    .ua-ticker{font-family:var(--font-mono);font-size:14px;font-weight:600;flex:1}
    .ua-vol-bar-wrap{background:var(--surface2);border-radius:3px;height:4px;overflow:hidden}
    .ua-vol-bar-fill{height:100%;border-radius:3px}
    .ua-detail{display:flex;gap:12px;margin-top:4px;font-size:12px;color:var(--fg3);font-family:var(--font-mono)}

    /* sector rows */
    .sector-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border)}
    .sector-row:last-child{border-bottom:none}
    .sector-name{flex:1;font-size:14px;color:var(--fg2)}
    .sector-bar-wrap{width:80px;background:var(--surface2);border-radius:3px;height:5px;overflow:hidden}
    .sector-bar-fill{height:100%;border-radius:3px}
    .sector-pct{font-family:var(--font-mono);font-size:13px;min-width:32px;text-align:right}
    .sector-lbl{font-size:12px;font-weight:600;padding:2px 7px;border-radius:4px;min-width:52px;text-align:center}

    /* PCR */
    .pcr-main-val{font-family:var(--font-mono);font-size:26px;font-weight:700;margin-bottom:2px}
    .pcr-sub{font-size:12px;color:var(--fg3);margin-bottom:8px}
    .pcr-mini-row{display:flex;justify-content:space-between;align-items:center;font-size:12px;padding:4px 0;border-bottom:1px solid var(--border)}
    .pcr-mini-row:last-child{border-bottom:none}
    .pcr-mini-label{color:var(--fg3)}
    .pcr-mini-val{font-family:var(--font-mono);font-weight:500;font-size:13px}
    .section-title{font-size:12px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;color:var(--fg3);margin-bottom:8px}

    /* high trust */
    .ht-item{padding:9px 0;border-bottom:1px solid var(--border)}
    .ht-item:last-child{border-bottom:none}
    .ht-ticker{font-family:var(--font-mono);font-size:15px;font-weight:700}
    .ht-stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
    .ht-stat{background:var(--surface2);border-radius:7px;padding:6px 8px}
    .ht-stat .val{font-family:var(--font-mono);font-size:16px;font-weight:600}
    .ht-stat .lbl{font-size:12px;color:var(--fg3);margin-top:1px}

    /* weak signals */
    .weak-row{display:flex;gap:10px;align-items:flex-start;padding:7px 0;border-bottom:1px solid var(--border)}
    .weak-row:last-child{border-bottom:none}
    .weak-ticker{font-family:var(--font-mono);font-size:14px;font-weight:600;width:86px;flex-shrink:0}
    .weak-info{flex:1}
    .weak-pct{font-family:var(--font-mono);font-size:13px;font-weight:600;color:var(--fg2)}
    .weak-reason{font-size:12px;color:var(--fg3);margin-top:2px}

    /* news mini grid */
    .news-mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid var(--border)}
    .news-mini{background:var(--surface2);border-radius:7px;padding:7px 9px;text-align:center}
    .news-mini .val{font-family:var(--font-mono);font-size:20px;font-weight:700}
    .news-mini .lbl{font-size:12px;color:var(--fg3);margin-top:1px}

    /* watch rows */
    .watch-row{display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border)}
    .watch-row:last-child{border-bottom:none}
    .watch-ticker{font-family:var(--font-mono);font-size:14px;font-weight:600;flex:1}
    .watch-reason{font-size:12px;color:var(--fg3)}
    .watch-tag{font-size:12px;font-weight:600;padding:2px 8px;border-radius:4px}

    /* geo section */
    .geo-section{padding:10px 0}
    .geo-section+.geo-section{border-top:1px solid var(--border)}

    /* risk footer */
    .risk-footer{background:#1e1b4b;color:#e0e7ff;padding:12px 18px;border-radius:10px;display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;border:1px solid #4338ca}
    .risk-footer-text h3{font-size:14px;font-weight:700;color:#a5b4fc;letter-spacing:.3px}
    .risk-footer-text p{font-size:12px;color:#818cf8;margin-top:2px}
    .risk-pills{display:flex;gap:8px;flex-wrap:wrap}
    .rpill{font-size:12px;font-weight:600;padding:4px 12px;border-radius:20px;font-family:var(--font-mono)}
    .rp-score{background:#7c3aed33;color:#a78bfa;border:1px solid #7c3aed55}
    .rp-xgb{background:#dc262633;color:#fca5a5;border:1px solid #dc262655}
    .rp-ens{background:#ea580c33;color:#fdba74;border:1px solid #ea580c55}
    .rp-stance{background:#16a34a33;color:#86efac;border:1px solid #16a34a55}

    /* util */
    .sub-caption{font-size:12px;color:var(--fg3);margin-top:6px;padding-top:6px;border-top:1px solid var(--border)}
    </style>
    """)

    # ─────────────────────────────────────────────────────────
    # Build top-signal rows HTML
    # ─────────────────────────────────────────────────────────
    def build_signal_rows():
        rows_html = ""
        if not ens_top5.empty:
            for i, (_, row) in enumerate(ens_top5.iterrows()):
                tk   = row["ticker"].replace(".NS", "")
                prob = float(row["ensemble_probability"])
                sig  = str(row.get("signal", "UP"))
                dircls = "dir-up" if "UP" in sig.upper() or "BULL" in sig.upper() else "dir-down"
                dir_lbl = "UP" if "UP" in sig.upper() or "BULL" in sig.upper() else "DOWN"
                bar_w = min(int(prob), 100)
                rows_html += f"""
                <div class="signal-row">
                  <div class="signal-rank">#{i+1}</div>
                  <div class="signal-ticker">{tk}</div>
                  <div class="signal-conf-bar"><div class="signal-conf-fill" style="width:{bar_w}%"></div></div>
                  <div class="signal-pct">{prob:.1f}%</div>
                  <div class="signal-dir {dircls}">{dir_lbl}</div>
                </div>"""
        elif not top_signals.empty:
            for i, (_, row) in enumerate(top_signals.iterrows()):
                tk   = row["ticker"].replace(".NS", "")
                prob = float(row.get("ensemble_probability", row.get("trust_score", 65)))
                bar_w = min(int(prob), 100)
                rows_html += f"""
                <div class="signal-row">
                  <div class="signal-rank">#{i+1}</div>
                  <div class="signal-ticker">{tk}</div>
                  <div class="signal-conf-bar"><div class="signal-conf-fill" style="width:{bar_w}%"></div></div>
                  <div class="signal-pct">{prob:.1f}%</div>
                  <div class="signal-dir dir-up">UP</div>
                </div>"""
        else:
            rows_html = '<div style="font-size:12px;color:var(--fg3);padding:8px 0">No signals above 65% today</div>'
        return rows_html

    # ─────────────────────────────────────────────────────────
    # Build UA rows HTML
    # ─────────────────────────────────────────────────────────
    def build_ua_rows():
        if not alerts:
            return '<div style="font-size:12px;color:var(--fg3);padding:8px 0">No unusual activity detected</div>'
        html = ""
        for icon, tk, msg in alerts[:4]:
            # classify
            if "surge" in msg.lower() or "volume" in msg.lower():
                type_badge = badge("Vol spike", "badge-amber")
                bar_color  = "var(--amber)"; bar_w = 75
            elif "crash" in msg.lower():
                type_badge = badge("Sentiment drop", "badge-red")
                bar_color  = "var(--red)"; bar_w = 30
            else:
                type_badge = badge("Alert", "badge-blue")
                bar_color  = "var(--accent)"; bar_w = 55
            html += f"""
            <div class="ua-row">
              <div class="ua-top">
                <div class="ua-ticker">{tk}</div>{type_badge}
              </div>
              <div class="ua-vol-bar-wrap"><div class="ua-vol-bar-fill" style="width:{bar_w}%;background:{bar_color}"></div></div>
              <div class="ua-detail"><span>{msg[:60]}</span></div>
            </div>"""
        return html

    # ─────────────────────────────────────────────────────────
    # Build sector rows HTML
    # ─────────────────────────────────────────────────────────
    def sector_signal_to_pct(status_str):
        s = status_str.upper()
        if any(w in s for w in ["BULLISH","RECOVERING","BENEFITS","POSITIVE","RELIEF"]):
            return 70, "var(--green)", "badge-green", "Bullish"
        elif any(w in s for w in ["NEUTRAL","MIXED","STABLE","VOLATILE"]):
            return 55, "var(--amber)", "badge-amber", "Neutral"
        else:
            return 35, "var(--red)", "badge-red", "Bearish"

    def build_sector_rows():
        html = ""
        for name, icon, status, reason in sectors[:7]:
            pct, color, badge_cls, label = sector_signal_to_pct(status)
            dot_cls = "dot-green" if "green" in badge_cls else "dot-red" if "red" in badge_cls else "dot-amber"
            pct_cls = "up" if pct >= 65 else "down" if pct <= 40 else "flat"
            html += f"""
            <div class="sector-row">
              <div class="dot {dot_cls}"></div>
              <div class="sector-name">{name}</div>
              <div class="sector-bar-wrap"><div class="sector-bar-fill" style="width:{pct}%;background:{color}"></div></div>
              <div class="sector-pct {pct_cls}">{pct}%</div>
              <div class="sector-lbl badge {badge_cls}">{label}</div>
            </div>"""
        return html

    # ─────────────────────────────────────────────────────────
    # Build High-Trust rows
    # ─────────────────────────────────────────────────────────
    def build_ht_rows():
        if top_signals.empty:
            return '<div style="font-size:12px;color:var(--fg3);padding:8px 0">Insufficient history — check back after more trading days</div>'
        html = ""
        for i, (_, row) in enumerate(top_signals.head(3).iterrows()):
            tk       = row["ticker"].replace(".NS", "")
            trust    = float(row.get("trust_score", 70))
            win_rate = float(row.get("win_rate_10d", 0))
            combined = float(row.get("combined_acc", 0))
            ens_prob = float(row.get("ensemble_probability", 65))
            streak   = int(row.get("pos_day_streak", 0))
            ss       = float(row.get("smart_score", 50))
            html += f"""
            <div class="ht-item">
              <div style="display:flex;align-items:center">
                <span class="ht-ticker">{tk}</span>
                <span class="badge badge-green" style="margin-left:8px;font-size:10px;font-weight:700;padding:3px 8px;border-radius:5px">UP</span>
              </div>
              <div class="ht-stat-grid">
                <div class="ht-stat"><div class="val up">{ens_prob:.1f}%</div><div class="lbl">Confidence</div></div>
                <div class="ht-stat"><div class="val">{combined:.1f}%</div><div class="lbl">Historical acc</div></div>
                <div class="ht-stat"><div class="val up">{win_rate:.0f}%</div><div class="lbl">Win rate 10d</div></div>
                <div class="ht-stat"><div class="val">{ss:.0f}</div><div class="lbl">Smart score</div></div>
              </div>
            </div>"""
        return html

    # ─────────────────────────────────────────────────────────
    # Build Weak Signal rows
    # ─────────────────────────────────────────────────────────
    def build_weak_rows():
        if avoid_df.empty:
            return '<div style="font-size:12px;color:var(--fg3);padding:8px 0">No strong negative signals today</div>'
        html = ""
        for _, row in avoid_df.head(4).iterrows():
            tk     = row["ticker"].replace(".NS", "")
            ss     = float(row.get("smart_score", 30))
            reason = get_avoid_reason(row["ticker"], ss)
            ep     = float(row.get("ensemble_probability", 40)) if pd.notna(row.get("ensemble_probability")) else 40
            d_cls  = "dir-down" if ep < 45 else "badge-amber"
            d_lbl  = "DOWN" if ep < 45 else "FLAT"
            html += f"""
            <div class="weak-row">
              <div class="weak-ticker">{tk}</div>
              <div class="weak-info">
                <div class="weak-pct flat">{ep:.1f}%</div>
                <div class="weak-reason">{reason}</div>
              </div>
              <div class="signal-dir {'dir-down' if ep < 45 else 'badge-amber'}">{d_lbl}</div>
            </div>"""
        return html

    # ─────────────────────────────────────────────────────────
    # Build Watch Tomorrow rows
    # ─────────────────────────────────────────────────────────
    def build_watch_rows():
        if not tomorrow:
            return '<div style="font-size:12px;color:var(--fg3);padding:8px 0">No major events flagged</div>'
        html = ""
        tag_map = {
            "✅": ("Positive", "badge-green"),
            "⚠️": ("Watch",   "badge-amber"),
            "🔴": ("Risk",    "badge-red"),
            "🟢": ("Positive","badge-green"),
            "🏛️": ("Macro",   "badge-blue"),
        }
        for icon, event, impl in tomorrow[:4]:
            tag_lbl, tag_cls = tag_map.get(icon, ("Event", "badge-blue"))
            html += f"""
            <div class="watch-row">
              <div class="watch-ticker" style="font-size:12px">{event[:28]}</div>
              <div class="watch-reason">{impl[:40]}</div>
              <span class="watch-tag badge {tag_cls}">{tag_lbl}</span>
            </div>"""
        return html

    # ─────────────────────────────────────────────────────────
    # News volume stats
    # ─────────────────────────────────────────────────────────
    total_arts = int(summary["total"].sum()) if not summary.empty and "total" in summary.columns else 0
    pos_arts   = int(summary["pos"].sum())   if not summary.empty and "pos"   in summary.columns else 0
    neg_arts   = int(summary["neg"].sum())   if not summary.empty and "neg"   in summary.columns else 0

    # ─────────────────────────────────────────────────────────
    # Live data rows
    # ─────────────────────────────────────────────────────────
    def macro_dot(key, positive_below=None, negative_above=None, positive_above=None):
        val = macro.get(key, {}).get("value", 0)
        chg = macro.get(key, {}).get("change", 0)
        if positive_below and val < positive_below:    return "dot-green"
        if negative_above and val > negative_above:    return "dot-red"
        if positive_above and chg > 0:                 return "dot-green"
        if chg > 0.3:                                  return "dot-red"
        if chg < -0.3:                                 return "dot-green"
        return "dot-amber"

    def chg_cls(v): return "up" if v > 0 else "down" if v < 0 else "flat"

    live_rows = ""
    uv  = macro.get("us_vix",    {}); gld = macro.get("gold",      {})
    oil = macro.get("crude_oil", {}); ur  = macro.get("usd_inr",   {})
    u10 = macro.get("us_10y",    {}); sp  = macro.get("sp500",     {})
    nf  = macro.get("nifty50",   {}); iv  = macro.get("india_vix", {})

    if uv:
        d = "dot-red" if uv.get("value",15) > 20 else "dot-amber" if uv.get("value",15) > 15 else "dot-green"
        live_rows += dot_row(d, "US VIX", f"{uv.get('value',0):.1f}", f"{uv.get('change',0):+.1f}%", chg_cls(-uv.get("change",0)))
    if gld:
        live_rows += dot_row("dot-green" if gld.get("change",0)>0 else "dot-amber", "Gold", f"${gld.get('value',0):,.0f}", f"{gld.get('change',0):+.1f}%", chg_cls(gld.get("change",0)))
    if oil:
        d = "dot-red" if oil.get("change",0)>2 else "dot-green" if oil.get("change",0)<-1 else "dot-amber"
        live_rows += dot_row(d, "Brent Crude", f"${oil.get('value',0):.1f}", f"{oil.get('change',0):+.1f}%", chg_cls(-oil.get("change",0)))
    if ur:
        d = "dot-red" if ur.get("change",0)>0.3 else "dot-green"
        live_rows += dot_row(d, "USD/INR", f"{ur.get('value',0):.2f}", f"{ur.get('change',0):+.2f}%", chg_cls(-ur.get("change",0)))
    if u10:
        d = "dot-red" if u10.get("value",4)>4.8 else "dot-amber" if u10.get("change",0)>0.2 else "dot-green"
        live_rows += dot_row(d, "US 10Y Yield", f"{u10.get('value',0):.2f}%", f"{u10.get('change',0):+.2f}%", chg_cls(-u10.get("change",0)))
    fii_d = "dot-red" if fii_latest < -2000 else "dot-amber" if fii_latest < 0 else "dot-green"
    live_rows += dot_row(fii_d, "FII Net", f"₹{fii_latest:+,.0f}cr", "Selling" if fii_latest<0 else "Buying", "down" if fii_latest<0 else "up")
    dii_d = "dot-green" if dii_latest > 1000 else "dot-amber"
    live_rows += dot_row(dii_d, "DII Net", f"₹{dii_latest:+,.0f}cr", "Buying" if dii_latest>0 else "Selling", "up" if dii_latest>0 else "down")
    if nf:
        d = "dot-green" if nf.get("change",0)>0.3 else "dot-red" if nf.get("change",0)<-0.5 else "dot-amber"
        live_rows += dot_row(d, "Nifty 50", f"{nf.get('value',0):,.0f}", f"{nf.get('change',0):+.1f}%", chg_cls(nf.get("change",0)))
    if iv:
        d = "dot-red" if iv.get("value",15)>18 else "dot-green" if iv.get("value",15)<14 else "dot-amber"
        live_rows += dot_row(d, "India VIX", f"{iv.get('value',0):.1f}", f"{iv.get('change',0):+.1f}%", chg_cls(-iv.get("change",0)))

    # ─────────────────────────────────────────────────────────
    # PCR color logic
    # ─────────────────────────────────────────────────────────
    pcr_color  = "var(--amber)" if pcr_latest >= 1.1 else "var(--red)" if pcr_latest >= 1.3 else "var(--green)"
    pcr_status = "OI-based · bearish" if pcr_latest >= 1.1 else "OI-based · neutral" if pcr_latest >= 0.8 else "OI-based · bullish"
    pcr_badge  = "badge-red" if pcr_latest >= 1.3 else "badge-amber" if pcr_latest >= 1.1 else "badge-green"
    pcr_zscore_cls = "down" if pcr_zscore > 1 else "up" if pcr_zscore < -1 else "flat"

    # ─────────────────────────────────────────────────────────
    # Suppression pill for headwind card
    # ─────────────────────────────────────────────────────────
    # Count from ens_signals if available
    if not ens_signals.empty:
        if "up_probability" in ens_signals.columns:
            xgb_suppressed = int((ens_signals["up_probability"] < 65).sum())
        if "ensemble_probability" in ens_signals.columns:
            ens_suppressed = int((ens_signals["ensemble_probability"] < 65).sum())

    # ─────────────────────────────────────────────────────────
    # Render risk banner first (top of tab)
    # ─────────────────────────────────────────────────────────
    stance = "Reduce size · High trust only" if risk_score >= 50 else "Normal sizing · Monitor" if risk_score >= 25 else "Full sizing · All signals"
    st.html(f"""
    <div class="mib">
    <div class="risk-footer">
      <div>
        <div class="risk-footer-text">
          <h3>⚠️ MACRO RISK · {risk_label} HEADWIND — Score {risk_score}/100</h3>
          <p>{" · ".join(risk_reasons[:4]) if risk_reasons else "No major risk factors active"}</p>
        </div>
      </div>
      <div class="risk-pills">
        <span class="rpill rp-score">Score {risk_score}/100</span>
        {'<span class="rpill rp-xgb">XGB −' + str(xgb_suppressed) + '</span>' if xgb_suppressed else ''}
        {'<span class="rpill rp-ens">Ens −' + str(ens_suppressed) + '</span>' if ens_suppressed else ''}
        <span class="rpill rp-stance">{stance}</span>
      </div>
    </div>
    </div>
    """)

    # ─────────────────────────────────────────────────────────
    # ROW 1 — Headwind Score | Live Data | Top Signals Today
    # ─────────────────────────────────────────────────────────
    sig_badge_count = len(ens_top5) if not ens_top5.empty else (len(top_signals) if not top_signals.empty else 0)
    st.html(f"""
    <div class="mib">
    <div class="row3">

      <!-- Headwind Score -->
      <div class="card">
        {card_head("Headwind Score", badge(risk_label + " RISK", risk_badge))}
        <div class="card-body">
          <div class="score-hero">
            <div class="score-big" style="color:{risk_color}">{risk_score}</div>
            <div class="score-label" style="color:{risk_color}">{risk_label}</div>
            <div class="score-bar-wrap"><div class="score-bar-fill"></div></div>
            <div class="score-pointer"><div class="score-arrow" style="left:{score_pct}%"></div></div>
            <div class="score-sub-grid">
              <div class="score-sub">
                <div class="val" style="color:var(--red)">{xgb_suppressed}</div>
                <div class="lbl">XGB suppressed</div>
              </div>
              <div class="score-sub">
                <div class="val" style="color:var(--red)">{ens_suppressed}</div>
                <div class="lbl">Ens suppressed</div>
              </div>
            </div>
          </div>
          {"".join(dot_row(
            "dot-red" if "VIX" in r or "selling" in r.lower() or "surge" in r.lower() else "dot-amber",
            r, "", "", "flat"
          ) for r in risk_reasons[:4]) if risk_reasons else dot_row("dot-green","No major risk factors","","","flat")}
        </div>
      </div>

      <!-- Live Data -->
      <div class="card">
        {card_head("Live Data", badge("LIVE", "badge-blue"))}
        <div class="card-body">{live_rows}</div>
      </div>

      <!-- Top Signals Today -->
      <div class="card">
        {card_head("Top Signals Today", badge(str(sig_badge_count) + " SIGNALS", "badge-green"))}
        <div class="card-body">
          {build_signal_rows()}
          <div class="sub-caption">XGBoost ensemble · threshold ≥ 65% · macro filter {"active" if risk_score >= 50 else "off"}</div>
        </div>
      </div>

    </div>
    </div>
    """)

    # ─────────────────────────────────────────────────────────
    # ROW 2 — Unusual Activity | Sector Signals | PCR
    # ─────────────────────────────────────────────────────────
    ua_badge_count = len(alerts)
    st.html(f"""
    <div class="mib">
    <div class="row3">

      <!-- Unusual Activity -->
      <div class="card">
        {card_head("Unusual Activity", badge(str(ua_badge_count) + " ALERTS" if ua_badge_count else "CLEAR", "badge-amber" if ua_badge_count else "badge-green"))}
        <div class="card-body">{build_ua_rows()}</div>
      </div>

      <!-- Sector Signals -->
      <div class="card">
        {card_head("Sector Signals", badge(regime_label, "badge-red" if regime_label == "RISK-OFF" else "badge-amber" if regime_label in ["HIGH VOLATILITY","NEUTRAL"] else "badge-green"))}
        <div class="card-body">{build_sector_rows()}</div>
      </div>

      <!-- PCR -->
      <div class="card">
        {card_head("Put / Call Ratio", badge(pcr_regime, pcr_badge))}
        <div class="card-body">
          <div class="pcr-main-val" style="color:{pcr_color}">{pcr_latest:.3f}</div>
          <div class="pcr-sub">{pcr_status}</div>
          <div class="pcr-mini-row"><span class="pcr-mini-label">Volume PCR</span><span class="pcr-mini-val">{pcr_vol_latest:.3f}</span></div>
          <div class="pcr-mini-row"><span class="pcr-mini-label">5d Avg</span><span class="pcr-mini-val">{pcr_5d_avg:.3f}</span></div>
          <div class="pcr-mini-row"><span class="pcr-mini-label">Z-score</span><span class="pcr-mini-val {pcr_zscore_cls}">{pcr_zscore:+.2f}σ</span></div>
          <div class="pcr-mini-row"><span class="pcr-mini-label">Change</span><span class="pcr-mini-val {'down' if pcr_change>0 else 'up'}">{pcr_change:+.3f}</span></div>
          <div class="pcr-mini-row"><span class="pcr-mini-label">Regime</span><span class="pcr-mini-val {'down' if pcr_regime=='STRESS' else 'up'}">{pcr_regime}</span></div>
          <div class="pcr-mini-row"><span class="pcr-mini-label">As of</span><span class="pcr-mini-val flat">{pcr_date}</span></div>
          <div class="sub-caption">🔴 PCR ≥ 1.2 bearish · ⚪ 0.8-1.2 neutral · 🟢 ≤ 0.7 contrarian buy</div>
        </div>
      </div>

    </div>
    </div>
    """)

    # ─────────────────────────────────────────────────────────
    # ROW 3 — High-Trust | Weak Signals | Geo + Watch Tomorrow
    # ─────────────────────────────────────────────────────────
    ht_count = min(len(top_signals), 3) if not top_signals.empty else 0
    st.html(f"""
    <div class="mib">
    <div class="row3">

      <!-- High-Trust Signals -->
      <div class="card">
        {card_head("High-Trust Signals", badge(str(ht_count) + " STOCKS", "badge-green"))}
        <div class="card-body">
          {build_ht_rows()}
          <div class="sub-caption">Trust = 40% hist. acc + 35% win rate + 25% signal</div>
        </div>
      </div>

      <!-- Weak Signals + News Volume -->
      <div class="card">
        {card_head("Weak Signals", badge("WATCH", "badge-amber"))}
        <div class="card-body">
          {build_weak_rows()}
          <div class="news-mini-grid">
            <div class="news-mini"><div class="val">{total_arts}</div><div class="lbl">Total articles</div></div>
            <div class="news-mini"><div class="val up">{pos_arts}</div><div class="lbl">Positive</div></div>
            <div class="news-mini"><div class="val down">{neg_arts}</div><div class="lbl">Negative</div></div>
            <div class="news-mini"><div class="val flat">{total_arts - pos_arts - neg_arts}</div><div class="lbl">Neutral</div></div>
          </div>
        </div>
      </div>

      <!-- Geo + Watch Tomorrow -->
      <div class="card">
        {card_head("Geo Signals · Watch Tomorrow", badge("CAUTION", "badge-red"))}
        <div class="card-body">
          <div class="geo-section">
            <div class="section-title">Global flows</div>
            {dot_row(iran_dot,  "Iran / Hormuz",      iran_level,  str(iran_count)+" art")}
            {dot_row(fed_dot,   "US Fed",             fed_level,   str(fed_count)+" art")}
            {dot_row(jap_dot,   "US 10yr / Japan",    jap_level,   f"{u10v:.2f}%")}
            {dot_row(gold_dot,  "Gold / De-dollar",   gold_level,  f"{gchg:+.1f}%", chg_cls(gchg))}
          </div>
          <div class="geo-section">
            <div class="section-title">Watch tomorrow</div>
            {build_watch_rows()}
          </div>
        </div>
      </div>

    </div>
    </div>
    """)

    # ─────────────────────────────────────────────────────────
    # ROW 4 — SmartScore Leaderboard (left) | Chart (right)
    # ─────────────────────────────────────────────────────────

    # Prepare data
    _ss_cols = ["ticker","smart_score","S_recency","S_events","S_breadth","S_volume","pos","neg","total"]
    _ss_show = [c for c in _ss_cols if c in summary.columns]
    _ss_df   = summary[_ss_show].sort_values("smart_score", ascending=False).reset_index(drop=True) if not summary.empty else pd.DataFrame()

    # Inject extra CSS for the leaderboard section
    st.html("""
    <style>
    /* ── SmartScore section ── */
    .ss-section{display:grid;grid-template-columns:55% 1fr;gap:14px;margin-bottom:14px}
    @media(max-width:800px){.ss-section{grid-template-columns:1fr}}
    .ss-divider{border:none;border-top:1px solid var(--border);margin:6px 0 14px}

    /* leaderboard table */
    .ss-table{width:100%;border-collapse:collapse}
    .ss-th{font-size:12px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;color:var(--fg3);padding:7px 9px;text-align:left;border-bottom:2px solid var(--border)}
    .ss-th.right{text-align:right}
    .ss-tr:nth-child(even){background:var(--surface2)}
    .ss-tr:hover{background:var(--blue-bg)}
    .ss-td{padding:8px 9px;font-size:14px;border-bottom:1px solid var(--border)}
    .ss-td.mono{font-family:var(--font-mono);font-size:14px}
    .ss-td.right{text-align:right}
    .ss-rank{font-family:var(--font-mono);font-size:12px;color:var(--fg3);width:24px}
    .ss-bar-wrap{width:60px;height:5px;background:var(--border);border-radius:3px;overflow:hidden;display:inline-block;vertical-align:middle;margin-right:6px}
    .ss-bar-fill{height:100%;border-radius:3px}
    .ss-score-val{font-family:var(--font-mono);font-size:14px;font-weight:600}
    .ss-show-more{display:block;text-align:center;font-size:13px;color:var(--accent);padding:8px;cursor:pointer;background:none;border:none;width:100%;border-top:1px solid var(--border)}
    .ss-show-more:hover{background:var(--surface2)}
    .ss-hidden{display:none}
    .ss-section-label{font-size:12px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;color:var(--fg3);margin-bottom:10px}

    /* signal dir badge */
    .ss-dir{font-size:12px;font-weight:700;padding:2px 7px;border-radius:3px;display:inline-block}
    .ss-dir-up{background:var(--green-bg);color:var(--green)}
    .ss-dir-down{background:var(--red-bg);color:var(--red)}
    .ss-dir-none{background:var(--surface2);color:var(--fg3)}
    </style>
    """)

    # Build leaderboard rows HTML
    def build_ss_table(df, show_n=10):
        if df.empty:
            return '<div style="font-size:12px;color:var(--fg3);padding:12px">No SmartScore data available</div>'

        # Merge with ens_signals for direction
        sig_map = {}
        if not ens_signals.empty and "ticker" in ens_signals.columns:
            for _, r in ens_signals.iterrows():
                tk = r["ticker"]
                sig = str(r.get("signal","")).upper()
                prob = float(r.get("ensemble_probability", 0))
                if prob >= 65:
                    sig_map[tk] = ("UP" if "UP" in sig or "BULL" in sig else "DOWN", prob)

        rows_html = ""
        for i, row in df.head(show_n).iterrows():
            tk   = row["ticker"].replace(".NS","") if "ticker" in row else "—"
            ss   = float(row.get("smart_score", 0))
            pos  = int(row.get("pos", 0))
            neg  = int(row.get("neg", 0))
            tot  = int(row.get("total", 0))
            bar_w = min(int(ss), 100)
            bar_color = "var(--green)" if ss >= 65 else "var(--amber)" if ss >= 45 else "var(--red)"
            rank = i + 1
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"

            # direction from ens_signals
            tk_ns = row["ticker"] if "ticker" in row else ""
            if tk_ns in sig_map:
                d, prob = sig_map[tk_ns]
                dir_badge = f'<span class="ss-dir ss-dir-{"up" if d=="UP" else "down"}">{d} {prob:.0f}%</span>'
            else:
                dir_badge = '<span class="ss-dir ss-dir-none">—</span>'

            rows_html += f"""
            <tr class="ss-tr">
              <td class="ss-td ss-rank">{medal}</td>
              <td class="ss-td mono">{tk}</td>
              <td class="ss-td right">
                <div style="display:flex;align-items:center;justify-content:flex-end;gap:6px">
                  <span class="ss-bar-wrap"><span class="ss-bar-fill" style="width:{bar_w}%;background:{bar_color}"></span></span>
                  <span class="ss-score-val" style="color:{bar_color}">{ss:.0f}</span>
                </div>
              </td>
              <td class="ss-td right">{dir_badge}</td>
              <td class="ss-td right mono" style="color:var(--green)">{pos}</td>
              <td class="ss-td right mono" style="color:var(--red)">{neg}</td>
              <td class="ss-td right mono" style="color:var(--fg3)">{tot}</td>
            </tr>"""

        # Extra rows (hidden by default, toggled by JS)
        extra_html = ""
        for i, row in df.iloc[show_n:].iterrows():
            tk   = row["ticker"].replace(".NS","") if "ticker" in row else "—"
            ss   = float(row.get("smart_score", 0))
            pos  = int(row.get("pos", 0))
            neg  = int(row.get("neg", 0))
            tot  = int(row.get("total", 0))
            bar_w = min(int(ss), 100)
            bar_color = "var(--green)" if ss >= 65 else "var(--amber)" if ss >= 45 else "var(--red)"
            rank = i + 1
            tk_ns = row["ticker"] if "ticker" in row else ""
            if tk_ns in sig_map:
                d, prob = sig_map[tk_ns]
                dir_badge = f'<span class="ss-dir ss-dir-{"up" if d=="UP" else "down"}">{d} {prob:.0f}%</span>'
            else:
                dir_badge = '<span class="ss-dir ss-dir-none">—</span>'
            extra_html += f"""
            <tr class="ss-tr ss-hidden" id="ss-extra-row">
              <td class="ss-td ss-rank">#{rank}</td>
              <td class="ss-td mono">{tk}</td>
              <td class="ss-td right">
                <div style="display:flex;align-items:center;justify-content:flex-end;gap:6px">
                  <span class="ss-bar-wrap"><span class="ss-bar-fill" style="width:{bar_w}%;background:{bar_color}"></span></span>
                  <span class="ss-score-val" style="color:{bar_color}">{ss:.0f}</span>
                </div>
              </td>
              <td class="ss-td right">{dir_badge}</td>
              <td class="ss-td right mono" style="color:var(--green)">{pos}</td>
              <td class="ss-td right mono" style="color:var(--red)">{neg}</td>
              <td class="ss-td right mono" style="color:var(--fg3)">{tot}</td>
            </tr>"""

        remaining = max(0, len(df) - show_n)
        toggle_btn = ""
        if remaining > 0:
            toggle_btn = f"""
            <tr><td colspan="7" style="padding:0">
              <button class="ss-show-more" onclick="
                var rows=document.querySelectorAll('#ss-extra-row');
                var showing=this.dataset.open==='1';
                rows.forEach(r=>r.style.display=showing?'none':'table-row');
                this.textContent=showing?'Show all {remaining} more ↓':'Show less ↑';
                this.dataset.open=showing?'0':'1';
              ">Show all {remaining} more ↓</button>
            </td></tr>"""

        return f"""
        <table class="ss-table">
          <thead>
            <tr>
              <th class="ss-th">#</th>
              <th class="ss-th">Ticker</th>
              <th class="ss-th right">SmartScore</th>
              <th class="ss-th right">Signal</th>
              <th class="ss-th right">Pos</th>
              <th class="ss-th right">Neg</th>
              <th class="ss-th right">Total</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
            {extra_html}
            {toggle_btn}
          </tbody>
        </table>"""

    # Build chart HTML (horizontal bars, top 15)
    def build_ss_chart_html(df, top_n=15):
        if df.empty:
            return '<div style="font-size:12px;color:var(--fg3);padding:12px">No data</div>'
        chart_df = df.head(top_n)
        max_ss = float(chart_df["smart_score"].max()) if not chart_df.empty else 100
        bars = ""
        for _, row in chart_df.iterrows():
            tk = row["ticker"].replace(".NS","")
            ss = float(row.get("smart_score", 0))
            bar_w = int((ss / max_ss) * 100) if max_ss > 0 else 0
            color = "var(--green)" if ss >= 65 else "var(--amber)" if ss >= 45 else "var(--red)"
            bars += f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
              <div style="width:72px;font-family:var(--font-mono);font-size:13px;color:var(--fg2);text-align:right;flex-shrink:0">{tk}</div>
              <div style="flex:1;background:var(--surface2);border-radius:3px;height:14px;overflow:hidden">
                <div style="width:{bar_w}%;background:{color};height:100%;border-radius:3px;transition:width .3s"></div>
              </div>
              <div style="width:28px;font-family:var(--font-mono);font-size:12px;font-weight:600;color:{color};text-align:right;flex-shrink:0">{ss:.0f}</div>
            </div>"""
        return bars

    # Render the split section
    top_n_ss = 10
    st.html(f"""
    <div class="mib">
      <hr class="ss-divider">
      <div class="ss-section">

        <!-- Left: Leaderboard -->
        <div class="card">
          {card_head("SmartScore Leaderboard", badge("TOP " + str(min(top_n_ss, len(_ss_df))), "badge-blue"))}
          <div class="card-body" style="padding:0">
            {build_ss_table(_ss_df, show_n=top_n_ss)}
          </div>
        </div>

        <!-- Right: Chart -->
        <div class="card">
          {card_head("SmartScore Chart", badge("TOP 15", "badge-blue"))}
          <div class="card-body">
            <div class="ss-section-label">Score out of 100 · news sentiment signal strength</div>
            {build_ss_chart_html(_ss_df, top_n=15)}
            <div class="sub-caption" style="margin-top:10px">
              🟢 ≥ 65 strong · 🟡 45–64 moderate · 🔴 &lt; 45 weak<br>
              SmartScore = 45% recency + 25% events + 20% breadth + 10% volume
            </div>
          </div>
        </div>

      </div>
    </div>
    """)

    # ─────────────────────────────────────────────────────────
    # Collapsed extras — About + Disclaimer only
    # ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ About SmartScore & Trust Score", expanded=False):
        st.markdown("""
        **SmartScore** (0-100):
        - **S_recency** (45%) — EWMA sentiment: 8h half-life for 1d, 36h for 3d
        - **S_events** (25%) — Event magnitude (earnings, litigation, order wins)
        - **S_breadth** (20%) — Positive vs negative ratio (Z-score)
        - **S_volume** (10%) — News volume signal (Z-score)

        **Trust Score** (0-100):
        - 40% historical accuracy | 35% win rate (10d) | 25% current signal

        🟢 Trust ≥ 75 = High confidence | 🟡 60–74 = Moderate | 🔴 < 60 = Low

        **What this system cannot see:** Fed/RBI exact dates, earnings calendar,
        options expiry, intraday patterns, block deals.
        """)

    with st.expander("⚠️ Research Tool Disclaimer", expanded=False):
        st.warning(
            "Research tool only — not financial advice. "
            "This dashboard aggregates public data (RSS news, yfinance, NSE FII/DII, NSE options). "
            "Macro risk scores do NOT include Fed/RBI meeting calendars or earnings dates. "
            "Always verify independently before any financial decision."
        )
with tab2:
    st.subheader("Predicted Next-Day Returns")

    if preds.empty:
        st.info("No predictions yet. Train a model and run predict_next.py.")
    else:
        mae_guess = 0.25
        preds["confidence"] = (preds["pred_ret_1d_pct"].abs() / mae_guess).clip(0, 2.0)

        c1, c2, c3 = st.columns(3)
        with c1:
            thr = st.number_input("Min predicted return (%)", value=0.20, step=0.05)
        with c2:
            min_rec = st.slider("Min S_recency", 0, 100, 60)
        with c3:
            min_events = st.slider("Min S_events", 0, 100, 55)

        merged = preds.merge(summary, on="ticker", how="left")
        filt = (
            (merged["pred_ret_1d_pct"] >= thr) &
            (merged["S_recency"] >= min_rec) &
            (merged["S_events"] >= min_events) &
            (merged["total"] >= 3) &
            (merged["S_breadth"] >= 50)
        )
        pick = merged[filt].sort_values("pred_ret_1d_pct", ascending=False)

        st.write(f"**Candidates meeting filters: {len(pick)}**")
        st.dataframe(
            pick[["ticker","pred_ret_1d_pct","confidence","smart_score","S_recency","S_events","S_breadth","S_volume","total"]],
            use_container_width=True, hide_index=True
        )

        colA, colB = st.columns(2)
        with colA:
            figp = px.bar(preds.sort_values("pred_ret_1d_pct", ascending=False).head(15),
                          x="ticker", y="pred_ret_1d_pct", title="Top Predicted Gainers (Next Day)")
            st.plotly_chart(figp, use_container_width=True)
        with colB:
            fign = px.bar(preds.sort_values("pred_ret_1d_pct").head(15),
                          x="ticker", y="pred_ret_1d_pct", title="Top Predicted Losers (Next Day)")
            st.plotly_chart(fign, use_container_width=True)

        st.caption("Note: Predictions are research signals, not guarantees. Use thresholds and breadth/event filters.")

# ================================ SIGNALS ============================
with tab3:
    st.subheader("XGBoost Signal Dashboard")
    st.caption("UP/DOWN probability signals from XGBoost Classifier — updated after each daily run")

    if not signals.empty and os.path.exists(os.path.join(BASE_DIR, "data/signals_3d.csv")):
        sig3d_top = load_csv(os.path.join(BASE_DIR, "data/signals_3d.csv"))
        both_agree_count = len(sig3d_top[sig3d_top.get("combined_signal", "") == "🟢 STRONG — Both 1d & 3d agree"]) if "combined_signal" in sig3d_top.columns else 0
        strong_1d = len(signals[signals["signal"] == "🟢 STRONG BULLISH"])

        if both_agree_count > 0 or strong_1d > 0:
            st.success(f"🎯 Today: **{both_agree_count} stocks** where both 1d & 3d agree (highest confidence) | **{strong_1d} STRONG BULLISH** 1-day signals")
        else:
            st.warning("⚪ No high-confidence signals today — models uncertain about near-term direction")

    if signals.empty:
        st.info("No XGBoost signals yet. Run the weekly training first to generate signals.")
    else:
        sig_merged = signals.merge(summary, on="ticker", how="left")

        c1, c2 = st.columns(2)
        with c1:
            min_prob = st.slider("Min UP probability (%)", 0, 100, 55)
        with c2:
            signal_filter = st.multiselect(
                "Filter by signal",
                options=["🟢 STRONG BULLISH", "🟡 MILD BULLISH", "⚪ NEUTRAL",
                         "🟠 MILD BEARISH", "🔴 STRONG BEARISH"],
                default=["🟢 STRONG BULLISH", "🟡 MILD BULLISH"]
            )

        filtered = sig_merged[
            (sig_merged["up_probability"] >= min_prob) &
            (sig_merged["signal"].isin(signal_filter))
        ].sort_values("up_probability", ascending=False)

        st.write(f"**Stocks meeting criteria: {len(filtered)}**")

        display_cols = ["ticker", "signal", "confidence", "up_probability",
                        "smart_score", "S_recency", "S_events"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            top_bull = signals.nlargest(15, "up_probability")
            fig_bull = px.bar(
                top_bull, x="ticker", y="up_probability",
                color="up_probability",
                color_continuous_scale="Greens",
                title="Top Bullish Signals (% probability UP)"
            )
            fig_bull.add_hline(y=50, line_dash="dash", line_color="gray",
                               annotation_text="50% baseline")
            st.plotly_chart(fig_bull, use_container_width=True)

        with col2:
            top_bear = signals.nsmallest(15, "up_probability")
            fig_bear = px.bar(
                top_bear, x="ticker", y="up_probability",
                color="up_probability",
                color_continuous_scale="Reds_r",
                title="Top Bearish Signals (% probability UP)"
            )
            fig_bear.add_hline(y=50, line_dash="dash", line_color="gray",
                               annotation_text="50% baseline")
            st.plotly_chart(fig_bear, use_container_width=True)

        st.caption("""
        ⚠️ Signals are research indicators only — not financial advice.
        High confidence signals (>65% or <35%) are more meaningful than neutral ones.
        Use alongside SmartScore and your own research before any decision.
        """)

        st.markdown("---")
        st.subheader("3-Day Horizon Signals")
        st.caption("Predicts direction over next 3 trading days — news sentiment takes 2-3 days to fully price in")

        signals_3d_path = os.path.join(BASE_DIR, "data/signals_3d.csv")
        if os.path.exists(signals_3d_path):
            sig3d = load_csv(signals_3d_path)

            if not sig3d.empty:
                both_agree = sig3d[sig3d["combined_signal"] == "🟢 STRONG — Both 1d & 3d agree"] if "combined_signal" in sig3d.columns else pd.DataFrame()
                bullish_3d = sig3d[sig3d["ensemble_3d_prob"] > 55] if "ensemble_3d_prob" in sig3d.columns else pd.DataFrame()

                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 Both 1d+3d Agree", len(both_agree))
                c2.metric("🔵 3d Bullish Only", len(bullish_3d) - len(both_agree))
                c3.metric("Total Stocks", len(sig3d))

                st.info("🟢 **STRONG** = Both 1-day AND 3-day models agree → highest confidence signal")

                min_3d_prob = st.slider("Min 3-Day Ensemble Probability (%)", 50, 80, 55, key="slider_3d")

                if "ensemble_3d_prob" in sig3d.columns:
                    filtered_3d = sig3d[sig3d["ensemble_3d_prob"] >= min_3d_prob].copy()
                    filtered_3d = filtered_3d.sort_values("ensemble_3d_prob", ascending=False)

                    display_cols_3d = ["ticker", "xgb_3d_prob", "ensemble_3d_prob",
                                       "ens_1d_prob", "combined_signal"]
                    display_cols_3d = [c for c in display_cols_3d if c in filtered_3d.columns]

                    st.dataframe(
                        filtered_3d[display_cols_3d].rename(columns={
                            "xgb_3d_prob":     "XGBoost 3d (%)",
                            "ensemble_3d_prob": "Ensemble 3d (%)",
                            "ens_1d_prob":      "Ensemble 1d (%)",
                            "combined_signal":  "Signal",
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    if not filtered_3d.empty:
                        fig_3d = px.bar(
                            filtered_3d.head(20),
                            x="ticker",
                            y="ensemble_3d_prob",
                            color="combined_signal",
                            color_discrete_map={
                                "🟢 STRONG — Both 1d & 3d agree": "#2ecc71",
                                "🔵 3d only bullish":              "#3498db",
                                "🟡 1d only bullish":              "#f1c40f",
                                "⚪ NEUTRAL":                      "#7f8c8d",
                            },
                            title="3-Day Ensemble Probability by Stock",
                            labels={"ensemble_3d_prob": "3-Day UP Probability (%)"}
                        )
                        fig_3d.add_hline(y=55, line_dash="dash",
                                         annotation_text="55% threshold")
                        st.plotly_chart(fig_3d, use_container_width=True)

                st.markdown("""
                **How to use 3-day signals:**
                - 🟢 **Both agree**: Highest confidence — act on this
                - 🔵 **3d only bullish**: Buy today, hold 3 days
                - 🟡 **1d only bullish**: Quick trade only
                - ⚪ **NEUTRAL**: Skip this stock
                """)
        else:
            st.info("3-day signals will appear after the next pipeline run.")

with tab4:
    st.subheader("Stock Signal Accuracy")
    st.caption("Track which stocks our model predicts most reliably — updated daily")

    accuracy_path = os.path.join(BASE_DIR, "data/stock_accuracy.csv")
    history_path  = os.path.join(BASE_DIR, "data/signal_history.csv")
    streaks_path  = os.path.join(BASE_DIR, "data/stock_streaks.csv")

    if os.path.exists(accuracy_path):
        acc_df = load_csv(accuracy_path)

        if not acc_df.empty:

            c1, c2, c3, c4 = st.columns(4)
            trusted  = acc_df[acc_df["combined_trust"] == "✅ TRUST"]
            moderate = acc_df[acc_df["combined_trust"] == "🟡 MODERATE"]
            weak     = acc_df[acc_df["combined_trust"] == "❌ WEAK"]
            c1.metric("Total Stocks Tracked", len(acc_df))
            c2.metric("✅ Trust", len(trusted))
            c3.metric("🟡 Moderate", len(moderate))
            c4.metric("❌ Weak", len(weak))

            st.markdown("---")

            st.markdown("### Accuracy by Signal Type")
            col1, col2 = st.columns(2)
            with col1:
                fig_ss = px.bar(
                    acc_df.dropna(subset=["ss_acc_overall"]).head(20),
                    x="ticker", y="ss_acc_overall",
                    color="ss_trust",
                    color_discrete_map={
                        "✅ TRUST": "#2ecc71",
                        "🟡 MODERATE": "#f1c40f",
                        "❌ WEAK": "#e74c3c"
                    },
                    title="SmartScore Signal Accuracy (%)"
                )
                fig_ss.add_hline(y=50, line_dash="dash", annotation_text="50% baseline")
                st.plotly_chart(fig_ss, use_container_width=True)

            with col2:
                fig_xgb = px.bar(
                    acc_df.dropna(subset=["xgb_acc_overall"]).head(20),
                    x="ticker", y="xgb_acc_overall",
                    color="xgb_trust",
                    color_discrete_map={
                        "✅ TRUST": "#2ecc71",
                        "🟡 MODERATE": "#f1c40f",
                        "❌ WEAK": "#e74c3c"
                    },
                    title="XGBoost Signal Accuracy (%)"
                )
                fig_xgb.add_hline(y=50, line_dash="dash", annotation_text="50% baseline")
                st.plotly_chart(fig_xgb, use_container_width=True)

            st.markdown("### Combined Signal Accuracy (Both Models Agree)")
            fig_combined = px.bar(
                acc_df.dropna(subset=["combined_acc"]).head(30),
                x="ticker", y="combined_acc",
                color="combined_trust",
                color_discrete_map={
                    "✅ TRUST": "#2ecc71",
                    "🟡 MODERATE": "#f1c40f",
                    "❌ WEAK": "#e74c3c"
                },
                title="Combined Signal Accuracy — When Both SmartScore AND XGBoost Agree"
            )
            fig_combined.add_hline(y=50, line_dash="dash", annotation_text="50% baseline")
            st.plotly_chart(fig_combined, use_container_width=True)

            st.markdown("---")

            st.markdown("### 📈 Performance Streaks")
            st.caption("Stocks on a roll right now — consecutive positive days and correct predictions")

            if os.path.exists(streaks_path):
                streaks = load_csv(streaks_path)
                if not streaks.empty and streaks["pos_day_streak"].max() > 0:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Positive Return Streak 🔥**")
                        st.caption("Most consecutive positive days recently")
                        top_pos = streaks[streaks["pos_day_streak"] > 0].head(10)
                        st.dataframe(
                            top_pos[["ticker","pos_day_streak","pos_days_10d"]].rename(columns={
                                "pos_day_streak": "Streak 🔥",
                                "pos_days_10d":   "Positive Days (10d)"
                            }),
                            use_container_width=True, hide_index=True
                        )
                    with c2:
                        st.markdown("**Model Correct Streak 🎯**")
                        st.caption("Stocks XGBoost has been correctly predicting")
                        top_correct = streaks[streaks["xgb_correct_streak"] > 0].head(10)
                        if not top_correct.empty:
                            st.dataframe(
                                top_correct[["ticker","xgb_correct_streak","win_rate_10d"]].rename(columns={
                                    "xgb_correct_streak": "Correct Streak 🎯",
                                    "win_rate_10d":        "Win Rate 10d (%)"
                                }),
                                use_container_width=True, hide_index=True
                            )
                        else:
                            st.caption("Prediction streaks appear after more signals settle")
                else:
                    st.caption("Streaks grow as more trading days accumulate — check back next week")
            else:
                st.caption("Streak data not available yet")

            st.markdown("---")

            st.markdown("### Full Accuracy Table")
            st.caption("Complete signal accuracy + streak data per stock")

            if os.path.exists(streaks_path):
                streaks_df = load_csv(streaks_path)
                if not streaks_df.empty:
                    acc_merged = acc_df.merge(
                        streaks_df[["ticker","pos_day_streak","xgb_correct_streak","win_rate_10d","pos_days_10d"]],
                        on="ticker", how="left"
                    )
                else:
                    acc_merged = acc_df.copy()
            else:
                acc_merged = acc_df.copy()

            display_cols = [
                "ticker","total_signals",
                "ss_acc_overall","ss_trust",
                "xgb_acc_overall","xgb_trust",
                "combined_acc","combined_trust",
                "best_signal",
                "pos_day_streak","xgb_correct_streak","win_rate_10d"
            ]
            display_cols = [c for c in display_cols if c in acc_merged.columns]
            st.dataframe(
                acc_merged[display_cols].rename(columns={
                    "pos_day_streak":      "Streak 🔥",
                    "xgb_correct_streak":  "Model Streak 🎯",
                    "win_rate_10d":        "Win Rate 10d"
                }),
                use_container_width=True, hide_index=True
            )

    else:
        if os.path.exists(history_path):
            hist_df = load_csv(history_path)
            st.info(f"""
            Signal history is being built automatically.
            **{len(hist_df)} predictions** saved so far.
            Accuracy data will appear after tomorrow's market close.
            Check back tomorrow!
            """)
        else:
            st.info("Signal history will start building from today's predictions.")

# ================================ DRILLDOWN ===========================
with tab5:
    st.markdown("## 🔍 Stock Intelligence")
    st.caption("Deep dive on any stock — sentiment + technical + fundamental + ML signals")

    tickers_list = sorted(summary["ticker"].unique().tolist())
    col_search, col_btn = st.columns([3, 1])
    with col_search:
        tk = st.selectbox("Search stock", tickers_list,
                          index=0,
                          format_func=lambda x: x.replace(".NS",""))
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_analysis = st.button("🔍 Analyse", type="primary")
        if run_analysis:
            with st.spinner(f"Fetching fundamentals for {tk.replace('.NS','')}..."):
                st.session_state["fund_tk"]   = tk
                st.session_state["fund_data"] = get_fundamental_signal(tk)

    if tk:
        row = summary[summary["ticker"] == tk]
        if row.empty:
            st.warning("No data for this stock")
        else:
            row = row.iloc[0]
            tk_clean = tk.replace(".NS","")

            # ── Conviction Matrix ──
            st.markdown(f"### {tk_clean} — Conviction Matrix")

            # Sentiment signal
            ss = float(row.get("smart_score", 50))
            if ss >= 65:   sent_sig, sent_col = "🟢 STRONG",   "green"
            elif ss >= 50: sent_sig, sent_col = "🟡 MODERATE", "orange"
            elif ss >= 35: sent_sig, sent_col = "⚪ NEUTRAL",  "gray"
            else:          sent_sig, sent_col = "🔴 WEAK",     "red"

            # Technical signal from signals CSV
            tech_sig, tech_col = "⚪ N/A", "gray"
            if not signals.empty:
                sig_row = signals[signals["ticker"] == tk]
                if not sig_row.empty:
                    prob = float(sig_row.iloc[0].get("up_probability", 50))
                    if prob >= 60:   tech_sig, tech_col = "🟢 BULLISH",  "green"
                    elif prob >= 50: tech_sig, tech_col = "🟡 MODERATE", "orange"
                    else:            tech_sig, tech_col = "🔴 BEARISH",  "red"

            # Macro signal
            regime_l, regime_i, _ = classify_market_regime(macro, fii_dii)
            if regime_l == "RISK-ON":      mac_sig, mac_col = "🟢 POSITIVE", "green"
            elif regime_l == "NEUTRAL":    mac_sig, mac_col = "🟡 NEUTRAL",  "orange"
            else:                          mac_sig, mac_col = "🔴 CAUTION",  "red"

            # PCR signal
            if pcr_latest >= 1.2:  pcr_sig, pcr_col = "🔴 BEARISH",  "red"
            elif pcr_latest <= 0.7: pcr_sig, pcr_col = "🟢 BULLISH",  "green"
            else:                   pcr_sig, pcr_col = "⚪ NEUTRAL",  "gray"

            # Count greens for conviction score
            _fd_conv = st.session_state.get("fund_data") if st.session_state.get("fund_tk") == tk else None
            fund_sig = _fd_conv["signal"] if _fd_conv else ""
            greens = sum([
                1 if "🟢" in sent_sig  else 0,
                1 if "🟢" in tech_sig  else 0,
                1 if "🟢" in mac_sig   else 0,
                1 if "🟢" in pcr_sig   else 0,
                1 if "🟢" in fund_sig  else 0,
            ])

            # Matrix display
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"**Sentiment**\n\n{sent_sig}")
                st.caption(f"SmartScore: {ss:.0f}/100")
            with m2:
                st.markdown(f"**Technical**\n\n{tech_sig}")
                st.caption("RSI + MACD + MA")
            with m3:
                st.markdown(f"**Macro**\n\n{mac_sig}")
                st.caption(f"Regime: {regime_l}")
            with m4:
                st.markdown(f"**PCR/Options**\n\n{pcr_sig}")
                st.caption(f"PCR: {pcr_latest:.3f}")
            with m5:
                fund_placeholder = "Loading..."
                _fd = st.session_state.get("fund_data") if st.session_state.get("fund_tk") == tk else None
                _fsig = _fd["signal"] if _fd else "⚪ Click Analyse"
                st.markdown(f"**Fundamental**\n\n{_fsig}")
                st.caption("P/E, ROE, Debt")

            if greens >= 5:   conv_label = "🟢 VERY HIGH CONVICTION"
            elif greens >= 4: conv_label = "🟡 HIGH CONVICTION"
            elif greens >= 3: conv_label = "🟠 MODERATE CONVICTION"
            else:             conv_label = "🔴 LOW CONVICTION"
            st.info(f"**Conviction: {conv_label}** ({greens}/5 factors aligned)")

            st.markdown("---")

            # ── ML Signals ──
            c1, c2, c3 = st.columns(3)
            with c1:
                if not signals.empty:
                    sig_row = signals[signals["ticker"] == tk]
                    if not sig_row.empty:
                        prob_1d = float(sig_row.iloc[0].get("up_probability", 50))
                        st.metric("1-Day ML Signal", f"{prob_1d:.1f}% UP")
            with c2:
                sig3d_path = os.path.join(BASE_DIR, "data/signals_3d.csv")
                if os.path.exists(sig3d_path):
                    sig3d = load_csv(sig3d_path)
                    if not sig3d.empty:
                        r3d = sig3d[sig3d["ticker"] == tk]
                        if not r3d.empty:
                            prob_3d = float(r3d.iloc[0].get("ensemble_3d_prob", 50))
                            st.metric("3-Day ML Signal", f"{prob_3d:.1f}% UP")
            with c3:
                acc_path = os.path.join(BASE_DIR, "data/stock_accuracy.csv")
                if os.path.exists(acc_path):
                    acc_df = load_csv(acc_path)
                    if not acc_df.empty:
                        acc_row = acc_df[acc_df["ticker"] == tk]
                        if not acc_row.empty:
                            trust = acc_row.iloc[0].get("combined_trust","N/A")
                            acc   = acc_row.iloc[0].get("combined_acc", 0)
                            st.metric("Historical Accuracy", f"{acc:.1f}%", delta=trust)

            st.markdown("---")

            # ── Sentiment Analysis ──
            st.markdown("#### 📰 Sentiment Analysis")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                total_arts = int(row.get("total", 0))
                pos_arts   = int(row.get("pos",   0))
                neg_arts   = int(row.get("neg",   0))
                st.metric("SmartScore",  f"{ss:.0f}/100")
                st.metric("Articles",    f"{pos_arts}✅ {neg_arts}❌ {total_arts} total")

            with sc2:
                # Sentiment velocity
                if not hist.empty:
                    tk_hist = hist[hist["ticker"] == tk].sort_values("date")
                    if len(tk_hist) >= 4:
                        score_now  = float(tk_hist["smart_score"].iloc[-1])
                        score_3ago = float(tk_hist["smart_score"].iloc[-4])
                        velocity   = score_now - score_3ago
                        v_icon = "📈" if velocity > 5 else "📉" if velocity < -5 else "➡️"
                        st.metric("Sentiment Velocity (3d)", f"{velocity:+.1f}",
                                  delta="Improving" if velocity > 5 else "Falling" if velocity < -5 else "Stable")
                    else:
                        st.caption("Need more history for velocity")

            with sc3:
                # News freshness
                if not hist.empty:
                    tk_hist = hist[hist["ticker"] == tk].sort_values("date")
                    if not tk_hist.empty:
                        last_date = pd.to_datetime(tk_hist["date"].iloc[-1])
                        days_ago  = (pd.Timestamp.now() - last_date).days
                        freshness = "🟢 FRESH" if days_ago == 0 else "🟡 RECENT" if days_ago <= 2 else "🔴 STALE"
                        st.metric("News Freshness", freshness)
                        st.caption(f"Last article: {days_ago} day(s) ago")

            # Sentiment history chart
            if not hist.empty:
                tk_hist = hist[hist["ticker"] == tk].sort_values("date").tail(30)
                if not tk_hist.empty and "smart_score" in tk_hist.columns:
                    fig_vel = px.line(tk_hist, x="date", y="smart_score",
                                      title=f"{tk_clean} — SmartScore Trend (30 days)",
                                      labels={"smart_score": "SmartScore", "date": ""})
                    fig_vel.add_hline(y=65, line_dash="dash", line_color="green",  annotation_text="Bullish zone")
                    fig_vel.add_hline(y=35, line_dash="dash", line_color="red",    annotation_text="Bearish zone")
                    fig_vel.update_layout(height=200, margin=dict(t=30,b=0,l=0,r=0))
                    st.plotly_chart(fig_vel, use_container_width=True)

            st.markdown("---")

            # ── Fundamentals (on button click) ──
            st.markdown("#### 📊 Fundamental Check")
            fund = st.session_state.get("fund_data") if st.session_state.get("fund_tk") == tk else None
            if fund:
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    if fund["pe"]:
                        st.metric("P/E Ratio",    f"{fund['pe']:.1f}")
                    if fund["forward_pe"]:
                        st.metric("Forward P/E",  f"{fund['forward_pe']:.1f}")
                    if fund["eps_growth"] is not None:
                        st.metric("EPS Growth",   f"{fund['eps_growth']*100:.1f}%")
                with fc2:
                    if fund["roe"] is not None:
                        st.metric("ROE",          f"{fund['roe']*100:.1f}%")
                    if fund["debt_equity"] is not None:
                        st.metric("Debt/Equity",  f"{fund['debt_equity']:.2f}")
                    if fund["pos_52w"] is not None:
                        st.metric("52W Position", f"{fund['pos_52w']:.0f}%",
                                  delta="Near low ✅" if fund["pos_52w"] < 30 else "Near high ⚠️" if fund["pos_52w"] > 80 else "Mid range")
                with fc3:
                    if fund["target"] and fund["current"]:
                        st.metric("Analyst Target", f"₹{fund['target']:,.0f}",
                                  delta=f"{fund['upside']:+.1f}% upside" if fund["upside"] else None)
                    if fund["recommendation"]:
                        st.metric("Consensus",    fund["recommendation"])
                    if fund["n_analysts"]:
                        st.metric("# Analysts",   str(fund["n_analysts"]))

                if fund["color"] == "success":
                    st.success(f"Fundamental Signal: {fund['signal']}")
                elif fund["color"] == "warning":
                    st.warning(f"Fundamental Signal: {fund['signal']}")
                elif fund["color"] == "error":
                    st.error(f"Fundamental Signal: {fund['signal']}")
                else:
                    st.info(f"Fundamental Signal: {fund['signal']}")
            else:
                st.caption("Click **Analyse** button above to load fundamentals (fetches live from yfinance)")

            st.caption("⚠️ Research tool only — not financial advice. Verify independently.")

# ================================ MODEL HEALTH =======================
with tab6:
    st.subheader("Model Health Dashboard")
    st.markdown("""
    This project uses **two types of models** working in parallel:

    **Regression Models** (Ridge, RandomForest) — predict the exact % return tomorrow.
    Evaluated by MAE, R², and Direction Accuracy.

    **Classification Models** (XGBoost, Voting Ensemble) — predict UP or DOWN probability.
    Evaluated by Direction Accuracy only (MAE/R²/Spearman not applicable).

    The **Voting Ensemble** (XGBoost + LightGBM + RandomForest Classifier) is the most
    reliable signal — when all 3 models agree, confidence is highest.
    """)

    st.markdown("---")
    metrics_path = os.path.join(BASE_DIR, "data/modeling/model_metrics.csv")

    if os.path.exists(metrics_path):
        metrics = load_csv(metrics_path)
        metrics["train_date"] = pd.to_datetime(metrics["train_date"], errors="coerce")

        if not metrics.empty:

            st.markdown("### Regression Models (Ridge / RandomForest)")
            st.caption("Predicts exact next-day % return. Lower MAE = better.")

            reg_metrics = metrics[metrics["model"].isin(["Ridge", "RandomForest"])]
            best_rows = reg_metrics[reg_metrics["is_best"] == True]
            row = best_rows.iloc[-1] if not best_rows.empty else reg_metrics.iloc[-1]

            c1, c2, c3 = st.columns(3)
            c1.metric("Best Regression Model", row["model"])
            c2.metric("MAE", f"{row['mae']:.4f}")
            c3.metric("Direction Accuracy", f"{row['direction_accuracy']*100:.2f}%")
            c4, c5 = st.columns(2)
            c4.metric("R² Score", f"{row['r2']:.3f}")
            c5.metric("Spearman Corr.", f"{row['spearman']:.3f}")
            st.caption(f"Last trained: **{row['train_date']}** — using **{int(row['rows'])} samples**")

            if len(best_rows) > 1:
                fig_reg = px.line(
                    best_rows.sort_values("train_date"),
                    x="train_date",
                    y=["mae", "direction_accuracy", "spearman"],
                    markers=True,
                    title="Regression Model Trend Over Time"
                )
                st.plotly_chart(fig_reg, use_container_width=True)

            st.markdown("---")

            st.markdown("### XGBoost Classifier")
            st.caption("Predicts UP or DOWN direction. MAE/R²/Spearman not applicable — only Direction Accuracy matters.")

            xgb_rows = metrics[metrics["model"] == "XGBoost_Classifier"]
            if not xgb_rows.empty:
                latest_xgb = xgb_rows.sort_values("train_date").iloc[-1]
                c1, c2 = st.columns(2)
                c1.metric("XGBoost Direction Accuracy",
                          f"{latest_xgb['direction_accuracy']*100:.2f}%")
                c2.metric("Trained On", f"{int(latest_xgb['rows'])} samples")

                if len(xgb_rows) > 1:
                    fig_xgb = px.line(
                        xgb_rows.sort_values("train_date"),
                        x="train_date",
                        y="direction_accuracy",
                        markers=True,
                        title="XGBoost Accuracy Trend"
                    )
                    st.plotly_chart(fig_xgb, use_container_width=True)

            st.markdown("---")

            st.markdown("### Voting Ensemble (XGBoost + LightGBM + RandomForest Classifier)")
            st.caption("Most reliable signal — combines 3 classifiers. When all 3 agree → highest confidence.")

            ens_rows = metrics[metrics["model"] == "Voting_Ensemble"]
            if not ens_rows.empty:
                latest_ens = ens_rows.sort_values("train_date").iloc[-1]
                c1, c2 = st.columns(2)
                c1.metric("Ensemble Direction Accuracy",
                          f"{latest_ens['direction_accuracy']*100:.2f}%")
                c2.metric("Trained On", f"{int(latest_ens['rows'])} samples")

                if len(ens_rows) > 1:
                    fig_ens = px.line(
                        ens_rows.sort_values("train_date"),
                        x="train_date",
                        y="direction_accuracy",
                        markers=True,
                        title="Voting Ensemble Accuracy Trend"
                    )
                    st.plotly_chart(fig_ens, use_container_width=True)

            st.markdown("---")

            st.markdown("### All Training Runs")
            st.caption("MAE/R²/Spearman show 0 for classifiers — these metrics only apply to regression models.")
            st.dataframe(
                metrics.sort_values("train_date", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")
            st.markdown("### 📊 Baseline Comparison")
            st.caption("Proving our models add genuine predictive value beyond naive approaches")

            baseline_models = metrics[metrics["model"].str.startswith("Baseline_")]
            ml_models = metrics[metrics["model"].isin(["XGBoost_Classifier","Voting_Ensemble"])].groupby("model").last().reset_index()

            if not baseline_models.empty:
                latest_baselines = baseline_models.groupby("model").last().reset_index()

                comparison_data = []
                for _, row in latest_baselines.iterrows():
                    comparison_data.append({
                        "Model": row["model"].replace("Baseline_","").replace("_"," "),
                        "Accuracy": row["direction_accuracy"]*100,
                        "Type": "Baseline"
                    })
                for _, row in ml_models.iterrows():
                    comparison_data.append({
                        "Model": row["model"].replace("_"," "),
                        "Accuracy": row["direction_accuracy"]*100,
                        "Type": "ML Model"
                    })

                comp_df = pd.DataFrame(comparison_data).sort_values("Accuracy")
                fig_comp = px.bar(
                    comp_df, x="Model", y="Accuracy",
                    color="Type",
                    color_discrete_map={"Baseline": "#95a5a6", "ML Model": "#2ecc71"},
                    title="ML Models vs Naive Baselines",
                    labels={"Accuracy": "Direction Accuracy (%)"}
                )
                fig_comp.add_hline(y=50, line_dash="dash",
                                annotation_text="50% random",
                                line_color="red")
                st.plotly_chart(fig_comp, use_container_width=True)

                best_ml  = ml_models["direction_accuracy"].max() * 100
                best_bl  = latest_baselines["direction_accuracy"].max() * 100
                edge     = best_ml - best_bl
                st.success(f"✅ ML model edge over best baseline: **+{edge:.2f}%** "
                        f"({best_ml:.2f}% vs {best_bl:.2f}%)")

            st.markdown("---")
            st.markdown("### 📐 Statistical Significance")
            st.caption("Is the accuracy genuinely better than chance?")

            sig_rows = metrics[metrics["model"].str.startswith("Significance_")]
            if not sig_rows.empty:
                latest_sig = sig_rows.groupby("model").last().reset_index()
                for _, row in latest_sig.iterrows():
                    model_label = row["model"].replace("Significance_","").replace("_"," ")
                    p_val      = row["mae"]
                    ci_low     = row["r2"] * 100
                    ci_high    = row["direction_accuracy"] * 100
                    significant = row["spearman"] == 1.0
                    n_samples  = int(row["rows"])

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(f"{model_label}", f"{(ci_low+ci_high)/2:.2f}%")
                    col2.metric("p-value", f"{p_val:.6f}",
                                delta="Significant ✅" if significant else "Not significant ❌",
                                delta_color="normal" if significant else "inverse")
                    col3.metric("95% CI Low",  f"{ci_low:.2f}%")
                    col4.metric("95% CI High", f"{ci_high:.2f}%")
                    st.caption(f"Based on {n_samples:,} samples. "
                            f"{'Statistically significant improvement over 50% baseline.' if significant else 'Not yet statistically significant.'}")
            else:
                st.info("Statistical significance data will appear after next training run.")

    else:
        st.info("Model metrics not found yet. They will appear after the first weekly training run.")

    # ══════════════════════════════════════════════════
    # SHAP FEATURE ATTRIBUTION
    # ══════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔍 SHAP Feature Attribution")
    st.caption(
        "Which features drive the XGBoost 1-day UP/DOWN signal — and in which direction. "
        "Updated every time the model is retrained. "
        "Generated by `src/train_regression.py` → `reports/shap/`"
    )

    shap_csv_path  = os.path.join(BASE_DIR, "reports/shap/shap_summary.csv")
    beeswarm_path  = os.path.join(BASE_DIR, "reports/shap/beeswarm.png")
    bar_chart_path = os.path.join(BASE_DIR, "reports/shap/feature_importance.png")

    if not os.path.exists(shap_csv_path):
        st.info(
            "SHAP plots not generated yet. "
            "They appear automatically after the next training run. "
            "To generate now: `python src/train_regression.py`"
        )
    else:
        import time
        shap_age_h = (time.time() - os.path.getmtime(shap_csv_path)) / 3600
        shap_df = pd.read_csv(shap_csv_path)

        st.caption(f"Last updated: **{shap_age_h:.1f}h ago** · {len(shap_df)} features")

        # ── Top-15 bar chart (interactive, Plotly) ──
        top15 = shap_df.head(15).copy()
        top15["feature_label"] = top15["feature"].str.replace("_", " ")

        fig_shap_bar = px.bar(
            top15.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature_label",
            orientation="h",
            title="Top 15 Features by Mean |SHAP| Value (XGBoost 1-day)",
            labels={"mean_abs_shap": "Mean |SHAP| value", "feature_label": "Feature"},
            color="mean_abs_shap",
            color_continuous_scale="Blues",
        )
        fig_shap_bar.update_layout(
            height=420,
            margin=dict(t=40, b=10, l=10, r=10),
            coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(size=12)),
        )
        st.plotly_chart(fig_shap_bar, use_container_width=True)

        # ── Beeswarm image + Bar image side by side ──
        col_bee, col_bar = st.columns(2)
        with col_bee:
            st.markdown("**Beeswarm Plot** — direction + magnitude per sample")
            st.caption("Pink = high feature value pushes UP · Blue = high feature value pushes DOWN")
            if os.path.exists(beeswarm_path):
                st.image(beeswarm_path, use_column_width=True)
            else:
                st.info("Beeswarm image not found — rerun training.")

        with col_bar:
            st.markdown("**Feature Importance Bar** — mean |SHAP| ranking")
            st.caption("Higher bar = stronger average influence on prediction")
            if os.path.exists(bar_chart_path):
                st.image(bar_chart_path, use_column_width=True)
            else:
                st.info("Bar chart image not found — rerun training.")

        # ── Full table ──
        with st.expander("📋 Full SHAP Feature Table", expanded=False):
            st.dataframe(
                shap_df[["rank", "feature", "mean_abs_shap"]].rename(columns={
                    "rank":          "Rank",
                    "feature":       "Feature",
                    "mean_abs_shap": "Mean |SHAP|",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("""
        **How to read SHAP:**
        - **Beeswarm**: each dot = one sample (stock × day). X-axis = SHAP value (push toward UP or DOWN).
          Colour = feature value (red/pink = high, blue = low).
        - **Bar chart**: average absolute SHAP — overall feature importance ranking.
        - A feature with high |SHAP| is the model's strongest signal driver.
          If `smart_score` tops the list, sentiment is your edge. If `fii_net` tops it, macro flows dominate.
        """)

    st.markdown("---")
    st.markdown("### Notes")
    st.write("""
    - Predictions use TimeSeriesSplit CV to avoid leakage.
    - Direction accuracy usually improves as more history accumulates.
    - MAE helps judge confidence: lower MAE = more reliable predictions.
    - The model is retrained automatically **every weekday evening** (via GitHub Actions).
    - SmartScore features include recency, events, breadth, and volume signals.
    """)

# ================================ TECH & WORKFLOW =====================
with tab7:
    st.subheader("Technical Details & Workflow")

    st.markdown("""
    **What this app does, end-to-end:**
    I ingest **public RSS headlines** from Google News, Moneycontrol, ET Markets and Investing.com, map each headline to an **NSE ticker**, score sentiment with a **FinBERT + VADER ensemble**, classify the **event type** (earnings, M&A, penalties, etc.), and aggregate the last 10 days using **recency-decay (EWMA)**, **event weights**, **breadth**, and **news volume** into a single **SmartScore (0–100)**.
    Daily SmartScores are joined with **yfinance** prices to train a simple predictive model (**Ridge / RandomForest**) using **TimeSeriesSplit**. The app then publishes **next-day return signals** and interactive visualizations via **Streamlit + Plotly**.
    """)

    colA, colB = st.columns([1.45, 1])
    with colA:
        st.markdown("### Step-by-Step Workflow")
        st.markdown("""
        1. **Data ingestion** → `feedparser` pulls headlines from **319 RSS sources** (Google News, BS, ET Markets, Mint, BusinessLine, NDTVProfit and more).
           • Normalize URLs, parse timestamps to **UTC**, de-duplicate via a stable hash.
           • **Ticker mapping** via alias-regex (242 patterns) with a mapping confidence score.
        2. **NLP sentiment** → **VADER** (lexicon) and **FinBERT** (finance transformer) → **ensemble** ∈ [-1, 1].
           • Also track **model confidence** and assign {negative, neutral, positive}.
        3. **Event classification** → rule-based tags: **EARNINGS, GUIDANCE, M&A, LITIGATION, REGULATORY, MGMT_CHANGE, ORDER_WIN, PRODUCT_LAUNCH, MACRO**.
           • Each event type has a signed weight (e.g., **EARNINGS ↑**, **LITIGATION ↓**).
        4. **Aggregation → SmartScore (0–100)** over a 10-day window:
           • **S_recency:** Dual EWMA (8h half-life for 1d, 36h for 3d predictions)
           • **S_events:** event-weighted sentiment (tone × event impact)
           • **S_breadth:** (pos − neg) / total, cross-sectional Z-score
           • **S_volume:** log(news count), cross-sectional Z-score
           **SmartScore = 0.45·S_recency + 0.25·S_events + 0.20·S_breadth + 0.10·S_volume**
        5. **Modeling (next-day)** → join SmartScores with adjusted close from **yfinance**; label = **t→t+1 % return**.
           • 24 features including SmartScore components, price momentum, 18 macro indicators, FII/DII flows.
           • Train **Ridge, RandomForest, XGBoost, LightGBM, Voting Ensemble** with **TimeSeriesSplit**.
           • Report **MAE**, **Direction Accuracy**, **Spearman**.
        6. **Signals & UI** → generate **predicted next-day returns** per ticker; visualize in Streamlit with filters.
        """)

        st.markdown("### Why This Stack")
        st.markdown("""
        - **feedparser + RSS**: robust, legal, and fast access to public headlines (no paywalls scraped).
        - **FinBERT + VADER**: transformer tuned for finance **plus** a lexicon model ⇒ complementary strengths.
        - **EWMA + event weighting**: markets react more to **fresh** and **material** news; this captures both.
        - **Ridge / RandomForest / XGBoost**: strong baselines for tabular data; interpretable with TimeSeriesSplit.
        - **Streamlit + Plotly**: clean, interactive analytics UI.
        - **GitHub Actions**: reliable automation for 30-minute refresh and daily retraining.
        """)

        st.markdown("### Automation & Retraining")
        st.markdown("""
        - **Ingestion & scoring**: run **every 30 minutes** during India market hours (IST 08:30–15:30) → updates SmartScores & signals.
        - **Model retraining**: **every weekday evening** (IST 19:30) via GitHub Actions.
        - **Dashboard**: auto-reads the latest CSVs; no restart needed in cloud deployments.
        """)

        st.markdown("### Deployment")
        st.markdown("""
        Deployed on: **Streamlit Cloud**
        Pipeline auto-runs via **GitHub Actions** and commits refreshed CSVs to the repo; the app serves the newest files.
        """)

        st.markdown("### Legal & Ethics")
        st.markdown("""
        This project uses **public RSS headlines only** (no article bodies) for **academic/personal research**.
        Price data via `yfinance`. **No financial advice** and **no redistribution** of copyrighted content.
        All signals are research indicators only — not investment recommendations.
        """)

    with colB:
        st.markdown("### Key Files")
        st.code(
            "src/fetch_news.py                  # RSS ingest, mapping, UTC, de-dup\n"
            "src/sentiment_vader.py             # FinBERT+VADER ensemble sentiment\n"
            "src/event_classifier.py            # rule-based event tags from headlines\n"
            "src/aggregate_sentiment.py         # SmartScore v2 (0–100) per ticker\n"
            "src/backfill_history.py            # rebuild multi-day SmartScore history\n"
            "src/price_labels.py                # yfinance prices + forward returns\n"
            "src/build_dataset.py               # join features with prices, create labels\n"
            "src/train_regression.py            # Ridge/RF + TimeSeriesSplit CV, save best model\n"
            "src/predict_next.py                # next-day return predictions\n"
            "src/run_daily.py                   # local full pipeline + open dashboard\n"
            "dashboard/app.py                   # Streamlit UI\n"
            ".github/workflows/daily_ingest.yml # scheduled news + scoring\n"
            ".github/workflows/weekly_train.yml # scheduled retrain + metrics\n",
            language="text",
        )

        st.markdown("### Workflow")
        dot = r'''
        digraph G {
        rankdir=TB;
        node [shape=rectangle, style=rounded, fontsize=10];

        subgraph cluster_sources {
            label="Sources";
            style=dashed;
            Google [label="Google News RSS"];
            MC [label="Moneycontrol RSS"];
            ET [label="ET Markets RSS"];
            Inv [label="Investing.com RSS"];
            YF [label="Yahoo Finance\n(yfinance)"];
        }

        Fetch [label="fetch_news.py\n(clean, UTC, map tickers, dedup)"];
        Proc  [label="sentiment_vader.py\n(FinBERT+VADER → ensemble)"];
        Agg   [label="aggregate_sentiment.py\n(EWMA, events, breadth, volume → SmartScore)"];
        Hist  [label="history CSV\n(stock_sentiment_summary_history.csv)"];
        Build [label="build_dataset.py\n+ price_labels.py\n(join features with prices,\ncreate labels)"];
        Train [label="train_regression.py\n(Ridge/RF/XGBoost, TSSplit CV)"];
        Model [label="models/nextday_regressor.pkl"];
        Pred  [label="predict_next.py\n(next-day % returns)"];
        Dash  [label="Streamlit dashboard\n(dashboard/app.py)"];

        Google -> Fetch; MC -> Fetch; ET -> Fetch; Inv -> Fetch;
        YF -> Build;

        Fetch -> Proc -> Agg -> Hist;
        Hist -> Build -> Train -> Model -> Pred -> Dash;
        Agg  -> Dash;
        }
        '''
        st.graphviz_chart(dot)