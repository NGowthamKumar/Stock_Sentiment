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
    ["📊 Market Brief", "Predictions", "Signals", "Stock Accuracy", "Stock Drilldown", "Model Health", "Tech & Workflow"]
)

# ================================ MARKET INTELLIGENCE BRIEF ============================
with tab1:

    st.markdown("## 📊 Market Intelligence Brief")
    st.caption(f"Updated: {fmt_dt()} IST  •  Auto-refreshes each pipeline run")

    # ── IMPORTANT DISCLAIMER ──
    st.warning(
        "⚠️ **Research tool only — not financial advice.** "
        "This dashboard aggregates publicly available data (RSS news, yfinance prices, NSE FII/DII). "
        "Macro risk scores are computed from available data only and do NOT include Fed/RBI meeting calendars, "
        "earnings dates, or options expiry. Always verify independently before making any financial decisions."
    )

    # Load additional data
    ens_signals = load_csv("data/ensemble_signals.csv")
    streaks     = load_csv("data/stock_streaks.csv")
    accuracy    = load_csv("data/stock_accuracy.csv")
    fii_dii     = load_csv("data/fii_dii_history.csv")
    signals_3d  = load_csv("data/signals_3d.csv")
    metrics_df  = load_csv("data/modeling/model_metrics.csv")

    # Fetch macro
    with st.spinner("Fetching live macro data..."):
        macro = get_macro_snapshot()

    # ── Compute macro risk from available data only ──
    crude_chg  = macro.get("crude_oil", {}).get("change", 0)
    us_vix_val = macro.get("us_vix",    {}).get("value",  15)
    us_vix_chg = macro.get("us_vix",    {}).get("change", 0)
    us_10y_chg = macro.get("us_10y",    {}).get("change", 0)
    us_10y_val = macro.get("us_10y",    {}).get("value",  4.0)
    fii_latest = 0
    dii_latest = 0
    if not fii_dii.empty:
        fii_latest = float(fii_dii.iloc[-1]["fii_net"])
        dii_latest = float(fii_dii.iloc[-1]["dii_net"])

    # ── Load news counts for risk scoring ──
    raw_news_path = os.path.join(BASE_DIR, "data/raw_news.csv")
    news_counts   = {}
    today_str     = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(raw_news_path):
        try:
            raw_news = load_csv("data/raw_news.csv")
            if not raw_news.empty and "published_utc" in raw_news.columns:
                raw_news["published_utc"] = pd.to_datetime(
                    raw_news["published_utc"], utc=True, errors="coerce"
                )
                today_news = raw_news[raw_news["published_utc"].dt.date.astype(str) >= today_str]
                topics = {
                    "Oil/Crude":      ["crude","brent","oil price"],
                    "RBI":            ["rbi","repo rate","monetary policy"],
                    "Iran/Hormuz":    ["iran","hormuz","strait"],
                    "Fed/Rate":       ["fed","federal reserve","rate hike","fomc"],
                    "IT sector":      ["infosys","wipro","hcltech","it sector"],
                    "Gold":           ["gold price","bullion","gold etf"],
                    "Nifty/Sensex":   ["nifty","sensex"],
                    "Semiconductors": ["semiconductor","chip","nvidia"],
                    "Rupee":          ["rupee","inr"],
                    "FII/DII":        ["fii","dii","foreign institutional"],
                }
                for topic, kws in topics.items():
                    count = sum(
                        today_news["title"].str.lower().str.contains(kw, na=False).sum()
                        for kw in kws
                    )
                    if count > 0:
                        news_counts[topic] = count
        except:
            pass

    # Compute risk score
    risk_score   = 0
    risk_reasons = []
    if us_vix_val > 25:       risk_score += 30; risk_reasons.append(f"US VIX high ({us_vix_val:.1f})")
    elif us_vix_val > 20:     risk_score += 15; risk_reasons.append(f"US VIX elevated ({us_vix_val:.1f})")
    if crude_chg > 3:         risk_score += 25; risk_reasons.append(f"Crude surging (+{crude_chg:.1f}%)")
    elif crude_chg > 1.5:     risk_score += 12; risk_reasons.append(f"Crude rising (+{crude_chg:.1f}%)")
    if us_10y_chg > 0.5:      risk_score += 15; risk_reasons.append(f"US yield rising (+{us_10y_chg:.2f}%)")
    if us_10y_val > 4.8:      risk_score += 10; risk_reasons.append(f"US 10yr near 5% danger zone ({us_10y_val:.2f}%)")
    if fii_latest < -5000:    risk_score += 25; risk_reasons.append("Heavy FII selling")
    elif fii_latest < -2000:  risk_score += 12; risk_reasons.append("FII selling")
    elif fii_latest < -500:   risk_score += 8;  risk_reasons.append("FII mild selling")
    if us_vix_chg > 5:        risk_score += 10; risk_reasons.append("Global fear rising")
    # Add Fed/RBI news count to risk
    fed_count = news_counts.get("Fed/Rate", 0)
    if fed_count > 50:        risk_score += 20; risk_reasons.append("Fed decision imminent")
    elif fed_count > 20:      risk_score += 10; risk_reasons.append("Fed meeting approaching")
    rbi_count = news_counts.get("RBI", 0)
    if rbi_count > 100:       risk_score += 10; risk_reasons.append("RBI policy focus")

    # ── SECTION 1: MACRO ENVIRONMENT + ACCURACY ──
    st.markdown("---")
    st.markdown("#### 📊 Current Macro Environment")
    st.caption("Based on: yfinance (prices), NSE (FII/DII), RSS feeds (news counts). Does NOT include Fed/RBI meeting calendars.")

    col_v1, col_v2, col_v3 = st.columns([1, 1.5, 1.5])

    with col_v1:
        st.markdown("**Macro Headwind Score**")
        st.caption("Higher = more macro pressure on markets")
        if risk_score >= 50:
            st.error(f"### 🔴 HIGH\n{risk_score}/100")
        elif risk_score >= 25:
            st.warning(f"### 🟡 MODERATE\n{risk_score}/100")
        else:
            st.success(f"### 🟢 LOW\n{risk_score}/100")
        if risk_reasons:
            st.caption("Drivers detected:\n" + "\n".join(f"• {r}" for r in risk_reasons[:4]))
        st.caption("⚠️ This score uses only data your system fetches. Undetected events (Fed calendar, earnings) are NOT reflected here.")

    with col_v2:
        st.markdown("**Live Macro Data**")
        crude = macro.get("crude_oil", {})
        if crude:
            ic = "✅" if crude["change"] < -1 else "❌" if crude["change"] > 2 else "⚠️"
            st.markdown(f"{ic} **Brent Crude**: ${crude['value']:.1f} ({crude['change']:+.1f}%)")

        fii_ic = "🔴" if fii_latest < -3000 else "⚠️" if fii_latest < 0 else "🟢"
        dii_ic = "🟢" if dii_latest > 1000 else "🟡"
        st.markdown(f"{fii_ic} **FII**: ₹{fii_latest:+,.0f}cr  |  {dii_ic} **DII**: ₹{dii_latest:+,.0f}cr")

        gold = macro.get("gold", {})
        if gold:
            g_ic = "📈" if gold["change"] > 0.5 else "📉" if gold["change"] < -0.5 else "➡️"
            st.markdown(f"{g_ic} **Gold**: ${gold['value']:,.0f} ({gold['change']:+.1f}%)")

        uv = macro.get("us_vix", {})
        if uv:
            v_ic = "✅" if uv["change"] < -3 else "⚠️" if uv["value"] > 20 else "🟡"
            st.markdown(f"{v_ic} **US VIX**: {uv['value']:.1f} ({uv['change']:+.1f}%)")

        u10 = macro.get("us_10y", {})
        if u10:
            y_ic = "⚠️" if u10["value"] > 4.8 else "🟡" if u10["change"] > 0.3 else "✅"
            st.markdown(f"{y_ic} **US 10yr**: {u10['value']:.2f}% ({u10['change']:+.2f}%)")

        sp = macro.get("sp500", {})
        if sp:
            s_ic = "✅" if sp["change"] > 0.3 else "⚠️" if sp["change"] < -0.5 else "🟡"
            st.markdown(f"{s_ic} **S&P 500**: {sp['value']:,.0f} ({sp['change']:+.1f}%)")

        usdvix = macro.get("usd_inr", {})
        if usdvix:
            r_ic = "⚠️" if usdvix["change"] > 0.3 else "✅"
            st.markdown(f"{r_ic} **USD/INR**: {usdvix['value']:.2f} ({usdvix['change']:+.2f}%)")
            
        nifty = macro.get("nifty50", {})
        if nifty:
            n_ic = "✅" if nifty["change"] > 0.3 else "⚠️" if nifty["change"] < -0.5 else "🟡"
            st.markdown(f"{n_ic} **Nifty 50**: {nifty['value']:,.0f} ({nifty['change']:+.1f}%)")

        iv = macro.get("india_vix", {})
        if iv:
            i_ic = "⚠️" if iv["value"] > 18 else "✅" if iv["value"] < 14 else "🟡"
            st.markdown(f"{i_ic} **India VIX**: {iv['value']:.1f} ({iv['change']:+.1f}%)")

    with col_v3:
        st.markdown("**System Accuracy (latest run)**")
        if not metrics_df.empty:
            metrics_df["train_date"] = pd.to_datetime(metrics_df["train_date"], errors="coerce")
            for model_name, label in [
                ("Ridge",               "Ridge"),
                ("Voting_Ensemble",     "Ensemble 1d"),
                ("Voting_3Day_Ensemble","Ensemble 3d"),
                ("XGBoost_Classifier",  "XGBoost 1d"),
            ]:
                rows = metrics_df[metrics_df["model"] == model_name].sort_values("train_date")
                if not rows.empty:
                    val = rows.iloc[-1]["direction_accuracy"] * 100
                    n   = int(rows.iloc[-1]["rows"])
                    st.metric(label, f"{val:.2f}%", delta=f"{n:,} rows")
        else:
            st.info("Run training to see accuracy")

    st.markdown("---")

    # ── SECTION 2: SECTOR HEATMAP ──
    st.markdown("#### 🗺️ Sector Signals (based on available macro data)")
    st.caption("Derived from: Nifty IT/Bank change (yfinance), Crude oil change (yfinance), India/US VIX")
    sectors = get_sector_heatmap(macro)
    sec_cols = st.columns(min(4, len(sectors)))
    for i, (name, icon, status, reason) in enumerate(sectors):
        with sec_cols[i % len(sec_cols)]:
            st.markdown(f"{icon} **{name}**")
            st.caption(f"{status} — {reason}")

    st.markdown("---")

    # ── SECTION 3: TOP SIGNALS + AVOID ──
    col_buy, col_avoid = st.columns(2)

    with col_buy:
        st.markdown("#### 🏆 High-Trust Signals")
        st.caption(
            "Trust Score = 40% historical accuracy + 35% win rate (10d) + 25% current ensemble signal. "
            "Based on your system's own past prediction history."
        )

        top_signals = get_top_signals(accuracy, streaks, ens_signals, summary)
        if not top_signals.empty:
            medals = ["🥇", "🥈", "🥉"]
            for i, (_, row) in enumerate(top_signals.iterrows()):
                if i >= 3:
                    break
                tk        = row["ticker"].replace(".NS", "")
                trust     = row["trust_score"]
                bar       = "█" * int(trust/10) + "░" * (10-int(trust/10))
                t_color   = "🟢" if trust >= 75 else "🟡" if trust >= 60 else "🔴"
                streak    = int(row.get("pos_day_streak", 0))
                win_rate  = row.get("win_rate_10d", 0)
                combined  = row.get("combined_acc", 0)
                ens_prob  = row.get("ensemble_probability", 50)
                total_sig = int(row.get("total_signals", 0))
                ss        = row.get("smart_score", 0)

                with st.expander(f"{medals[i]} **{tk}** — {t_color} Trust: {trust:.0f}% | {bar}"):
                    c1, c2 = st.columns(2)
                    c1.metric("Win Rate (10d)",   f"{win_rate:.1f}%")
                    c2.metric("Positive Streak",   f"{streak} days 🔥")
                    c3, c4 = st.columns(2)
                    c3.metric("Historical Acc.",   f"{combined:.1f}%")
                    c4.metric("Ensemble Signal",   f"{ens_prob:.1f}%")
                    c5, c6 = st.columns(2)
                    c5.metric("Total Signals",     str(total_sig))
                    c6.metric("SmartScore",        f"{ss:.1f}")
                    st.caption(
                        "⚠️ Past accuracy does not guarantee future returns. "
                        "This is a research signal, not financial advice."
                    )
        else:
            st.info("Insufficient signal history. Check back after more trading days.")

    with col_avoid:
        st.markdown("#### ⚠️ Weak Signals (Negative Sentiment)")
        st.caption("Stocks with SmartScore < 35 and negative ML signals over last 10 days")

        avoid_df = get_avoid_list(summary, ens_signals)
        if not avoid_df.empty:
            for _, row in avoid_df.iterrows():
                tk     = row["ticker"].replace(".NS", "")
                ss     = row["smart_score"]
                reason = get_avoid_reason(row["ticker"], ss)
                ep     = row.get("ensemble_probability", 50)
                with st.expander(f"⚠️ **{tk}** — SmartScore: {ss:.0f}/100"):
                    st.markdown(f"**Reason for weak signal:** {reason}")
                    if pd.notna(ep):
                        st.metric("Ensemble Signal", f"{ep:.1f}%",
                                  delta="Bearish" if ep < 45 else "Neutral",
                                  delta_color="inverse")
                    st.caption("SmartScore < 35 = negative sentiment dominates last 10 days of news")
        else:
            st.info("No strong negative signals today")

    st.markdown("---")

    # ── SECTION 4: KEY DRIVERS ──
    st.markdown("#### 📰 News Volume by Topic Today")
    st.caption("Article counts from RSS feeds — higher count = more coverage of that topic today")

    if news_counts:
        sorted_topics = sorted(news_counts.items(), key=lambda x: x[1], reverse=True)
        icon_map = {
            "Oil/Crude":"🛢️","Iran/Hormuz":"🌍","Fed/Rate":"🇺🇸",
            "RBI":"🏛️","Gold":"💰","IT sector":"💻",
            "FII/DII":"💵","Nifty/Sensex":"📈",
            "Semiconductors":"🔧","Rupee":"₹"
        }
        n_cols = min(5, len(sorted_topics))
        drv_cols = st.columns(n_cols)
        for i, (topic, count) in enumerate(sorted_topics[:n_cols]):
            with drv_cols[i]:
                ic = icon_map.get(topic, "📰")
                st.metric(f"{ic} {topic}", f"{count} articles")
        st.caption("Note: Article count ≠ market impact. Use as a qualitative indicator of what the market is focused on today.")
    else:
        st.info("News data loading... Run pipeline to update")

    st.markdown("---")

    # ── SECTION 5: GEOPOLITICAL SIGNALS FROM NEWS ──
    st.markdown("#### 🌍 Geopolitical Signals (based on news article counts)")
    st.caption("Levels derived from today's article counts only — not from real-time intelligence sources")
    geo1, geo2 = st.columns(2)

    with geo1:
        iran  = news_counts.get("Iran/Hormuz", 0)
        il    = "🔴 HIGH" if iran > 50 else "🟡 MEDIUM" if iran > 20 else "🟢 LOW"
        st.markdown(f"**🇮🇷 Iran/Hormuz**: {il}")
        st.caption(f"{iran} articles today — crude & shipping disruption news")

        fed   = news_counts.get("Fed/Rate", 0)
        fl    = "🔴 HIGH" if fed > 50 else "🟡 MEDIUM" if fed > 20 else "🟢 LOW"
        st.markdown(f"**🇺🇸 US Fed**: {fl}")
        st.caption(f"{fed} articles today — rate expectations drive FII flows")

    with geo2:
        u10v  = macro.get("us_10y", {}).get("value", 4.5)
        jl    = "🔴 HIGH" if u10v > 4.9 else "🟡 MEDIUM" if u10v > 4.7 else "🟢 LOW"
        st.markdown(f"**🇯🇵 Japan / Treasury selloff**: {jl}")
        st.caption(f"US 10yr at {u10v:.2f}% — proxy for Treasury demand pressure")

        gchg  = macro.get("gold", {}).get("change", 0)
        gl    = "🔴 HIGH" if gchg > 1.5 else "🟡 MEDIUM" if gchg > 0 else "🟢 LOW"
        st.markdown(f"**🌐 De-dollarisation / Gold**: {gl}")
        st.caption(f"Gold {gchg:+.1f}% today — central bank demand signal")

    st.markdown("---")

    # ── SECTION 6: TOMORROW'S SIGNALS ──
    st.markdown("#### 🔮 What to Watch Tomorrow")
    st.caption("Based on current trends in data your system can see — NOT a forecast")
    tomorrow = get_tomorrow_outlook(macro, fii_dii, news_counts)
    if tomorrow:
        for icon, event, implication in tomorrow:
            st.markdown(f"{icon} **{event}**")
            st.caption(f"→ {implication}")
    else:
        st.info("No major risk signals detected in current data for tomorrow")

    st.markdown("---")

    # ── SECTION 7: ORIGINAL DATA (collapsed) ──
    with st.expander("📋 Full SmartScore Table", expanded=False):
        cols  = ["ticker","smart_score","S_recency","S_events","S_breadth","S_volume","pos","neg","total"]
        show  = [c for c in cols if c in summary.columns]
        st.dataframe(summary[show].sort_values("smart_score", ascending=False),
                     use_container_width=True, hide_index=True)

    with st.expander("📊 SmartScore Charts", expanded=False):
        topn = st.slider("Top N", 5, 20, 10, key="topn_overview")
        top_df = summary.nlargest(topn, "smart_score")
        fig = px.bar(top_df, x="ticker", y="smart_score", color="smart_score",
                     title=f"Top {topn} Smart Scores", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        comp_cols = [c for c in ["S_recency","S_events","S_breadth","S_volume"] if c in summary.columns]
        if comp_cols:
            comp_df = summary[["ticker", *comp_cols]].melt(
                id_vars="ticker", var_name="component", value_name="score"
            )
            figc = px.bar(comp_df, x="ticker", y="score", color="component",
                          barmode="group", title="Component Scores (0-100)")
            st.plotly_chart(figc, use_container_width=True)

    with st.expander("ℹ️ About SmartScore & Trust Score", expanded=False):
        st.markdown("""
        **SmartScore** (0-100) combines:
        - **S_recency** (45%) — EWMA sentiment: 8h half-life for 1d predictions, 36h for 3d
        - **S_events** (25%) — Major event magnitude (earnings, litigation, order wins)
        - **S_breadth** (20%) — Positive vs negative ratio (cross-sectional Z-score)
        - **S_volume** (10%) — News volume signal (cross-sectional Z-score)

        **Trust Score** (0-100) per stock:
        - 40% historical combined accuracy (your system's own prediction history)
        - 35% win rate last 10 days
        - 25% current ensemble signal strength

        🟢 Trust ≥ 75 = High confidence in signal quality  
        🟡 Trust 60-74 = Moderate confidence  
        🔴 Trust < 60 = Low confidence — treat with caution

        **What this system cannot see:**
        - Fed/RBI meeting exact dates
        - Earnings release calendar
        - Options expiry dates
        - Insider information
        - Intraday price movements
        """)

# ================================ PREDICTIONS =========================
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
    st.subheader("Stock Drilldown")

    tickers = sorted(summary["ticker"].unique().tolist())
    tk = st.selectbox("Choose ticker", tickers, index=0)

    left, right = st.columns([2, 1])
    with left:
        if not hist.empty:
            date_col = "pred_date" if "pred_date" in hist.columns else "date"
            h = hist[hist["ticker"] == tk].sort_values(date_col)
            if not h.empty:
                fig3 = px.line(h, x=date_col, y="smart_score",
                            title=f"{tk} — SmartScore History")
                st.plotly_chart(fig3, use_container_width=True)

                if "xgb_prob" in h.columns:
                    fig4 = px.line(h, x=date_col, y="xgb_prob",
                                title=f"{tk} — XGBoost UP Probability (%)")
                    fig4.add_hline(y=55, line_dash="dash",
                                annotation_text="55% threshold")
                    st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No signal history yet for this stock.")

    with right:
        row = summary[summary["ticker"] == tk].iloc[0]
        st.metric("Smart Score", f"{row.smart_score:.2f}")
        st.metric("S_recency", f"{row.S_recency:.1f}")
        st.metric("S_events", f"{row.S_events:.1f}")
        st.metric("S_breadth", f"{row.S_breadth:.1f}")
        st.metric("S_volume", f"{row.S_volume:.1f}")
        st.write(f" Pos: **{int(row.pos)}** Neg: **{int(row.neg)}** Total: **{int(row.total)}**")

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

    else:
        st.info("Model metrics not found yet. They will appear after the first weekly training run.")

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

        st.caption("Tip: CI runs fetch→sentiment→aggregate→predict on schedule; retraining happens daily.")
