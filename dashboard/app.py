# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Indian Stock Sentiment Analyser", layout="wide")

# ---------- helpers ----------
@st.cache_data
def load_csv_with_mtime(path, mtime, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()
    
def load_csv(path, parse_dates=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    mtime = os.path.getmtime(path)   # cache busting key
    return load_csv_with_mtime(path, mtime, parse_dates)

def fmt_dt(ts=None):
    return datetime.now().strftime("%Y-%m-%d %H:%M")

# ---------- load data ----------
summary = load_csv("data/stock_sentiment_summary.csv")              # today snapshot
hist    = load_csv("data/history/stock_sentiment_summary_history.csv", parse_dates=["date"])
preds   = load_csv("data/predictions_nextday.csv")

st.title("Indian Stock Sentiment Analyser")
st.caption(f"Last updated: {fmt_dt()}  •  Data: Google News, Moneycontrol, ET Markets, Investing.com")

# guard rails
if summary.empty:
    st.warning("No summary found at data/stock_sentiment_summary.csv. Run the pipeline first.")
    st.stop()

# ensure types
if "date" in hist.columns:
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")

# -------------------------------- TABS --------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Predictions", "Stock Drilldown", "Model Health", "Tech & Workflow"]
)

# ================================ OVERVIEW ============================
with tab1:
    st.subheader(" Portfolio Overview (Today)")

    # show all components
    cols = ["ticker","smart_score","S_recency","S_events","S_breadth","S_volume","pos","neg","total"]
    show = [c for c in cols if c in summary.columns]
    df_show = summary[show].sort_values("smart_score", ascending=False)
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.subheader("Top by Smart Score")
    topn = st.slider("Top N", 5, 20, 10, key="topn_overview")
    top_df = summary.nlargest(topn, "smart_score")
    fig = px.bar(top_df, x="ticker", y="smart_score", color="smart_score",
                 title=f"Top {topn} Smart Scores", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    # components breakdown
    st.subheader("Component Comparison")
    comp_cols = ["S_recency","S_events","S_breadth","S_volume"]
    comp_df = summary[["ticker", *comp_cols]].melt(id_vars="ticker", var_name="component", value_name="score")
    figc = px.bar(comp_df, x="ticker", y="score", color="component", barmode="group",
                  title="Component Scores (0–100)")
    st.plotly_chart(figc, use_container_width=True)

# ================================ PREDICTIONS =========================
with tab2:
    st.subheader("Predicted Next-Day Returns")

    if preds.empty:
        st.info("No predictions yet. Train a model and run predict_next.py.")
    else:
        # Optional confidence band if you log MAE; for now, compute a rough confidence proxy
        mae_guess = 0.25  # % — adjust after training logs
        preds["confidence"] = (preds["pred_ret_1d_pct"].abs() / mae_guess).clip(0, 2.0)

        # Filters
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
            (merged["total"] >= 3) &                 # breadth filter
            (merged["S_breadth"] >= 50)              # avoid single-headline bias
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

        st.caption("Note: Predictions are signals, not guarantees. Use thresholds and breadth/event filters.")

# ================================ DRILLDOWN ===========================
with tab3:
    st.subheader("Stock Drilldown")

    tickers = sorted(summary["ticker"].unique().tolist())
    tk = st.selectbox("Choose ticker", tickers, index=0)

    left, right = st.columns([2, 1])
    with left:
        if not hist.empty:
            h = hist[hist["ticker"] == tk].sort_values("date")
            if not h.empty:
                fig3 = px.line(h, x="date", y="smart_score", title=f"{tk} — SmartScore over time")
                st.plotly_chart(fig3, use_container_width=True)

                fig4 = px.line(h, x="date", y=["S_recency","S_events","S_breadth","S_volume"],
                               title=f"{tk} — Components (0–100)")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No historical snapshots yet for this ticker.")
        else:
            st.info("History file not found. It will appear after a few daily runs or backfill.")

    with right:
        row = summary[summary["ticker"] == tk].iloc[0]
        st.metric("Smart Score", f"{row.smart_score:.2f}")
        st.metric("S_recency", f"{row.S_recency:.1f}")
        st.metric("S_events", f"{row.S_events:.1f}")
        st.metric("S_breadth", f"{row.S_breadth:.1f}")
        st.metric("S_volume", f"{row.S_volume:.1f}")
        st.write(f" Pos: **{int(row.pos)}** Neg: **{int(row.neg)}** Total: **{int(row.total)}**")

# ================================ MODEL HEALTH =======================
with tab4:
    st.subheader("Model Health Dashboard")
    import os
    metrics_path = "data/modeling/model_metrics.csv"
    if os.path.exists(metrics_path):
        metrics = load_csv(metrics_path)
        if not metrics.empty:
            row = metrics.iloc[-1]  # latest training row
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Model", row["best_model"])
            c2.metric("MAE", f"{row['mae']:.4f}")
            c3.metric("Direction Accuracy", f"{row['direction_accuracy']*100:.2f}%")
            c4, c5 = st.columns(2)
            c4.metric("R² Score", f"{row['r2']:.3f}")
            c5.metric("Spearman Corr.", f"{row['spearman']:.3f}")
            st.caption(
                f"Last trained: **{row['train_date']}** — using **{int(row['rows'])} samples**"
            )
            st.caption(
                f"Metrics file updated at: "
                f"{datetime.fromtimestamp(os.path.getmtime(metrics_path))}"
            )

            st.markdown("---")
            if len(metrics) > 1:
                st.markdown("### Training Trend")
                st.line_chart(metrics[["mae", "direction_accuracy"]])
    else:
        st.info("Model metrics not found yet. They will appear after the first weekly training run.")
    st.markdown("---")
    st.markdown("### Notes")
    st.write("""
    - Predictions use TimeSeriesSplit CV to avoid leakage.
    - Direction accuracy usually improves as more history accumulates.
    - MAE helps judge confidence: lower MAE = more reliable predictions.
    - The model is retrained automatically **every Friday** (via GitHub Actions).
    - SmartScore features include recency, events, breadth, and volume signals.
    """)
    
# ================================ TECH & WORKFLOW =====================
with tab5:
    st.subheader("Technical Details & Workflow")

    # ---------- 0) One-paragraph explainer ----------
    st.markdown("""
    **What this app does, end-to-end:**  
    I ingest **public RSS headlines** from Google News, Moneycontrol, ET Markets and Investing.com, map each headline to an **NSE ticker**, score sentiment with a **FinBERT + VADER ensemble**, classify the **event type** (earnings, M&A, penalties, etc.), and aggregate the last 10 days using **recency-decay (EWMA)**, **event weights**, **breadth**, and **news volume** into a single **SmartScore (0–100)**.  
    Daily SmartScores are joined with **yfinance** prices to train a simple predictive model (**Ridge / RandomForest**) using **TimeSeriesSplit**. The app then publishes **next-day return signals** and interactive visualizations via **Streamlit + Plotly**.
    """)

    colA, colB = st.columns([1.45, 1])
    with colA:
        # ---------- 1) Step-by-step data flow ----------
        st.markdown("### Step-by-Step Workflow")
        st.markdown("""
        1. **Data ingestion** → `feedparser` pulls headlines from **Google News, Moneycontrol, ET Markets, Investing.com**.  
           • Normalize URLs, parse timestamps to **UTC**, de-duplicate via a stable hash.  
           • **Ticker mapping** via alias-regex (e.g., "HCL Tech" → `HCLTECH.NS`) with a mapping confidence score.
        2. **NLP sentiment** → **VADER** (lexicon) and **FinBERT** (finance transformer) → **ensemble** ∈ [-1, 1].  
           • Also track **model disagreement**/**confidence** and assign {negative, neutral, positive}.
        3. **Event classification** → rule-based tags: **EARNINGS, GUIDANCE, M&A, LITIGATION, REGULATORY, MGMT_CHANGE, ORDER_WIN, PRODUCT_LAUNCH, MACRO**.  
           • Each event type has a signed weight (e.g., **EARNINGS ↑**, **LITIGATION ↓**).
        4. **Aggregation → SmartScore (0–100)** over a 10-day window:  
           • **S_recency:** EWMA of ensemble sentiment (half-life ≈ 36h)  
           • **S_events:** event-weighted sentiment (tone × event impact)  
           • **S_breadth:** (pos − neg) / total, min–max normalized across tickers  
           • **S_volume:** log(news count), min–max normalized  
           **SmartScore = 0.45·S_recency + 0.25·S_events + 0.20·S_breadth + 0.10·S_volume**
        5. **Modeling (next-day)** → join SmartScores with adjusted close from **yfinance**; label = **t→t+1 % return**.  
           • Train **Ridge** and **RandomForest** with **TimeSeriesSplit**, report **MAE**, **Direction Accuracy**, **Spearman**.  
           • Pick the best model (lowest MAE; tie-break by direction accuracy).
        6. **Signals & UI** → generate **predicted next-day returns** per ticker; visualize in Streamlit with filters for Recency, Events, Breadth, and Volume.
        """)

        # ---------- 2) Why this stack ----------
        st.markdown("### Why This Stack")
        st.markdown("""
        - **feedparser + RSS**: robust, legal, and fast access to public headlines (no paywalls scraped).  
        - **FinBERT + VADER**: transformer tuned for finance **plus** a lexicon model ⇒ complementary strengths, more stable than either alone.  
        - **EWMA + event weighting**: markets react more to **fresh** and **material** news; this captures both.  
        - **Ridge / RandomForest**: strong baselines for tabular, limited-feature regimes; interpretable and low-variance with TimeSeriesSplit.  
        - **Streamlit + Plotly**: quickest way to ship a clean, interactive analytics UI without heavy frontend work.  
        - **GitHub Actions / Task Scheduler**: simple, reliable automation for 30-minute refresh and weekly retraining.
        """)

        # ---------- 3) Automation & retraining ----------
        st.markdown("### Automation & Retraining")
        st.markdown("""
        - **Ingestion & scoring**: run **every 30 minutes** during India market hours (IST 08:30–15:30) → updates SmartScores & signals.  
        - **Backfill**: **nightly** (optional) to keep rolling history consistent.  
        - **Model retraining**: **weekly** (Fri 19:00 IST) or after ≥5 new trading days.  
        - **Dashboard**: auto-reads the latest CSVs; no restart needed in cloud deployments.
        """)

        # ---------- 4) Deployment (fill in your host) ----------
        st.markdown("### Deployment")
        st.markdown("""
        Deployed on: **Streamlit Cloud**  
        Pipeline auto-runs via **GitHub Actions** and commits refreshed CSVs to the repo; the app serves the newest files.
        """)

        # ---------- 5) Legal / Ethics ----------
        st.markdown("### Legal & Ethics")
        st.markdown("""
        This project uses **public RSS headlines only** (no article bodies) for **academic/personal research**.  
        Price data via `yfinance`. **No financial advice** and **no redistribution** of copyrighted content.
        """)

    with colB:
        # ---------- Key files ----------
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

        # ---------- Graphviz workflow diagram ----------
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
        Train [label="train_regression.py\n(Ridge/RF, TSSplit CV)"];
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

        st.caption("Tip: CI runs fetch→sentiment→aggregate→predict on schedule; retraining happens weekly.")


    # ---------- SmartScore explanation ----------
st.markdown("---")
with st.expander(" About SmartScore Components"):
    st.markdown("""
        **SmartScore (0–100)** is a composite index combining multiple sentiment and event-driven factors:

        - **S_recency** - measures how recent and consistent the sentiment is; fresh news has higher weight  
        - **S_events** - captures the type of news (earnings, orders, penalties, etc.) and its sentiment impact  
        - **S_breadth** - represents the ratio of positive vs. negative headlines for the stock  
        - **S_volume** - reflects total news flow (how much the stock is being discussed)

        **Formula:**  
        SmartScore = 0.45 × S_recency + 0.25 × S_events + 0.20 × S_breadth + 0.10 × S_volume

        **Interpretation:**  
        - >70 → strong positive tone and momentum  
        - 50–70 → neutral to mildly positive  
        - <50 → negative or weak sentiment tone  
        """)
