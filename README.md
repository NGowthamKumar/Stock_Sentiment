# 🇮🇳 Indian Stock Sentiment Analyser

A production-grade automated market intelligence system for NSE-listed equities — combining financial NLP, macroeconomic signals, institutional flow data, and machine learning to generate next-day and 3-day directional signals.

**Live Dashboard:** [indianstocksentiment.streamlit.app](https://indianstocksentiment.streamlit.app/)

---

## Overview

This system continuously ingests public financial news, applies transformer-based NLP to extract sentiment and event signals, integrates macroeconomic indicators and institutional flow data, and trains a multi-model ensemble to predict stock return directions.

All pipelines are fully automated using GitHub Actions — news is fetched every 30 minutes, models retrain daily, and the dashboard updates automatically.

---

## System Architecture

```
318 RSS Sources (Google News, BusinessStandard, ET Markets, Mint, BusinessLine...)
        ↓
fetch_news.py — parallel fetch, alias-based ticker mapping, global keyword routing
        ↓
sentiment_vader.py — FinBERT + VADER ensemble sentiment, event classification
        ↓
aggregate_sentiment.py — SmartScore (dual EWMA decay, Z-score normalization)
        ↓
build_dataset.py — 24-feature dataset (SmartScore + macro + FII/DII + technical)
        ↓
train_regression.py — 7-model ensemble, TimeSeriesSplit, baseline comparison
        ↓
predict_next.py — directional signals + macro regime filter
        ↓
dashboard/app.py — Market Intelligence Brief (Streamlit + Plotly)
```

---

## Core Components

### News Pipeline
- Fetches live headlines from **318 RSS sources** across major Indian financial publications
- Parallel fetch with rate limiting and source reliability weights
- Maps headlines to NSE tickers using **242 alias patterns**
- **Global keyword routing**: macro/geopolitical news routed to affected sectors (e.g. Iran/Hormuz → energy stocks; Fed rate news → banking and IT)
- Tiered recency: recent news weighted higher than older news

### NLP Sentiment Engine
- **FinBERT** (financial domain transformer) + **VADER** (lexicon-based) ensemble
- Price movement override: explicit price movements in headlines override NLP score
- **Event classification**: EARNINGS, GUIDANCE, M&A, LITIGATION, REGULATORY, MGMT_CHANGE, ORDER_WIN, PRODUCT_LAUNCH, MACRO
- Source reliability and model confidence factored into final weights

### SmartScore Engine
Composite sentiment score (0–100) per stock:

```
SmartScore = 0.45 × S_recency + 0.25 × S_events + 0.20 × S_breadth + 0.10 × S_volume
```

- **S_recency**: Dual EWMA — shorter half-life for 1-day predictions, longer for 3-day
- **S_events**: Event type weighted by impact direction and magnitude
- **S_breadth**: Positive vs negative article ratio (cross-sectional Z-score normalized)
- **S_volume**: News volume signal (cross-sectional Z-score normalized)

### Feature Engineering
24 features across five categories:

| Category | Features |
|---|---|
| SmartScore components | smart_score, S_recency, S_recency_3d, S_events, S_breadth, S_volume, pos, neg, total |
| Price momentum | ret_lag1, ret_lag2 |
| Technical indicators | rsi, macd_diff, bb_pct, bb_width, price_vs_sma (shifted 1 day — no leakage) |
| Institutional flows | fii_net, dii_net (NSE FII/DII daily) |
| Macroeconomic | India VIX, crude oil, USD/INR, US VIX, Nifty sectors, US 10yr yield, gold + daily change features |

### Machine Learning
Seven-model ensemble trained with `TimeSeriesSplit` to prevent temporal data leakage:

- Ridge Regression
- RandomForest Regressor
- XGBoost Classifier (scale_pos_weight balanced)
- LightGBM Classifier
- RandomForest Classifier (class_weight balanced)
- Voting Ensemble 1-day (soft voting)
- Voting Ensemble 3-day (soft voting)

Training includes naive baseline comparison (Always-UP, Always-DOWN, Prior-Day-Direction) and statistical significance testing (binomial test + Wilson confidence intervals).

### Macro Regime Filter
Computes a macro headwind score from US/India VIX, crude oil, yield levels, FII flows, and Fed/RBI news volume. Suppresses weak signals during hostile macro environments to reduce false positives.

### Signal Tracking
Per-stock accuracy tracked daily after market settlement — direction accuracy, win rate, streaks, and a composite Trust Score per stock.

---

## Dashboard — Market Intelligence Brief

- **Macro Headwind Score** — composite risk score with detected drivers
- **Live Macro Data** — crude oil, FII/DII, gold, VIX, yields, Nifty, S&P 500
- **Sector Heatmap** — IT, Banking, FMCG, Energy, Pharma, Defence signals
- **High-Trust Signals** — top stocks by Trust Score with expandable details
- **Weak Signals** — stocks with negative sentiment and weak ML signals
- **News Volume by Topic** — article counts (Fed, RBI, crude, Iran, etc.)
- **Geopolitical Risk Monitor** — Iran/Hormuz, Fed, Japan, de-dollarisation
- **Tomorrow's Outlook** — forward-looking signals from current trend data

---

## Automation

| Workflow | Schedule | Action |
|---|---|---|
| `daily_ingest.yml` | Every 30 min during market hours | Fetch → sentiment → SmartScore → signals → settle |
| `weekly_train.yml` | Mon–Fri 19:30 IST | Build dataset → train models → save metrics |

---

## Project Structure

```
Stock_Sentiment/
├── .github/
│   └── workflows/
│       ├── daily_ingest.yml        # News ingestion & scoring
│       └── weekly_train.yml        # Model retraining
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── data/
│   ├── history/                    # Daily SmartScore snapshots
│   ├── modeling/                   # Dataset and model metrics
│   ├── fii_dii_history.csv         # Daily institutional flows
│   ├── signal_history.csv          # Per-stock prediction history
│   ├── stock_accuracy.csv          # Per-stock accuracy and trust
│   ├── stock_streaks.csv           # Win rates and streaks
│   └── stocks.yml                  # Tracked tickers config
├── models/
│   ├── xgb_classifier.pkl
│   ├── voting_ensemble.pkl
│   ├── xgb_3d_classifier.pkl
│   ├── voting_3d_ensemble.pkl
│   └── nextday_regressor.pkl
├── scripts/
│   └── resolve_conflicts.bat       # Git conflict resolution helper
├── src/
│   ├── aggregate_sentiment.py      # SmartScore computation
│   ├── backfill_history.py         # Historical SmartScore rebuild
│   ├── build_dataset.py            # Feature dataset builder
│   ├── event_classifier.py         # Event type classification
│   ├── fetch_fii_dii.py            # NSE institutional flow fetcher
│   ├── fetch_news.py               # 318-source RSS pipeline
│   ├── keyword_sector_router.py    # Global keyword routing
│   ├── predict_next.py             # Signal generation
│   ├── price_labels.py             # Price fetching and forward returns
│   ├── run_daily.py                # Local full pipeline runner
│   ├── sentiment_vader.py          # FinBERT + VADER ensemble
│   ├── settle_signals.py           # Settlement and accuracy tracking
│   ├── train_regression.py         # Model training and evaluation
│   └── utils_text.py               # Text cleaning utilities
├── config.yml
└── requirements.txt
```

---

## Local Setup

```bash
git clone https://github.com/NGowthamKumar/Stock_Sentiment.git
cd Stock_Sentiment
pip install -r requirements.txt
python src/run_daily.py
streamlit run dashboard/app.py
```

---

## Technologies

| Layer | Stack |
|---|---|
| NLP | FinBERT (Hugging Face Transformers), VADER (NLTK) |
| Machine Learning | XGBoost, LightGBM, Scikit-learn |
| Data | Pandas, NumPy, yfinance, feedparser |
| Dashboard | Streamlit, Plotly |
| Automation | GitHub Actions |
| Statistics | SciPy |

---

## Disclaimer

This project is built for **learning, research, and portfolio demonstration** purposes only.
It does not constitute financial advice or investment recommendations.
All signals are research indicators — not a basis for financial decisions.
Always verify independently before making any investment decisions.
