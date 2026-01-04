# Indian Stock Sentiment Analyser

A real-time **Indian equity news sentiment and prediction system** that ingests financial headlines, applies NLP-based sentiment analysis, aggregates signals into a custom **SmartScore**, and predicts **next-day stock returns** using machine learning.

This project was built for **learning, experimentation, and applied ML/NLP practice**, with a fully automated data pipeline and an interactive dashboard.

**Live Dashboard:**  
https://indianstocksentiment.streamlit.app/

---

## Overview

The Indian Stock Sentiment Analyser continuously tracks **public financial news** for selected NSE-listed stocks and converts unstructured headlines into quantitative signals.

The system:
- Scores news sentiment using a **hybrid NLP approach**
- Detects **event types** such as earnings, M&A, litigation, and regulatory updates
- Aggregates signals into a **SmartScore (0–100)**
- Trains ML models to **predict next-day percentage returns**
- Visualizes everything in a **Streamlit + Plotly dashboard**

All pipelines are **fully automated** using GitHub Actions.

---

## Features

### News & NLP
- Fetches live headlines from:
  - Google News
  - Moneycontrol
  - ET Markets
  - Investing.com
- Cleans, normalizes, and de-duplicates headlines
- Maps news to NSE tickers using alias-based entity matching
- Applies a **hybrid sentiment model**:
  - FinBERT (transformer-based financial sentiment)
  - VADER (lexicon-based sentiment)

### Event Intelligence
- Classifies headlines into event categories:
  - Earnings
  - Guidance
  - M&A
  - Litigation
  - Regulatory updates
  - Management changes
  - Order wins / product launches
- Assigns signed weights based on event impact

### SmartScore Engine
- Aggregates sentiment using:
  - **Recency (EWMA decay)**
  - **Event impact**
  - **Breadth (positive vs negative coverage)**
  - **News volume**
- Produces a **SmartScore (0–100)** per stock

### Machine Learning
- Builds a daily ML dataset by joining:
  - SmartScore features
  - Historical prices (via yfinance)
- Trains:
  - Ridge Regression
  - RandomForest Regressor
- Uses **TimeSeriesSplit** to avoid lookahead bias
- Evaluates using:
  - MAE
  - Direction Accuracy
  - R²
  - Spearman Correlation
- Predicts **next-day percentage returns**

### Dashboard
- Interactive Streamlit dashboard with:
  - Portfolio overview
  - SmartScore breakdown
  - Stock-level drilldowns
  - Prediction filtering
  - Model health monitoring

---

## System Workflow

1. **News Ingestion**  
   RSS headlines are fetched, normalized, timestamped, and mapped to NSE tickers.

2. **Sentiment Analysis**  
   Headlines are scored using FinBERT and VADER, then combined into an ensemble score.

3. **Event Classification**  
   Rule-based classification assigns event types and impact weights.

4. **Aggregation**  
   Signals are aggregated into SmartScore components:
   - Recency
   - Events
   - Breadth
   - Volume

5. **Model Training**  
   SmartScores are joined with market prices to train next-day return models.

6. **Prediction & Visualization**  
   Predictions and metrics are rendered in the Streamlit dashboard.

---

## Project Structure
STOCK_SENTIMENT/
│
├─ .github/workflows/
│   ├─ daily_ingest.yml        # Scheduled news ingestion & scoring
│   └─ weekly_train.yml        # Weekly model retraining
│
├─ dashboard/
│   └─ app.py                  # Streamlit dashboard
│
├─ data/
│   ├─ history/                # Historical SmartScore snapshots
│   ├─ modeling/               # Model metrics & training logs
│   ├─ raw_news.csv
│   ├─ processed_sentiment.csv
│   ├─ stock_sentiment_summary.csv
│   ├─ predictions_nextday.csv
│   └─ stocks.yml              # Tracked tickers
│
├─ models/
│   └─ nextday_regressor.pkl   # Best trained model
│
├─ src/
│ ├─ aggregate_sentiment.py # SmartScore computation logic
│ ├─ backfill_history.py # Historical SmartScore reconstruction
│ ├─ build_dataset.py # Feature + label dataset creation
│ ├─ config.py # Central configuration
│ ├─ entity_map.py # Ticker alias & entity resolution
│ ├─ event_classifier.py # Rule-based event classification
│ ├─ fetch_news.py # RSS ingestion & normalization
│ ├─ predict_next.py # Next-day return prediction
│ ├─ price_labels.py # Price fetching & forward returns
│ ├─ run_daily.py # Full daily pipeline runner
│ ├─ sentiment_vader.py # FinBERT + VADER ensemble sentiment
│ ├─ train_regression.py # ML training & evaluation
│ └─ utils_text.py # Text cleaning utilities
│
├─ config.yml
└─ requirements.txt

---

## Automation & Retraining

- **News ingestion & sentiment scoring:** every 30 minutes
- **SmartScore updates:** daily
- **Model retraining:** weekly (via GitHub Actions)
- **Dashboard updates:** automatic

---

## Installation & Local Setup

```bash
git clone https://github.com/yourusername/indian-stock-sentiment
cd indian-stock-sentiment
pip install -r requirements.txt
python src/run_daily.py
```

---

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Transformers (FinBERT)
- NLTK (VADER)
- Streamlit
- Plotly
- yfinance
- GitHub Actions

---

## Disclaimer

This project is intended for learning and research purposes only.  
It does **not** constitute financial advice or investment recommendations.

