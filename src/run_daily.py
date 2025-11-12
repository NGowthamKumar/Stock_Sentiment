"""
Runs the full sentiment update pipeline:
1. Fetch latest news
2. Run sentiment analysis
3. Aggregate results
4. Launch dashboard (optional)
"""

import os
from datetime import datetime
from pathlib import Path

# ensure working directory is project root
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

print(f"🚀 Starting full update: {datetime.now():%Y-%m-%d %H:%M:%S}")

# sequential pipeline
os.system("python -m src.fetch_news")
os.system("python -m src.sentiment_vader")
os.system("python -m src.aggregate_sentiment")

# optional — build dataset, retrain, predict
os.system("python -m src.build_dataset")
os.system("python -m src.train_regression")
os.system("python -m src.predict_next")

# finally launch dashboard
os.system("streamlit run dashboard/app.py")

print(f"✅ Pipeline complete at {datetime.now():%H:%M:%S}")
