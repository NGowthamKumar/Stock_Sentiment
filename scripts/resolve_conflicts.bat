@echo off
echo Resolving merge conflicts — keeping local data files...

git checkout --ours data/history/stock_sentiment_summary_history.csv
git checkout --ours data/modeling/dataset.parquet
git checkout --ours data/modeling/model_metrics.csv
git checkout --ours data/predictions_nextday.csv
git checkout --ours data/processed_sentiment.csv
git checkout --ours data/raw_news.csv
git checkout --ours data/stock_sentiment_summary.csv
git checkout --ours data/ensemble_signals.csv
git checkout --ours data/xgb_signals.csv
git checkout --ours data/fii_dii_history.csv
git checkout --ours models/nextday_regressor.pkl
git checkout --ours models/voting_ensemble.pkl
git checkout --ours models/xgb_classifier.pkl

git add .
git commit -m "resolved merge conflicts to keep local data"
git push origin main

echo Done!