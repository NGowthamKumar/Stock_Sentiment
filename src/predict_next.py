"""
Reads:  data/stock_sentiment_summary.csv
        models/nextday_regressor.pkl
Writes: data/predictions_nextday.csv
"""
import joblib
import pandas as pd

def main():
    latest = pd.read_csv("data/stock_sentiment_summary.csv")
    bundle = joblib.load("models/nextday_regressor.pkl")
    model = bundle["model"]
    features = bundle["features"]

    X = latest[["ticker", *features]].dropna()
    if X.empty:
        raise SystemExit("No rows with full features in latest snapshot.")
    preds = model.predict(X[features])

    out = X[["ticker"]].copy()
    out["pred_ret_1d_pct"] = preds
    out = out.sort_values("pred_ret_1d_pct", ascending=False)
    out.to_csv("data/predictions_nextday.csv", index=False)
    print("Wrote predictions → data/predictions_nextday.csv")
    print(out.head(10))

if __name__ == "__main__":
    main()
