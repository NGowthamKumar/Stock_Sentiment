"""
Reads:  data/stock_sentiment_summary.csv
        models/nextday_regressor.pkl
Writes: data/predictions_nextday.csv
"""
import joblib
import pandas as pd
from src.price_labels import fetch_prices, add_forward_return

def main():
    latest = pd.read_csv("data/stock_sentiment_summary.csv")
    bundle = joblib.load("models/nextday_regressor.pkl")
    model = bundle["model"]
    features = bundle["features"]

    # Fetch lag price features if model needs them
    if "ret_lag1" in features:
        tickers = latest["ticker"].dropna().unique().tolist()
        end   = pd.Timestamp.now().strftime("%Y-%m-%d")
        start = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        
        prices = fetch_prices(tickers, start, end)
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
        prices = add_forward_return(prices, horizon_days=1)
        prices = prices.sort_values(["ticker","date"])
        prices["ret_lag1"] = prices.groupby("ticker")["ret_fwd"].shift(1)
        prices["ret_lag2"] = prices.groupby("ticker")["ret_fwd"].shift(2)

        # Take most recent row per ticker
        lag_today = prices.groupby("ticker").tail(1)[["ticker","ret_lag1","ret_lag2"]]
        latest = latest.merge(lag_today, on="ticker", how="left")


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
