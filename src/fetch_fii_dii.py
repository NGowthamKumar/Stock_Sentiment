# src/backfill_fii_dii.py
"""
Backfill FII/DII from NSE market activity page
Uses a different NSE endpoint that has more history
"""
import requests
import pandas as pd
import time
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/market-data/live-market-indices",
}

def fetch_historical_fii():
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # Prime cookies
    session.get("https://www.nseindia.com", timeout=15)
    time.sleep(2)
    
    # Try the historical endpoint
    urls_to_try = [
        "https://www.nseindia.com/api/fiidiiTradeReact?type=historical",
        "https://www.nseindia.com/api/historicalFiiDii",
        "https://www.nseindia.com/api/fii-dii-monthly",
    ]
    
    for url in urls_to_try:
        try:
            print(f"Trying: {url}")
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                print(f"Got {len(data)} records from {url}")
                print(f"Sample: {data[:2]}")
                return pd.DataFrame(data)
        except Exception as e:
            print(f"  Failed: {e}")
        time.sleep(2)
    
    return pd.DataFrame()

def main():
    os.makedirs("data", exist_ok=True)
    out_path = "data/fii_dii_history.csv"

    try:
        df = fetch_historical_fii()
        
        if df.empty:
            print("No data fetched")
            return

        fii = df[df["category"] == "FII/FPI"][["date","netValue"]].rename(
            columns={"netValue": "fii_net"}
        )
        dii = df[df["category"] == "DII"][["date","netValue"]].rename(
            columns={"netValue": "dii_net"}
        )
        row = fii.merge(dii, on="date")
        row["date"] = pd.to_datetime(
            row["date"], format="%d-%b-%Y"
        ).dt.strftime("%Y-%m-%d")
        row["fii_net"] = pd.to_numeric(row["fii_net"], errors="coerce")
        row["dii_net"] = pd.to_numeric(row["dii_net"], errors="coerce")

        print(f"New row: {row.to_dict('records')}")

        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            existing["date"] = existing["date"].astype(str)
            row["date"] = row["date"].astype(str)
            combined = pd.concat([existing, row]).drop_duplicates(
                subset=["date"], keep="last"
            ).sort_values("date")
            combined.to_csv(out_path, index=False)
            print(f"Saved {len(combined)} rows → {out_path}")
        else:
            row.to_csv(out_path, index=False)
            print(f"Saved {len(row)} rows → {out_path}")

    except Exception as e:
        print(f"FII/DII fetch failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()