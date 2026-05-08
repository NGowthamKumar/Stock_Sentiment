# src/fetch_fii_dii.py
"""
Fetches daily FII/DII net flow data from NSE
Writes: data/fii_dii_history.csv
"""
import os
import time
import requests
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/reports/fii-dii",
}

def fetch_fii_dii() -> pd.DataFrame | None:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=10)
    time.sleep(2)
    session.get("https://www.nseindia.com/reports/fii-dii", timeout=10)
    time.sleep(2)
    resp = session.get(
        "https://www.nseindia.com/api/fiidiiTradeReact",
        timeout=10
    )
    resp.raise_for_status()
    return pd.DataFrame(resp.json())

def main():
    os.makedirs("data", exist_ok=True)
    out_path = "data/fii_dii_history.csv"

    try:
        df = fetch_fii_dii()

        # Pivot to one row per date: fii_net, dii_net
        fii = df[df["category"] == "FII/FPI"][["date","netValue"]].rename(columns={"netValue":"fii_net"})
        dii = df[df["category"] == "DII"][["date","netValue"]].rename(columns={"netValue":"dii_net"})
        row = fii.merge(dii, on="date")
        row["date"] = pd.to_datetime(row["date"], format="%d-%b-%Y")

        print(row)

        if os.path.exists(out_path):
            # Avoid duplicate for same date
            existing = pd.read_csv(out_path, parse_dates=["date"])
            combined = pd.concat([existing, row]).drop_duplicates(subset=["date"], keep="last")
            combined.to_csv(out_path, index=False)
        else:
            row.to_csv(out_path, index=False)

        print(f"Saved FII/DII data → {out_path}")

    except Exception as e:
        print(f"FII/DII fetch failed: {e}")

if __name__ == "__main__":
    main()