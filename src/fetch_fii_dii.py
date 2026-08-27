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
    df = fetch_historical_fii()
    if not df.empty:
        print(df.head())

if __name__ == "__main__":
    main()