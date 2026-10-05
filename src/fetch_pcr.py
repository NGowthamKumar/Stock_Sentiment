# src/fetch_pcr.py
"""
Fetches Nifty 50 Put/Call Ratio (PCR) from NSE India daily.
Uses indiaopt library which handles NSE bot protection automatically.
Saves to: data/pcr_history.csv

PCR interpretation:
> 1.2  = HIGH hedging = institutions buying puts = bearish signal
0.8-1.2 = NEUTRAL = balanced
< 0.7  = LOW hedging = complacent = often contrarian buy signal

Run after market close (18:30 IST+):
python -m src.fetch_pcr
"""
import os
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta

PCR_PATH = "data/pcr_history.csv"

async def fetch_pcr_async():
    """Fetch PCR using indiaopt library"""
    try:
        from indiaopt import NSEClient
        async with NSEClient() as client:
            result = await client.fetch_option_chain("NIFTY")
            return {
                "pcr_oi":      round(float(result.pcr), 4),
                "pcr_vol":     round(float(result.total_put_oi) / max(float(result.total_call_oi), 1), 4),
                "total_ce_oi": int(result.total_call_oi),
                "total_pe_oi": int(result.total_put_oi),
                "spot_price":  float(result.spot_price),
            }
    except ImportError:
        print("indiaopt not installed. Run: pip install indiaopt")
        return {}
    except Exception as e:
        print(f"  indiaopt fetch failed: {e}")
        return {}

def fetch_pcr_requests_fallback():
    """Fallback using requests with proper NSE session"""
    import requests
    import time

    headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         "https://www.nseindia.com/option-chain",
        "Connection":      "keep-alive",
        "sec-ch-ua":       '"Not_A Brand";v="8", "Chromium";v="120"',
        "sec-fetch-dest":  "empty",
        "sec-fetch-mode":  "cors",
        "sec-fetch-site":  "same-origin",
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        # Must visit main page first to get cookies
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(2)
        session.get("https://www.nseindia.com/option-chain", timeout=10)
        time.sleep(2)

        # Now fetch option chain
        resp = session.get(
            "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
            timeout=15
        )

        if resp.status_code == 200:
            data = resp.json()
            if "filtered" in data:
                filtered    = data["filtered"]
                total_ce_oi = filtered.get("CE", {}).get("totOI", 0)
                total_pe_oi = filtered.get("PE", {}).get("totOI", 0)
                total_ce_vol= filtered.get("CE", {}).get("totVol", 0)
                total_pe_vol= filtered.get("PE", {}).get("totVol", 0)
                pcr_oi  = round(total_pe_oi  / max(total_ce_oi,  1), 4)
                pcr_vol = round(total_pe_vol / max(total_ce_vol, 1), 4)
                return {
                    "pcr_oi":      pcr_oi,
                    "pcr_vol":     pcr_vol,
                    "total_ce_oi": total_ce_oi,
                    "total_pe_oi": total_pe_oi,
                    "spot_price":  0,
                }
        print(f"  HTTP {resp.status_code}")
    except Exception as e:
        print(f"  requests fallback failed: {e}")
    return {}

def main():
    print("=" * 50)
    print("NSE PCR FETCHER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Load existing history
    if os.path.exists(PCR_PATH):
        history = pd.read_csv(PCR_PATH, parse_dates=["date"])
        print(f"Existing PCR history: {len(history)} rows")
        latest_date = history["date"].max().date() if not history.empty else None
    else:
        history     = pd.DataFrame()
        latest_date = None
        print("No existing PCR history — starting fresh")

    today     = datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
    today_str = str(today)

    if latest_date and str(latest_date) == today_str:
        print(f"Already have PCR for {today_str} — skipping")
        if not history.empty:
            r = history.iloc[-1]
            print(f"Latest: PCR OI={r['pcr_oi']:.3f}")
        return

    print(f"\nFetching PCR for {today_str}...")

    # Try indiaopt first
    print("Trying indiaopt library...")
    pcr_data = asyncio.run(fetch_pcr_async())

    # Fallback to requests
    if not pcr_data:
        print("Trying requests fallback...")
        pcr_data = fetch_pcr_requests_fallback()

    if not pcr_data:
        print("⚠️  Both methods failed")
        print("   Possible reasons:")
        print("   1. Market is closed (weekend/holiday)")
        print("   2. NSE blocking automated requests")
        print("   3. Network issue")
        if not history.empty:
            prev = history.iloc[-1]
            pcr_data = {
                "pcr_oi":      float(prev.get("pcr_oi",  1.0)),
                "pcr_vol":     float(prev.get("pcr_vol", 1.0)),
                "total_ce_oi": 0,
                "total_pe_oi": 0,
                "spot_price":  0,
            }
            print(f"   Forward-filling previous PCR: {pcr_data['pcr_oi']:.3f}")
        else:
            print("   No previous data — skipping")
            return

    # Build new row
    new_row = {
        "date":        today_str,
        "pcr_oi":      pcr_data.get("pcr_oi",      1.0),
        "pcr_vol":     pcr_data.get("pcr_vol",      1.0),
        "total_ce_oi": pcr_data.get("total_ce_oi",  0),
        "total_pe_oi": pcr_data.get("total_pe_oi",  0),
        "spot_price":  pcr_data.get("spot_price",   0),
    }

    pcr = new_row["pcr_oi"]
    if pcr >= 1.3:   signal = "🔴 HIGH HEDGING — bearish"
    elif pcr >= 1.1: signal = "🟡 MODERATE — mild bearish"
    elif pcr >= 0.8: signal = "⚪ NEUTRAL"
    elif pcr >= 0.6: signal = "🟡 LOW HEDGING — mild bullish"
    else:            signal = "🟢 EXTREME LOW — contrarian bullish"

    print(f"\n✅ PCR fetched:")
    print(f"   PCR (OI):   {pcr:.3f} → {signal}")
    print(f"   PCR (Vol):  {new_row['pcr_vol']:.3f}")
    if new_row["spot_price"]:
        print(f"   Nifty spot: {new_row['spot_price']:,.0f}")

    # Append and save
    new_df = pd.DataFrame([new_row])
    if not history.empty:
        history = pd.concat([history, new_df], ignore_index=True)
    else:
        history = new_df

    history["date"] = pd.to_datetime(history["date"])
    history = history.drop_duplicates(subset=["date"], keep="last")
    history = history.sort_values("date").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    history.to_csv(PCR_PATH, index=False)
    print(f"\nSaved {len(history)} rows → {PCR_PATH}")

    if len(history) > 1:
        print("\nRecent PCR history:")
        print(history.tail(5)[["date","pcr_oi","pcr_vol"]].to_string(index=False))

if __name__ == "__main__":
    main()