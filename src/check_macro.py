"""Quick diagnostic to verify macro indicator values"""
import yfinance as yf
import pandas as pd
from datetime import datetime

end = pd.Timestamp.now().strftime("%Y-%m-%d")
start = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

macro_map = {
    "^INDIAVIX": "India VIX",
    "BZ=F":      "Brent Crude Oil (USD/barrel)",
    "USDINR=X":  "USD/INR Exchange Rate",
}

print(f"\n{'='*50}")
print(f"Macro Indicator Check — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*50}\n")

for ticker, name in macro_map.items():
    try:
        data = yf.download(ticker, start=start, end=end,
                          progress=False, auto_adjust=True)
        
        # Flatten MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if not data.empty:
            latest_val  = float(data["Close"].iloc[-1])
            prev_val    = float(data["Close"].iloc[-2]) if len(data) > 1 else latest_val
            pct_change  = (latest_val - prev_val) / prev_val * 100
            latest_date = data.index[-1].strftime("%Y-%m-%d")
            
            print(f"📊 {name} ({ticker})")
            print(f"   Latest date:  {latest_date}")
            print(f"   Latest value: {latest_val:.4f}")
            print(f"   Prev value:   {prev_val:.4f}")
            print(f"   Daily change: {pct_change:+.2f}%")
            print(f"   Last 5 days:")
            for date, row in data["Close"].tail(5).items():
                print(f"     {date.strftime('%Y-%m-%d')}: {float(row):.4f}")
            print()
        else:
            print(f"❌ {name} ({ticker}) — NO DATA RETURNED\n")
            
    except Exception as e:
        print(f"❌ {name} ({ticker}) — ERROR: {e}\n")

# ── FII/DII Data ──
print(f"\n\n🏦 FII/DII DATA (via NSE)")
print("-"*55)

# Check saved history file
fii_path = "data/fii_dii_history.csv"
try:
    fii_df = pd.read_csv(fii_path, parse_dates=["date"])
    print(f"\n📁 Saved FII/DII history ({fii_path}):")
    print(f"   Total rows: {len(fii_df)}")
    print(f"   Date range: {fii_df['date'].min().date()} → {fii_df['date'].max().date()}")
    print(f"\n   Last 5 entries:")
    print(fii_df.tail(5).to_string(index=False))
except Exception as e:
    print(f"   ❌ Could not read history file: {e}")

# Live fetch from NSE
print(f"\n🔴 Live fetch from NSE right now:")
try:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/reports/fii-dii",
    }
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=10)
    time.sleep(2)
    session.get("https://www.nseindia.com/reports/fii-dii", timeout=10)
    time.sleep(2)
    resp = session.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    
    fii = df[df["category"] == "FII/FPI"][["date","netValue"]].rename(columns={"netValue":"fii_net"})
    dii = df[df["category"] == "DII"][["date","netValue"]].rename(columns={"netValue":"dii_net"})
    row = fii.merge(dii, on="date")
    
    print(f"   Date:    {row['date'].values[0]}")
    print(f"   FII net: ₹{float(row['fii_net'].values[0]):,.2f} crore", 
          "✅ BUYING" if float(row['fii_net'].values[0]) > 0 else "❌ SELLING")
    print(f"   DII net: ₹{float(row['dii_net'].values[0]):,.2f} crore",
          "✅ BUYING" if float(row['dii_net'].values[0]) > 0 else "❌ SELLING")
    
except Exception as e:
    print(f"   ❌ Live NSE fetch failed: {e}")

print(f"{'='*50}")
print("Cross-check these values manually:")
print("India VIX  → https://www.nseindia.com/products-services/vix")
print("Brent Crude → https://finance.yahoo.com/quote/BZ=F")
print("USD/INR    → https://finance.yahoo.com/quote/USDINR=X")
print("FII/DII     → https://www.nseindia.com/reports/fii-dii")
print(f"{'='*50}\n")