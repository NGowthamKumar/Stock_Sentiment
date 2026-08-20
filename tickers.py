# test_tickers.py
import yfinance as yf

new_tickers = [
    # Cement
    "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "ULTRACEMCO.NS",
    # Consumer/FMCG
    "HAVELLS.NS", "PIDILITIND.NS", "BERGEPAINT.NS", "MARICO.NS",
    "COLPAL.NS", "BRITANNIA.NS", "TATACONSUM.NS", "EMAMILTD.NS",
    "VBL.NS", "UBL.NS", "RADICO.NS", "JYOTHYLAB.NS",
    # Real Estate
    "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS",
    # Finance
    "MUTHOOTFIN.NS", "CHOLAFIN.NS", "SBICARD.NS",
    "HDFCAMC.NS", "NAM-INDIA.NS", "ANGELONE.NS", "BSE.NS", "CAMS.NS",
    # Travel/Hospitality
    "INDHOTEL.NS", "INDIGO.NS", "IRCTC.NS",
    # Pharma
    "TORNTPHARM.NS", "LUPIN.NS", "BIOCON.NS", "AUROPHARMA.NS",
    "MANKIND.NS", "ZYDUSLIFE.NS", "ALKEM.NS", "IPCALAB.NS",
    "AJANTPHARM.NS", "LAURUSLABS.NS", "GRANULES.NS",
    # Energy/Power
    "ADANIPOWER.NS", "TORNTPOWER.NS", "JPPOWER.NS", "SUZLON.NS",
    # Industrial/Capital Goods
    "CUMMINSIND.NS", "ABB.NS", "SIEMENS.NS", "BHARATFORG.NS", "SOLARINDS.NS",
    # Auto/Tyres
    "MRF.NS", "APOLLOTYRE.NS", "BALKRISIND.NS", "TIINDIA.NS",
    # Consumer Durables
    "VOLTAS.NS", "BLUESTARCO.NS", "PAGEIND.NS", "DIXON.NS", "AMBER.NS",
    # IT/Tech
    "PERSISTENT.NS", "COFORGE.NS", "LTIM.NS", "KPITTECH.NS",
    "KAYNES.NS", "DATAPATTNS.NS",
    # Defence
    "HAL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "GRSE.NS",
    # Chemicals
    "SRF.NS", "AARTIIND.NS", "DEEPAKNTR.NS", "NAVINFLUOR.NS",
    # New age/Others
    "NYKAA.NS", "IDEA.NS", "HUDCO.NS", "SHYAMMETL.NS", "APLAPOLLO.NS",
]

print(f"Testing {len(new_tickers)} tickers...\n")
working = []
failed = []

for ticker in new_tickers:
    try:
        data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if not data.empty:
            working.append(ticker)
            print(f"✅ {ticker}")
        else:
            failed.append(ticker)
            print(f"❌ {ticker} — empty data")
    except Exception as e:
        failed.append(ticker)
        print(f"❌ {ticker} — {e}")

print(f"\n{'='*50}")
print(f"✅ Working: {len(working)}/{len(new_tickers)}")
for w in working:
    print(f"   {w}")
print(f"\n❌ Failed: {len(failed)}")
for f in failed:
    print(f"   {f}")
print(f"{'='*50}")