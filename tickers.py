import yfinance as yf
tickers = ['BLUESTARCO.NS', 'BLUESTAR.NS', 'SHYAMMETL.NS', 'SHYAMMET.NS', 'GRSE.NS', 'GARDENREACH.NS']
for t in tickers:
    d = yf.download(t, period='5d', progress=False)
    print(t, '✅' if not d.empty else '❌')
