# test_rss.py
import feedparser
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

def fetch_feed(url):
    """Try feedparser first, fall back to requests with browser headers"""
    # Try direct feedparser
    feed = feedparser.parse(url)
    if feed.entries:
        return feed, "direct"
    
    # Try with browser headers via requests
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            feed = feedparser.parse(resp.content)
            if feed.entries:
                return feed, "requests+headers"
    except Exception as e:
        pass
    
    return None, "failed"

sources = {
    # Already confirmed working
    "BusinessStandard_Latest":  "https://www.business-standard.com/rss/latest.rss",
    "Livemint_Markets":         "https://www.livemint.com/rss/markets",
    "BusinessLine_Markets":     "https://www.thehindubusinessline.com/markets/?service=rss",
    "NDTVProfit_Markets":       "https://feeds.feedburner.com/ndtvprofit-latest",
    # CNBCTV18 alternatives
    "CNBCTV18_v1": "https://www.cnbctv18.com/commonfeeds/v1/eng/rss/market.xml",
    "CNBCTV18_v2": "https://www.cnbctv18.com/commonfeeds/v1/eng/rss/news.xml",
    "CNBCTV18_v3": "https://www.cnbctv18.com/commonfeeds/v1/eng/rss/business.xml",
    # Financial Express alternatives
    "FinancialExpress_v1": "https://www.financialexpress.com/market/feed/",
    "FinancialExpress_v2": "https://www.financialexpress.com/feed/",
    "FinancialExpress_v3": "https://www.financialexpress.com/market/stock-market/feed/",
}

print(f"\n{'='*60}")
print("RSS Feed URL Test Results (with browser headers fallback)")
print(f"{'='*60}\n")

working = []
broken = []

for name, url in sources.items():
    feed, method = fetch_feed(url)
    if feed and feed.entries:
        entries = len(feed.entries)
        first = getattr(feed.entries[0], "title", "no title")
        working.append((name, method))
        print(f"✅ {name}: {entries} entries [{method}]")
        print(f"   URL: {url}")
        print(f"   Sample: {first[:60]}...")
    else:
        broken.append(name)
        print(f"❌ {name}: 0 entries")
        print(f"   URL: {url}")
    print()

print(f"{'='*60}")
print(f"✅ Working: {len(working)}")
for w, m in working:
    print(f"   - {w} [{m}]")
print(f"\n❌ Broken: {len(broken)}")
for b in broken:
    print(f"   - {b}")
print(f"{'='*60}\n")