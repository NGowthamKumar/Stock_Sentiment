# test_rss_new.py
import feedparser
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

def fetch_feed(url):
    feed = feedparser.parse(url)
    if feed.entries:
        return feed, "direct"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            feed = feedparser.parse(resp.content)
            if feed.entries:
                return feed, "headers"
    except:
        pass
    return None, "failed"

new_sources = {
    "TheHindu_Business":        "https://thehindu.com/business/feeder/default.rss",
"TheHindu_Markets":         "https://www.thehindu.com/business/markets/?service=rss",

"IndianExpress_Business":   "https://indianexpress.com/section/business/feed/",
"IndianExpress_Companies":  "https://indianexpress.com/section/business/companies/feed/",
"IndianExpress_Market":     "https://indianexpress.com/section/business/market/feed/",
"IndianExpress_Economy":    "https://indianexpress.com/section/business/economy/feed/",

"IndiaToday_Business":      "https://www.indiatoday.in/rss/1206574",

"Firstpost_Business":       "https://www.firstpost.com/commonfeeds/v1/mfp/rss/business.xml",

"BusinessLine_Home":        "https://www.thehindubusinessline.com/feeder/default.rss",

"NSE_Corporate":            "https://www.nseindia.com/static/rss/rss_corporate.xml",
}

print(f"Testing {len(new_sources)} new sources...\n")
working = []
failed = []

for name, url in new_sources.items():
    feed, method = fetch_feed(url)
    if feed and feed.entries:
        entries = len(feed.entries)
        first = getattr(feed.entries[0], "title", "no title")
        working.append((name, url, method))
        print(f"✅ {name}: {entries} entries [{method}]")
        print(f"   Sample: {first[:70]}...")
    else:
        failed.append((name, url))
        print(f"❌ {name}: 0 entries")
    print()

print(f"{'='*60}")
print(f"✅ Working: {len(working)}")
for w, url, m in working:
    print(f"   {w} [{m}]")
print(f"\n❌ Failed: {len(failed)}")
for f, url in failed:
    print(f"   {f}")
print(f"{'='*60}")