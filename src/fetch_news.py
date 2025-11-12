# src/fetch_news.py
"""
Fetch latest Indian stock-market headlines from:
  1) Google News (per-stock query)
  2) Moneycontrol (RSS)
  3) Economic Times - Markets (RSS)
  4) Investing.com India (RSS)

Outputs a deduplicated CSV with:
  source_name, source_domain, title, link, published_utc, news_id, ticker, map_confidence

Saves: data/raw_news.csv
"""

import re
import hashlib
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from datetime import timezone

import feedparser
import pandas as pd
from dateutil import parser as dparser
from tqdm import tqdm


# ---------------------------
# ✅ Portfolio (used for Google queries)
# ---------------------------
STOCKS_FOR_GOOGLE = [
    "HCL Tech", "Reliance Industries", "Vedanta", "Bansal Wires Industries",
    "Nippon India ETF Gold Bees", "Niftybees", "UPL", "HDFC Bank",
    "Jio Financial Services", "Coal India", "Tata Steel", "Nippon India Silver Bees",
    "IRFC", "Tata Elxsi", "Infosys", "NMDC", "Bharat Electronics",
    "Adani Energy Solutions", "MMTC", "NHPC", "Nestle",
    "Tata Motors Passenger Vehicles", "Nippon India EFT IT",
    "Sun TV Network", "Reliance Power", "Delta Corp", "PNB",
    "Yes Bank", "ITC", "IndusInd Bank", "Sail", "ONGC", "EaseMyTrip",
    "BHEL", "BLS International Services", "Natco Pharma", "NBCC(India)",
    "IEX", "Tata Technologies", "Indian Overseas Bank", "SJVN", "SpiceJet",
]

# ---------------------------
# ✅ Alias → NSE ticker map (extend as needed)
# ---------------------------
ALIAS_TO_TICKER = {
    # Large caps / common names
    "hcl tech": "HCLTECH.NS",
    "hcl technologies": "HCLTECH.NS",
    "reliance industries": "RELIANCE.NS",
    "reliance": "RELIANCE.NS",
    "infosys": "INFY.NS",
    "coal india": "COALINDIA.NS",
    "tata steel": "TATASTEEL.NS",
    "hdfc bank": "HDFCBANK.NS",
    "tata elxsi": "TATAELXSI.NS",
    "nmdc": "NMDC.NS",
    "bharat electronics": "BEL.NS",
    # Adani Energy Solutions current symbol:
    "adani energy solutions": "ADANIENSOL.NS",
    "mmtc": "MMTC.NS",
    "nhpc": "NHPC.NS",
    "nestle": "NESTLEIND.NS",
    "tata motors": "TATAMOTORS.NS",
    "sun tv network": "SUNTV.NS",
    "reliance power": "RPOWER.NS",
    "delta corp": "DELTACORP.NS",
    "pnb": "PNB.NS",
    "punjab national bank": "PNB.NS",
    "yes bank": "YESBANK.NS",
    "itc": "ITC.NS",
    "indusind bank": "INDUSINDBK.NS",
    "sail": "SAIL.NS",
    "ongc": "ONGC.NS",
    "bhel": "BHEL.NS",
    "natco pharma": "NATCOPHARM.NS",
    "nbcc": "NBCC.NS",
    "iex": "IEX.NS",
    "tata technologies": "TATATECH.NS",
    "indian overseas bank": "IOB.NS",
    "sjvn": "SJVN.NS",
    "spicejet": "SPICEJET.NS",
    "upl": "UPL.NS",
    "easemytrip": "EASEMYTRIP.NS",
    "bls international": "BLS.NS",
    "irfc": "IRFC.NS",
    "vedanta": "VEDL.NS",
    "jio financial services": "JIOFIN.NS",
    # ETFs / BEES
    "niftybees": "NIFTYBEES.NS",
    "nippon india etf gold bees": "GOLDBEES.NS",
    "nippon india silver bees": "SILVERBEES.NS",
    "nippon india eft it": "ITBEES.NS",   # spelling in your list
    "nippon india etf it": "ITBEES.NS",
    # New / corrected listings:
    "bansal wires": "BANSALWIRE.NS",
    # NOTE: OLA not included until official NSE symbol is confirmed
}

# Precompile alias regex matchers (word/sep boundaries)
ALIAS_REGEX = [
    (re.compile(rf"(?:^|[^A-Za-z0-9]){re.escape(alias)}(?:[^A-Za-z0-9]|$)", re.I), ticker)
    for alias, ticker in ALIAS_TO_TICKER.items()
]


# ---------------------------
# ✅ Sources
# ---------------------------
def build_sources() -> dict:
    sources = {}
    # Google News per stock
    for name in STOCKS_FOR_GOOGLE:
        q = f"{name.replace(' ', '+')}+stock+India"
        sources[f"Google_{name}"] = f"https://news.google.com/rss/search?q={q}"
    # Finance portals
    sources["MoneyControl"] = "https://www.moneycontrol.com/rss/MCtopnews.xml"
    sources["EconomicTimes_Markets"] = "https://economictimes.indiatimes.com/rssfeeds/2146842.cms"
    sources["Investing_India"] = "https://in.investing.com/rss/"
    return sources


# ---------------------------
# 🔧 Helpers: URL clean, date parse, id, mapping
# ---------------------------
def normalize_url(u: str) -> str:
    if not u:
        return ""
    try:
        p = urlparse(u)
        # strip UTM/tracking, drop fragment
        q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm")]
        p = p._replace(query=urlencode(q, doseq=True), fragment="")
        return urlunparse(p)
    except Exception:
        return u

def domain(u: str) -> str:
    try:
        return urlparse(u).netloc.lower().replace("www.", "")
    except Exception:
        return ""

def parse_date_any(s: str | None):
    if not s:
        return None
    try:
        dt = dparser.parse(s)
        if not dt.tzinfo:
            dt = dt.tz_localize("UTC") if hasattr(dt, "tz_localize") else dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def stable_news_id(title: str, link: str, published_dt) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip().lower())
    d = domain(link or "")
    day = published_dt.strftime("%Y%m%d") if published_dt is not None else "nodate"
    return sha1(f"{t}|{d}|{day}")

def map_ticker(title: str, source_name: str):
    """
    Returns (ticker or None, confidence 0..1).
    1) Alias match in title
    2) If Google_{stock}, map by the stock portion of source_name
    """
    txt = (title or "").lower()

    # 1) Alias-by-title
    for rx, tk in ALIAS_REGEX:
        if rx.search(txt):
            return tk, 0.9

    # 2) Google_{stock} fallback
    if source_name.startswith("Google_"):
        stock_name = source_name.split("_", 1)[1].strip().lower()
        stock_name = re.sub(r"[\(\)]", "", stock_name)
        for alias, tk in ALIAS_TO_TICKER.items():
            if alias in stock_name:
                return tk, 0.6

    return None, 0.0


# ---------------------------
# ✅ RSS fetcher
# ---------------------------
def fetch_rss(url: str) -> list[dict]:
    feed = feedparser.parse(url)
    out = []
    for e in getattr(feed, "entries", []):
        out.append({
            "title": getattr(e, "title", None),
            "link": getattr(e, "link", None),
            "published_raw": getattr(e, "published", None) or getattr(e, "updated", None),
        })
    return out


# ---------------------------
# ✅ Main
# ---------------------------
def main():
    sources = build_sources()
    all_rows = []
    print(f"📰 Starting news fetch from {len(sources)} sources...\n")

    for name, url in tqdm(sources.items()):
        try:
            for it in fetch_rss(url):
                link_norm = normalize_url(it["link"] or "")
                pub_dt = parse_date_any(it["published_raw"])
                pub_iso = pub_dt.isoformat() if pub_dt is not None else None
                nid = stable_news_id(it["title"] or "", link_norm, pub_dt)
                tk, conf = map_ticker(it["title"] or "", name)
                all_rows.append({
                    "source_name": name,
                    "source_domain": domain(link_norm),
                    "title": it["title"],
                    "link": link_norm,
                    "published_utc": pub_iso,
                    "news_id": nid,
                    "ticker": tk,
                    "map_confidence": conf,
                })
        except Exception as e:
            print(f"⚠️ {name} error: {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("❌ No news fetched.")
        return

    before = len(df)
    df = df.drop_duplicates(subset=["news_id"]).sort_values("published_utc", na_position="last")
    after = len(df)

    out = "data/raw_news.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n✅ Saved {after} deduped items to {out} (dropped {before - after} dups)")
    print("\n🧾 Sample:")
    print(df.head(8)[["source_name", "source_domain", "ticker", "map_confidence", "published_utc", "title"]])


if __name__ == "__main__":
    main()
