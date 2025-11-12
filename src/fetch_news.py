# src/fetch_news.py
"""
Fetch latest Indian stock-market headlines from:
  1) Google News (per-stock query)
  2) Moneycontrol (RSS)
  3) Economic Times - Markets (RSS)
  4) Investing.com India (RSS)

Polite behaviours:
  - Custom User-Agent
  - Retry with exponential backoff
  - 0.6–1.2s pacing between sources

Cleans & normalizes:
  - Canonical titles (lowercase, debracket, squash spaces)
  - URL normalization (remove UTM, fragments)
  - UTC timestamps
  - Stable news_id hash

Outputs (CSV): data/raw_news.csv
  columns: source_name, source_domain, title, link, published_utc,
           news_id, ticker, map_confidence, title_canon
"""

from __future__ import annotations
import os
import re
import time
import random
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import feedparser
import pandas as pd

# ---------------------------
# Config
# ---------------------------
USER_AGENT = "IndianStockSentiment/1.0 (+https://github.com/NGowthamKumar)"
SLEEP_MIN, SLEEP_MAX = 0.6, 1.2   # pacing between sources
RETRIES = 3
BACKOFF_BASE = 1.5                # 1.0, 2.5, 4.25 ... + jitter

# ---------------------------
# Portfolio for Google queries
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
# Alias → NSE ticker map (extend as needed)
#    (regex patterns for robustness)
# ---------------------------
ALIAS_TO_TICKER_PATTERNS = [
    (r"\bHCL\s*Tech(?:nologies)?\b",       "HCLTECH.NS"),
    (r"\bReliance(?:\s+Industries)?\b",    "RELIANCE.NS"),
    (r"\bInfosys\b",                        "INFY.NS"),
    (r"\bCoal\s*India\b",                   "COALINDIA.NS"),
    (r"\bTata\s*Steel\b",                   "TATASTEEL.NS"),
    (r"\bHDFC\s*Bank\b",                    "HDFCBANK.NS"),
    (r"\bTata\s*Elxsi\b",                   "TATAELXSI.NS"),
    (r"\bNMDC\b",                           "NMDC.NS"),
    (r"\bBharat\s*Electronics\b|\bBEL\b",   "BEL.NS"),
    (r"\bVedanta\b",                        "VEDL.NS"),
    (r"\bJio\s*Financial\b",                "JIOFIN.NS"),
    (r"\bMMTC\b",                           "MMTC.NS"),
    (r"\bNHPC\b",                           "NHPC.NS"),
    (r"\bNestle\b",                         "NESTLEIND.NS"),
    (r"\bTata\s*Motors\b",                  "TATAMOTORS.NS"),
    (r"\bSun\s*TV\b",                       "SUNTV.NS"),
    (r"\bReliance\s*Power\b",               "RPOWER.NS"),
    (r"\bDelta\s*Corp\b",                   "DELTACORP.NS"),
    (r"\bPNB\b|\bPunjab\s*National\s*Bank\b","PNB.NS"),
    (r"\bYes\s*Bank\b",                     "YESBANK.NS"),
    (r"\bITC\b",                            "ITC.NS"),
    (r"\bIndusInd\s*Bank\b",                "INDUSINDBK.NS"),
    (r"\bSAIL\b",                           "SAIL.NS"),
    (r"\bONGC\b",                           "ONGC.NS"),
    (r"\bBHEL\b",                           "BHEL.NS"),
    (r"\bNBCC\b",                           "NBCC.NS"),
    (r"\bIEX\b",                            "IEX.NS"),
    (r"\bTata\s*Technologies\b",            "TATATECH.NS"),
    (r"\bIndian\s*Overseas\s*Bank\b",       "IOB.NS"),
    (r"\bSJVN\b",                           "SJVN.NS"),
    (r"\bSpiceJet\b",                       "SPICEJET.NS"),
    (r"\bEaseMyTrip\b",                     "EASEMYTRIP.NS"),
    (r"\bBLS\s*International\b",            "BLS.NS"),
    (r"\bUPL\b",                            "UPL.NS"),
    (r"\bNiftybees\b",                      "NIFTYBEES.NS"),
    (r"\bGold\s*Bees\b",                    "GOLDBEES.NS"),
    (r"\bSilver\s*Bees\b",                  "SILVERBEES.NS"),
    (r"\bNippon.*?\bET[F]?\s*IT\b|\bIT\s*Bees\b", "ITBEES.NS"),
    (r"\bBansal\s*Wires\b",                 "BANSALWIRE.NS"),
    # Adani Energy Solutions sometimes appears as "Adani Trans"
    (r"\bAdani\s*(?:Energy\s*Solutions|Trans(?:mission)?)\b", "ADANIENSOL.NS"),
    # Optional / new listings:
    # (r"\bOla\s*Electric\b",               "OLA.NS"),  # if listed later
]

ALIAS_REGEX = [(re.compile(pat, re.I), tk) for pat, tk in ALIAS_TO_TICKER_PATTERNS]

# ---------------------------
# Source builder
# ---------------------------
def build_sources() -> dict[str, str]:
    sources = {}
    for name in STOCKS_FOR_GOOGLE:
        q = f"{name.replace(' ', '+')}+stock+India"
        sources[f"Google_{name}"] = f"https://news.google.com/rss/search?q={q}"
    sources["MoneyControl"] = "https://www.moneycontrol.com/rss/MCtopnews.xml"
    sources["EconomicTimes_Markets"] = "https://economictimes.indiatimes.com/rssfeeds/2146842.cms"
    sources["Investing_India"] = "https://in.investing.com/rss/"
    return sources

# ---------------------------
# Helpers
# ---------------------------
def canon_title(t: str | None) -> str:
    """Lowercase, strip bracketed parts, squeeze spaces."""
    t = (t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\[[^\]]+\]|\([^\)]+\)", "", t)
    return t.strip()

def normalize_url(u: str) -> str:
    """Remove UTM/tracking params and fragments; keep stable core URL."""
    if not u:
        return ""
    try:
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm")]
        p = p._replace(query=urlencode(q, doseq=True), fragment="")
        return urlunparse(p)
    except Exception:
        return u

def domain_of(u: str) -> str:
    try:
        d = urlparse(u).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def parse_published(entry) -> str:
    """Return ISO8601 UTC string. Fall back to now if missing."""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            return ts.isoformat(timespec="seconds")
        if "published" in entry:
            dt = pd.to_datetime(entry.published, errors="coerce", utc=True)
            if pd.isna(dt):
                raise ValueError
            return dt.isoformat()
    except Exception:
        pass
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def parse_with_retry(url: str):
    """Feedparser with custom UA + backoff."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, RETRIES + 1):
        feed = feedparser.parse(url, request_headers=headers)
        if getattr(feed, "entries", None):
            return feed
        sleep_s = BACKOFF_BASE ** (attempt - 1) + random.random()
        time.sleep(sleep_s)
    return feedparser.parse(url, request_headers=headers)

def map_ticker_from_title(title: str) -> tuple[str | None, float]:
    txt = title or ""
    for rx, tk in ALIAS_REGEX:
        if rx.search(txt):
            return tk, 0.9
    return None, 0.0

def map_ticker(title: str, source_name: str) -> tuple[str | None, float]:
    """Priority: alias in title (0.9) → fallback from Google_<Name> (0.6) → None."""
    ticker, conf = map_ticker_from_title(title)
    if ticker:
        return ticker, conf
    if source_name.startswith("Google_"):
        stock_name = source_name.split("_", 1)[1].strip().lower()
        for rx, tk in ALIAS_REGEX:
            if rx.search(stock_name):
                return tk, 0.6
    return None, 0.0

# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs("data", exist_ok=True)
    sources = build_sources()

    rows = []
    print(f"Starting news fetch from {len(sources)} sources...\n")

    for name, url in sources.items():
        try:
            feed = parse_with_retry(url)
            for e in getattr(feed, "entries", []):
                title = getattr(e, "title", "") or ""
                link_raw = getattr(e, "link", "") or ""
                link = normalize_url(link_raw)
                published_utc = parse_published(e)
                title_c = canon_title(title)
                tk, conf = map_ticker(title, name)

                # Stable ID: (title_canon | domain | YYYY-MM-DD)
                day = published_utc[:10] if published_utc else "nodate"
                nid = sha1(f"{title_c}|{domain_of(link)}|{day}")

                rows.append({
                        "source_name": name,
                        "source_domain": domain_of(link) or domain_of(url),  # use real domain
                        "title": title,
                        "link": link,
                        "published_utc": published_utc,
                        "news_id": nid,
                        "ticker": tk,
                        "map_confidence": conf,
                        "title_canon": title_c,
                        })
        except Exception as ex:
            print(f" {name} failed: {ex}")

        # polite pacing between sources
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    if not rows:
        print("No items fetched.")
        return

    df = pd.DataFrame(rows)

    # Ensure UTC dtype and hour bucket (secondary dedup safety)
    df["published_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df["pub_hour"] = df["published_utc"].dt.floor("h")

    before = len(df)

    # Primary dedup: by stable news_id
    df = df.drop_duplicates(subset=["news_id"]).copy()

    # Secondary dedup: collapse near-duplicates across outlets within same hour
    df.sort_values(["title_canon", "pub_hour", "source_name"], inplace=True)
    df = df.drop_duplicates(subset=["title_canon", "pub_hour"], keep="first")

    after = len(df)

    out = "data/raw_news.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"\nSaved {after} deduped items to {out} (dropped {before - after} dups)\n")
    print("Sample:")
    sample_cols = ["source_name", "source_domain", "ticker", "map_confidence", "published_utc", "title"]
    print(df.head(8)[sample_cols])

if __name__ == "__main__":
    main()
