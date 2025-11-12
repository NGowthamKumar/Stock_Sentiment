from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import hashlib, re
from dateutil import parser as dparser
from datetime import datetime, timezone

def normalize_url(u: str) -> str:
    try:
        p = urlparse(u)
        q = [(k,v) for k,v in parse_qsl(p.query) if not k.lower().startswith("utm")]
        p = p._replace(query=urlencode(q, doseq=True), fragment="")
        return urlunparse(p)
    except Exception:
        return u or ""

def domain(u: str) -> str:
    try: return urlparse(u).netloc.lower().replace("www.","")
    except: return ""

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def parse_date_any(s: str | None) -> datetime | None:
    if not s: return None
    try:
        dt = dparser.parse(s)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def stable_news_id(title: str, link: str, published_dt: datetime | None) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip().lower())
    d = domain(link or "")
    day = published_dt.strftime("%Y%m%d") if published_dt else "nodate"
    return sha1(f"{t}|{d}|{day}")
