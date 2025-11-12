# src/event_classifier.py
import re
import pandas as pd

# Canonical event buckets
EVENTS = [
    "EARNINGS", "GUIDANCE", "MA", "LITIGATION",
    "REGULATORY", "MGMT_CHANGE", "ORDER_WIN",
    "PRODUCT_LAUNCH", "MACRO", "OTHER"
]

_PATTERNS = {
    "EARNINGS": r"\b(results?|q[1-4]|quarter|profit|loss|revenue|ebitda|earnings|eps|margin|beat|miss)\b",
    "GUIDANCE": r"\b(guidance|outlook|forecast|raises?|cuts?)\b",
    "MA": r"\b(acquires?|acquisition|merger|stake|buyout|takeover|investment)\b",
    "LITIGATION": r"\b(lawsuit|probe|investigation|litigation|penalty|fine|allegation|fraud)\b",
    "REGULATORY": r"\b(sebi|regulator|ban|approval|clearance|license|licen[cs]e|rbi|dppa)\b",
    "MGMT_CHANGE": r"\b(ceo|cfo|chairman|director|board|resign|steps down|appointment|appoints?)\b",
    "ORDER_WIN": r"\b(order|contract|tender|bagged|secures?)\b",
    "PRODUCT_LAUNCH": r"\b(launch|unveil|rollout|introduces?|debut)\b",
    "MACRO": r"\b(inflation|gdp|rate hike|policy|budget|currency|oil|commodit(?:y|ies))\b",
}
RX = {k: re.compile(v, re.I) for k, v in _PATTERNS.items()}

def classify_event(title: str) -> str:
    t = title or ""
    for k, rx in RX.items():
        if rx.search(t):
            return k
    return "OTHER"

def add_event_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["event_type"] = d["title"].astype(str).map(classify_event)
    return d
