import re
from typing import Optional

class EntityMapper:
    def __init__(self, stocks: list[dict]):
        self.alias_index = []
        for s in stocks:
            aliases = [s["name"], *s.get("aliases", [])]
            for a in aliases:
                if not a: 
                    continue
                pat = r"(?:^|[^A-Za-z0-9])" + re.escape(a.lower()) + r"(?:[^A-Za-z0-9]|$)"
                self.alias_index.append((re.compile(pat), s["ticker"]))
    def map_text(self, text: str) -> tuple[Optional[str], float]:
        txt = (text or "").lower()
        hits = []
        for rx, tk in self.alias_index:
            if rx.search(txt):
                hits.append(tk)
        if not hits: 
            return (None, 0.0)
        return (hits[0], min(1.0, 0.5 + 0.25*len(hits)))
