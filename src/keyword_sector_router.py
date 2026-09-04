# src/keyword_sector_router.py
"""
Global keyword routing — maps unmatched global news to affected NSE sectors/stocks
Based on RavenPack taxonomy + institutional research on Indian market correlations
"""

GLOBAL_KEYWORD_ROUTING = {

    # ═══════════════════════════════════════════════════════
    # GEOPOLITICAL — affects energy, aviation, defence
    # ═══════════════════════════════════════════════════════

    "Middle East / Oil supply routes": {
        "keywords": [
            "hormuz", "strait of hormuz", "iran", "iraq", "opec",
            "oil supply", "oil disruption", "middle east", "gulf",
            "saudi", "houthi", "tanker", "shipping route",
            "oman", "qatar", "uae conflict"
        ],
        "affected_tickers": [
            "IOC.NS",      # Indian Oil Corp — crude importer
            "BPCL.NS",     # BPCL — crude importer
            "ONGC.NS",     # ONGC — oil producer benefits
            "RELIANCE.NS", # Reliance — refiner
            "INDIGO.NS",   # IndiGo — aviation fuel costs
            "ADANIPORTS.NS" # Adani Ports — shipping
        ],
        "sentiment_direction": "negative",  # disruption = bad for India importers
        "override_for": ["ONGC.NS"],        # ONGC benefits from high oil price
    },

    "Russia-Ukraine / Europe energy": {
        "keywords": [
            "russia ukraine", "ukraine war", "nato", "europe gas",
            "nord stream", "russian oil", "sanctions russia",
            "europe energy crisis"
        ],
        "affected_tickers": [
            "IOC.NS", "BPCL.NS", "RELIANCE.NS", "ONGC.NS"
        ],
        "sentiment_direction": "negative",
    },

    "China geopolitics / Taiwan": {
        "keywords": [
            "china taiwan", "south china sea", "china conflict",
            "pla", "xi jinping", "us china tension",
            "china military", "taiwan strait"
        ],
        "affected_tickers": [
            "INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS",  # IT supply chain
            "DIXON.NS", "KAYNES.NS",                           # Electronics
            "ADANIPORTS.NS"                                    # Shipping
        ],
        "sentiment_direction": "negative",
    },

    "India-Pakistan / India border": {
        "keywords": [
            "india pakistan", "kashmir", "loc", "border tension india",
            "india china border", "lac", "doklam",
            "india military", "surgical strike"
        ],
        "affected_tickers": [
            "HAL.NS", "BEL.NS", "MAZDOCK.NS", "GRSE.NS",  # Defence benefits
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"       # Banking risk
        ],
        "sentiment_direction": {
            "HAL.NS": "positive",   # Defence stocks benefit
            "BEL.NS": "positive",
            "HDFCBANK.NS": "negative",
        },
    },

    # ═══════════════════════════════════════════════════════
    # MACRO ECONOMIC — affects broad market
    # ═══════════════════════════════════════════════════════

    "US Federal Reserve": {
        "keywords": [
            "federal reserve", "fed rate", "fomc", "powell",
            "fed chair", "rate hike", "rate cut", "jackson hole",
            "us interest rate", "quantitative tightening", "qt",
            "fed minutes", "dot plot"
        ],
        "affected_tickers": [
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",  # Banks
            "BAJFINANCE.NS", "CHOLAFIN.NS",                            # NBFC
            "INFY.NS", "TCS.NS", "HCLTECH.NS",                        # IT (US revenue)
            "SUNPHARMA.NS", "DRREDDY.NS"                               # Pharma (US exports)
        ],
        "sentiment_direction": "negative",  # Rate hike = bad for India
        "override_keywords": {
            "rate cut": "positive",  # Rate cut = good for India
            "pivot": "positive",
            "pause": "positive",
        },
    },

    "RBI / India monetary policy": {
        "keywords": [
            "rbi", "reserve bank of india", "repo rate", "rbi policy",
            "mpc meeting", "monetary policy committee", "rbi governor",
            "inflation target rbi", "crr", "slr", "rbi circular"
        ],
        "affected_tickers": [
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
            "KOTAKBANK.NS", "BAJFINANCE.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS"
        ],
        "sentiment_direction": "context_dependent",
    },

    "India GDP / Economic data": {
        "keywords": [
            "india gdp", "gdp growth", "india economy", "cpi india",
            "india inflation", "wpi", "iip", "industrial production",
            "india fiscal deficit", "india current account",
            "india trade deficit", "india pmi", "manufacturing pmi"
        ],
        "affected_tickers": "ALL",  # GDP affects all stocks
        "sentiment_direction": "positive",  # Strong GDP = bullish
    },

    "US Economic data": {
        "keywords": [
            "us gdp", "us inflation", "us cpi", "us pce",
            "nonfarm payroll", "us jobs", "unemployment us",
            "us retail sales", "us manufacturing", "ism"
        ],
        "affected_tickers": [
            "INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS",  # IT (US clients)
            "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"         # Pharma (US exports)
        ],
        "sentiment_direction": "positive",
    },

    # ═══════════════════════════════════════════════════════
    # TRADE / TARIFF — affects export-heavy sectors
    # ═══════════════════════════════════════════════════════

    "US tariffs / Trade war": {
        "keywords": [
            "tariff", "trade war", "import duty", "export ban",
            "us trade policy", "trump tariff", "section 301",
            "trade restriction", "trade agreement", "fta",
            "wto dispute", "anti-dumping", "countervailing"
        ],
        "affected_tickers": [
            "INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS",   # IT services
            "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",         # Pharma
            "TATASTEEL.NS", "JSWSTEEL.NS",                    # Steel
            "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"                  # Auto exports
        ],
        "sentiment_direction": "negative",
    },

    "India FTA / Trade deals": {
        "keywords": [
            "india fta", "india trade deal", "india uk fta",
            "india eu trade", "india us trade", "india exports",
            "india import duty", "gst council", "india tariff"
        ],
        "affected_tickers": [
            "INFY.NS", "TCS.NS", "SUNPHARMA.NS",
            "TATASTEEL.NS", "JSWSTEEL.NS"
        ],
        "sentiment_direction": "positive",
    },

    # ═══════════════════════════════════════════════════════
    # COMMODITY — affects specific sectors
    # ═══════════════════════════════════════════════════════

    "Crude oil price": {
        "keywords": [
            "crude oil", "brent crude", "wti", "oil price",
            "opec cut", "opec output", "oil demand",
            "oil supply glut", "petrol price", "diesel price"
        ],
        "affected_tickers": {
            "negative": ["IOC.NS", "BPCL.NS", "INDIGO.NS", "SPICEJET.NS"],  # crude up = bad
            "positive": ["ONGC.NS", "OIL.NS", "RELIANCE.NS"]                 # crude up = good for producers
        },
        "sentiment_direction": "split",
    },

    "Gold price": {
        "keywords": [
            "gold price", "gold rally", "gold demand",
            "gold import", "gold smuggling", "bullion",
            "sovereign gold bond", "sgb"
        ],
        "affected_tickers": [
            "TITAN.NS",    # Jewellery
            "GOLDBEES.NS", # Gold ETF
            "MUTHOOTFIN.NS" # Gold loans
        ],
        "sentiment_direction": "positive",
    },

    "Steel / Metal prices": {
        "keywords": [
            "steel price", "iron ore", "coking coal",
            "metal prices", "aluminium price", "copper price",
            "china steel", "steel demand", "steel capacity"
        ],
        "affected_tickers": [
            "TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS",
            "HINDALCO.NS", "VEDL.NS", "HINDCOPPER.NS"
        ],
        "sentiment_direction": "positive",
    },

    "Semiconductor / Chip": {
        "keywords": [
            "semiconductor", "chip shortage", "chip supply",
            "nvidia", "tsmc", "intel chip", "memory chip",
            "ai chip", "foundry", "wafer", "fab"
        ],
        "affected_tickers": [
            "INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS",
            "DIXON.NS", "KAYNES.NS", "TATAELXSI.NS",
            "TATAMOTORS.NS"  # EV chips
        ],
        "sentiment_direction": "negative",  # shortage = negative for tech users
        "override_keywords": {
            "india semiconductor": "positive",  # India fab = positive
            "chip investment india": "positive",
        },
    },

    # ═══════════════════════════════════════════════════════
    # SECTOR SPECIFIC — direct sector signals
    # ═══════════════════════════════════════════════════════

    "Banking / Financial sector": {
        "keywords": [
            "bad loan", "npa", "banking crisis", "credit growth",
            "bank license", "bank merger", "banking reform",
            "financial inclusion", "digital lending", "credit card",
            "consumer credit", "household debt"
        ],
        "affected_tickers": [
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
            "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS",
            "BAJFINANCE.NS", "CHOLAFIN.NS"
        ],
        "sentiment_direction": "context_dependent",
    },

    "IT / Technology sector": {
        "keywords": [
            "it sector", "software exports", "tech layoffs",
            "ai demand", "cloud computing", "digital transformation",
            "tech spending", "it budget", "visa h1b",
            "outsourcing", "offshoring", "gig economy"
        ],
        "affected_tickers": [
            "INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS",
            "TECHM.NS", "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS"
        ],
        "sentiment_direction": "positive",
    },

    "Pharma / Healthcare": {
        "keywords": [
            "usfda", "fda approval", "drug recall", "fda warning",
            "patent cliff", "biosimilar", "generics us",
            "drug pricing", "us pharma policy", "clinical trial"
        ],
        "affected_tickers": [
            "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
            "LUPIN.NS", "AUROPHARMA.NS", "BIOCON.NS",
            "ALKEM.NS", "TORNTPHARM.NS"
        ],
        "sentiment_direction": "context_dependent",
    },

    "FMCG / Consumer": {
        "keywords": [
            "consumer spending", "rural demand", "fmcg growth",
            "monsoon", "kharif crop", "rabi crop", "harvest",
            "food inflation", "vegetable prices", "onion price",
            "consumer confidence", "retail sales india"
        ],
        "affected_tickers": [
            "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
            "BRITANNIA.NS", "DABUR.NS", "MARICO.NS",
            "COLPAL.NS", "VBL.NS", "TATACONSUM.NS"
        ],
        "sentiment_direction": "positive",
    },

    "Real Estate / Infrastructure": {
        "keywords": [
            "real estate", "housing demand", "home loan",
            "infrastructure spending", "capex", "roads",
            "pm awas yojana", "affordable housing",
            "smart city", "metro rail", "bullet train"
        ],
        "affected_tickers": [
            "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS",
            "PRESTIGE.NS", "LTIM.NS", "L&T.NS",
            "ADANIPORTS.NS", "ADANIGREEN.NS"
        ],
        "sentiment_direction": "positive",
    },

    "Auto sector": {
        "keywords": [
            "auto sales", "ev adoption", "electric vehicle",
            "vehicle sales", "auto retail", "auto exports",
            "chip shortage auto", "battery price",
            "scrappage policy", "ev subsidy", "fame scheme"
        ],
        "affected_tickers": [
            "TMCV.NS", "MARUTI.NS", "M&M.NS",
            "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
            "OLAELEC.NS"
        ],
        "sentiment_direction": "positive",
    },

    # ═══════════════════════════════════════════════════════
    # REGULATORY / SEBI / CORPORATE EVENTS
    # ═══════════════════════════════════════════════════════

    "SEBI regulatory": {
        "keywords": [
            "sebi", "sebi circular", "sebi order", "sebi ban",
            "insider trading", "market manipulation", "sebi penalty",
            "ipo sebi", "mutual fund sebi", "derivatives sebi"
        ],
        "affected_tickers": "BROAD",  # SEBI actions affect broad market
        "sentiment_direction": "negative",
    },

    "Corporate governance / Fraud": {
        "keywords": [
            "fraud", "scam", "cheating", "embezzlement",
            "corporate governance", "promoter pledge",
            "accounting fraud", "audit issue", "qualified opinion",
            "related party transaction", "forensic audit"
        ],
        "affected_tickers": "CONTEXT",  # Match to company name in headline
        "sentiment_direction": "negative",
    },

    "Earnings / Results events": {
        "keywords": [
            "quarterly results", "q1 results", "q2 results",
            "q3 results", "q4 results", "earnings beat",
            "earnings miss", "net profit", "revenue growth",
            "ebitda", "margin expansion", "margin compression",
            "guidance raised", "guidance cut"
        ],
        "affected_tickers": "CONTEXT",
        "sentiment_direction": "context_dependent",
    },

    # ═══════════════════════════════════════════════════════
    # POSITIVE SENTIMENT — market boosters
    # ═══════════════════════════════════════════════════════

    "FII / Foreign investment": {
        "keywords": [
            "fii buying", "foreign investment", "fpi inflow",
            "india attractive", "emerging market",
            "india valuation", "morgan stanley india",
            "jp morgan india", "goldman india", "upgrade india"
        ],
        "affected_tickers": "ALL",
        "sentiment_direction": "positive",
    },

    "Budget / Government spending": {
        "keywords": [
            "budget", "union budget", "capex budget",
            "government spending", "fiscal stimulus",
            "infrastructure push", "production linked incentive", "pli",
            "make in india", "atmanirbhar"
        ],
        "affected_tickers": [
            "LT.NS", "NTPC.NS", "POWERGRID.NS",
            "HAL.NS", "BEL.NS", "IRFC.NS", "RVNL.NS",
            "DIXON.NS", "KAYNES.NS"
        ],
        "sentiment_direction": "positive",
    },
}

def get_sentiment_for_ticker(category_config: dict, ticker: str) -> str:
    """
    Returns correct sentiment direction for a specific ticker.
    Handles override_for — e.g. ONGC benefits when oil rises
    even though Iran disruption is negative for importers.
    """
    direction = category_config.get("sentiment_direction", "negative")
    override_for = category_config.get("override_for", [])

    if isinstance(direction, dict):
        # Split sentiment (e.g. crude oil)
        for sentiment_type, tickers in direction.items():
            if ticker in tickers:
                return sentiment_type
        return "negative"

    if ticker in override_for:
        # Flip direction for override tickers
        if direction == "negative":
            return "positive"
        elif direction == "positive":
            return "negative"

    return direction if isinstance(direction, str) else "negative"


def route_global_news(title: str, ticker) -> list:
    """
    If ticker is None/NaN, route article to affected sector stocks.
    Returns list of (ticker, sentiment_direction) tuples.
    """
    import pandas as pd
    if pd.notna(ticker):
        return []  # already mapped — no routing needed

    title_lower = title.lower()
    routed = []
    seen = set()

    for category, config in GLOBAL_KEYWORD_ROUTING.items():
        keywords = config.get("keywords", [])
        if not any(kw in title_lower for kw in keywords):
            continue

        affected = config.get("affected_tickers", [])

        if affected in ("ALL", "BROAD", "CONTEXT"):
            routed.append(("NIFTY_BROAD", "positive"))
            continue

        if isinstance(affected, list):
            for tk in affected:
                if tk not in seen:
                    direction = get_sentiment_for_ticker(config, tk)
                    routed.append((tk, direction))
                    seen.add(tk)

        elif isinstance(affected, dict):
            for sentiment_type, tickers in affected.items():
                for tk in tickers:
                    if tk not in seen:
                        routed.append((tk, sentiment_type))
                        seen.add(tk)

    return routed

# ═══════════════════════════════════════════════════════
# SENTIMENT MULTIPLIERS (for price movement language)
# ═══════════════════════════════════════════════════════

PRICE_MOVEMENT_KEYWORDS = {
    "negative": [
        "falls", "drops", "declines", "tumbles", "plunges",
        "hits 52-week low", "hits all-time low", "crashes",
        "down %", "loses", "slips", "weakens",
        "warning", "penalty", "lawsuit", "fraud", "ban",
        "miss", "below estimate", "cuts guidance", "downgrade"
    ],
    "positive": [
        "rises", "gains", "surges", "rallies", "jumps",
        "hits 52-week high", "all-time high", "record high",
        "up %", "beats", "above estimate", "raises guidance",
        "upgrade", "strong results", "outperforms", "breakout"
    ],
    "multiplier": 0.3,  # Boost/reduce ensemble score by 30%
}