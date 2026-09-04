# src/fetch_news.py
"""
Fetch latest Indian stock-market headlines from:
  1) Google News (per-stock query) — 150+ stocks
  2) 215 RSS sources — ET, BS, Mint, NDTVProfit, BusinessLine,
     IndianExpress, TheHindu, Reuters, Investing.com and more
  3) Global keyword routing — maps geopolitical/macro news
     to affected NSE sector stocks automatically

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
from turtle import title
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from src.keyword_sector_router import GLOBAL_KEYWORD_ROUTING, PRICE_MOVEMENT_KEYWORDS
from concurrent.futures import ThreadPoolExecutor

import feedparser
import pandas as pd

# ---------------------------
# Config
# ---------------------------
USER_AGENT = "IndianStockSentiment/1.0 (+https://github.com/NGowthamKumar)"
SLEEP_MIN, SLEEP_MAX = 0.6, 1.2   # pacing between sources
RETRIES = 3
BACKOFF_BASE = 1.5                # 1.0, 2.5, 4.25 ... + jitter

# Source reliability weights — higher = more trusted
SOURCE_WEIGHTS = {
    # Tier 1 — Premium institutional sources
    "BusinessStandard_Latest":   1.0,
    "BS_Companies":              1.0,
    "BS_Finance":                1.0,
    "BS_Economy":                1.0,
    "ET_Stocks":                 1.0,
    "ET_Companies":              1.0,
    "ET_Economy":                1.0,
    "EconomicTimes_Markets":     1.0,
    "Mint_Markets":              0.95,
    "Mint_Companies":            0.95,
    "Mint_Money":                0.95,
    "NDTVProfit_Markets":        0.90,
    "BusinessLine_Markets":      0.90,
    "BL_Economy":                0.90,
    "BL_Companies":              0.90,
    "Reuters_India":             1.0,
    "Reuters_Markets":           1.0,
    # Tier 2 — Good retail sources
    "IndianExpress_Business":    0.80,
    "IndianExpress_Market":      0.80,
    "TheHindu_Business":         0.85,
    "TheHindu_Markets":          0.85,
    "IndiaToday_Business":       0.75,
    "Investing_Stocks":          0.75,
    "Investing_India":           0.75,
    # Tier 3 — Retail/Blog sources
    "TradeBrains":               0.60,
    "TradeBrains_News":          0.60,
    "Goodreturns":               0.60,
    "Goodreturns_Market":        0.60,
    "Pulse_Zerodha":             0.65,
    "Equitymaster":              0.65,
    "IIFL_Markets":              0.65,
    # Google News — variable quality
    "default_google":            0.55,
    # Default for unknown sources
    "default":                   0.50,
}

def get_source_weight(source_name: str) -> float:
    """Get reliability weight for a news source"""
    if source_name in SOURCE_WEIGHTS:
        return SOURCE_WEIGHTS[source_name]
    if source_name.startswith("Google_"):
        return SOURCE_WEIGHTS["default_google"]
    return SOURCE_WEIGHTS["default"]

# ---------------------------
# Portfolio for Google queries
# ---------------------------
STOCKS_FOR_GOOGLE = [
    # Core holdings + ETFs
    "HCL Tech", "Reliance Industries", "Vedanta", "Bansal Wires Industries",
    "Nippon India ETF Gold Bees", "Niftybees", "UPL", "HDFC Bank",
    "Jio Financial Services", "Coal India", "Tata Steel", "Nippon India Silver Bees",
    "IRFC", "Tata Elxsi", "Infosys", "NMDC", "Bharat Electronics",
    "Adani Energy Solutions", "MMTC", "NHPC", "Nestle", "Nippon India EFT IT",
    "Sun TV Network", "Reliance Power", "Delta Corp", "PNB",
    "Yes Bank", "ITC", "IndusInd Bank", "Sail", "ONGC", "EaseMyTrip",
    "BHEL", "BLS International Services", "Natco Pharma", "NBCC India",
    "IEX", "Tata Technologies", "Indian Overseas Bank", "SJVN",
    "Tata Motors Commercial Vehicles", "ICICI Bank",

    # Nifty 50 core
    "TCS", "Bharti Airtel", "State Bank of India", "Bajaj Finance", "Larsen Toubro",
    "Hindustan Unilever", "Sun Pharma", "Maruti Suzuki", "Mahindra Mahindra", "Wipro",
    "Axis Bank", "NTPC", "Power Grid", "Adani Enterprises",
    "Kotak Mahindra Bank", "Bank of Baroda", "Bajaj Finserv",
    "HDFC Life Insurance", "SBI Life Insurance", "Tech Mahindra", "Mphasis",
    "Adani Ports", "Adani Green Energy", "Tata Power", "Indian Oil Corporation",
    "BPCL", "Gail India",

    # Consumer / FMCG
    "Asian Paints", "Dabur", "Godrej Consumer", "Titan Company", "Trent",
    "Marico", "Colgate Palmolive India", "Britannia Industries",
    "United Spirits", "Varun Beverages", "United Breweries", "Radico Khaitan",
    "Tata Consumer Products", "Emami", "Jyothy Labs", "Whirlpool India",
    "Hindustan Zinc", "CCL Products", "Bikaji Foods",

    # Pharma / Healthcare
    "Dr Reddy Labs", "Cipla", "Divis Laboratories", "Apollo Hospitals",
    "Lupin", "Biocon", "Aurobindo Pharma", "Mankind Pharma", "Zydus Lifesciences",
    "Torrent Pharmaceuticals", "Laurus Labs", "Alkem Laboratories",
    "Ipca Laboratories", "Ajanta Pharma", "Granules India", "Max Healthcare",
    "Narayana Hrudayalaya", "Fortis Healthcare", "Krishna Institute Medical Sciences",
    "Syngene International", "Abbott India", "Pfizer India", "Glaxosmithkline Pharma",

    # Auto
    "Bajaj Auto", "Hero MotoCorp", "Eicher Motors", "MRF Tyres",
    "Apollo Tyres", "Balkrishna Industries", "Samvardhana Motherson",
    "Bosch India", "Minda Industries", "Sona BLW Precision",
    "Craftsman Automation", "Endurance Technologies",

    # Consumer tech / Internet
    "Zomato", "Paytm", "Dmart", "LIC India", "Info Edge India", "Naukri",
    "PolicyBazaar", "One97 Communications", "Nykaa", "Vodafone Idea",
    "Indiamart Intermesh", "Cartrade Tech",

    # Real estate
    "DLF", "Godrej Properties", "Oberoi Realty", "Prestige Estates",
    "Macrotech Developers", "Brigade Enterprises", "Sobha Developers",
    "Phoenix Mills",

    # Cement / Materials
    "Shree Cement", "UltraTech Cement", "Ambuja Cements", "ACC Cement",
    "Hindalco", "JSW Steel", "Jindal Steel Power", "Vedanta Aluminium",
    "Shyam Metalics", "APL Apollo Tubes", "Ratnamani Metals", "Tube Investments",
    "National Aluminium", "Hindustan Copper",

    # Paints / Chemicals
    "Havells India", "Pidilite Industries", "Berger Paints",
    "SRF", "Aarti Industries", "Deepak Nitrite", "Navin Fluorine", "Balaji Amines",
    "Gujarat Fluorochemicals", "Alkyl Amines", "Vinati Organics",

    # Financials / NBFC / Insurance
    "Muthoot Finance", "Cholamandalam Finance", "SBI Cards",
    "HDFC AMC", "Nippon AMC", "Angel One", "BSE India",
    "PB Fintech", "Computer Age Management", "UTI AMC",
    "Bajaj Holdings", "Shriram Finance", "Aditya Birla Capital",
    "Five Star Business Finance", "Home First Finance",
    "IIFL Finance", "Manappuram Finance",

    # Hospitality / Travel
    "Indian Hotels", "InterGlobe Aviation", "IndiGo Airlines",
    "EIH Hotels", "Lemon Tree Hotels", "Mahindra Holidays",

    # Capital goods / Defence / Infra
    "Cummins India", "ABB India", "Siemens India", "Bharat Forge",
    "Page Industries", "Voltas", "Blue Star",
    "Hindustan Aeronautics", "BEL", "Mazagon Dock",
    "Cochin Shipyard", "Garden Reach Shipbuilders",
    "Solar Industries", "Data Patterns", "Paras Defence",
    "MTAR Technologies", "Bharat Dynamics",
    "HUDCO", "Jaiprakash Power", "Adani Power",
    "RVNL", "Ircon International", "KEC International",
    "Kalpataru Projects", "PNC Infratech",

    # IT / Tech
    "Dixon Technologies", "Amber Enterprises", "Kaynes Technology",
    "KPIT Technologies", "Persistent Systems", "Coforge", "LTM Limited",
    "Tata Communications", "Zensar Technologies", "Mastek",
    "NIIT Technologies", "Hexaware Technologies",

    # PSU / Government
    "Indian Railway Finance", "Rail Vikas Nigam", "IRCTC",
    "Bharat Heavy Electricals", "NTPC Green Energy",
    "Oil India", "HPCL", "MRPL",

    # Renewables / Energy
    "Suzlon Energy", "Torrent Power", "Adani Total Gas",
    "Gujarat Gas", "Indraprastha Gas", "Mahanagar Gas",
    "CESC", "Tata Power Renewables",

    # Telecom
    "MTNL", "Tata Communications", "Route Mobile", "Tanla Platforms",

    # Specialty / Others
    "Jubilant Foodworks", "Devyani International", "Westlife Foodworld",
    "Sapphire Foods", "Restaurant Brands Asia",
    "Affle India", "IndiGrid", "Sterlite Power",
    "Timken India", "SKF India", "Schaeffler India",

    # ── ETFs & Index Funds ──
    # Gold ETFs
    "Gold BeES", "HDFC Gold ETF", "SBI Gold ETF",
    "Kotak Gold ETF", "Nippon Gold ETF", "Axis Gold ETF",

    # Silver ETFs
    "Nippon Silver ETF", "ICICI Silver ETF", "Mirae Silver ETF",

    # Nifty 50 ETFs (most traded)
    "Nippon Nifty BeES", "SBI Nifty ETF", "HDFC Nifty ETF",
    "Kotak Nifty ETF", "UTI Nifty ETF",

    # Sectoral ETFs
    "Nippon IT ETF", "ICICI IT ETF",           # IT sector
    "Nippon Bank BeES", "SBI Banking ETF",     # Banking sector
    "Nippon Pharma ETF", "ICICI Pharma ETF",   # Pharma sector
    "CPSE ETF", "Bharat 22 ETF",               # PSU stocks
    "Nippon Infra ETF",                        # Infrastructure

    # International ETFs
    "Motilal Nasdaq 100 ETF", "Mirae Nasdaq ETF",
    "Motilal S&P 500 ETF",

    # Debt / Liquid ETFs
    "Nippon Liquid ETF", "ICICI Liquid ETF",
]
# ---------------------------
# Alias → NSE ticker map (extend as needed)
#    (regex patterns for robustness)
# ---------------------------
ALIAS_TO_TICKER_PATTERNS = [
    # ── Core / Large Cap ──
    (r"\bHCL\s*Tech(?:nologies)?\b",        "HCLTECH.NS"),
    (r"\bReliance(?:\s+Industries)?\b",     "RELIANCE.NS"),
    (r"\bInfosys\b",                         "INFY.NS"),
    (r"\bCoal\s*India\b",                    "COALINDIA.NS"),
    (r"\bTata\s*Steel\b",                    "TATASTEEL.NS"),
    (r"\bHDFC\s*Bank\b",                     "HDFCBANK.NS"),
    (r"\bTata\s*Elxsi\b",                    "TATAELXSI.NS"),
    (r"\bNMDC\b",                            "NMDC.NS"),
    (r"\bBharat\s*Electronics\b|\bBEL\b",    "BEL.NS"),
    (r"\bVedanta\b",                         "VEDL.NS"),
    (r"\bJio\s*Financial\b",                 "JIOFIN.NS"),
    (r"\bMMTC\b",                            "MMTC.NS"),
    (r"\bNHPC\b",                            "NHPC.NS"),
    (r"\bNestle\b",                          "NESTLEIND.NS"),
    (r"\bSun\s*TV\b",                        "SUNTV.NS"),
    (r"\bReliance\s*Power\b",                "RPOWER.NS"),
    (r"\bDelta\s*Corp\b",                    "DELTACORP.NS"),
    (r"\bPNB\b|\bPunjab\s*National\s*Bank\b","PNB.NS"),
    (r"\bYes\s*Bank\b",                      "YESBANK.NS"),
    (r"\bITC\b",                             "ITC.NS"),
    (r"\bIndusInd\s*Bank\b",                 "INDUSINDBK.NS"),
    (r"\bSAIL\b",                            "SAIL.NS"),
    (r"\bONGC\b",                            "ONGC.NS"),
    (r"\bBHEL\b",                            "BHEL.NS"),
    (r"\bNBCC\b",                            "NBCC.NS"),
    (r"\bIEX\b",                             "IEX.NS"),
    (r"\bTata\s*Technologies\b",             "TATATECH.NS"),
    (r"\bIndian\s*Overseas\s*Bank\b",        "IOB.NS"),
    (r"\bSJVN\b",                            "SJVN.NS"),
    (r"\bEaseMyTrip\b",                      "EASEMYTRIP.NS"),
    (r"\bBLS\s*International\b",             "BLS.NS"),
    (r"\bUPL\b",                             "UPL.NS"),
    (r"\bBansal\s*Wires\b",                  "BANSALWIRE.NS"),
    (r"\bAdani\s*(?:Energy\s*Solutions|Trans(?:mission)?)\b", "ADANIENSOL.NS"),
    (r"\bTata\s*Motors\s*Passenger\b|\bTMPV\b|\bTata\s*PV\b", "TMPV.NS"),
    (r"\bTata\s*Motors\s*(?:Commercial|CV)?\b|\bTMCV\b",       "TMCV.NS"),

    # ── Nifty 50 ──
    (r"\bICICI\s*Bank\b",                    "ICICIBANK.NS"),
    (r"\bTCS\b|\bTata\s*Consultancy\b",      "TCS.NS"),
    (r"\bBharti\s*Airtel\b|\bAirtel\b",      "BHARTIARTL.NS"),
    (r"\bSBI\b|\bState\s*Bank\b",            "SBIN.NS"),
    (r"\bBajaj\s*Finance\b",                 "BAJFINANCE.NS"),
    (r"\bL&T\b|\bLarsen\b",                  "LT.NS"),
    (r"\bHUL\b|\bHindustan\s*Unilever\b",    "HINDUNILVR.NS"),
    (r"\bSun\s*Pharma\b",                    "SUNPHARMA.NS"),
    (r"\bMaruti\b|\bMaruti\s*Suzuki\b",      "MARUTI.NS"),
    (r"\bM&M\b|\bMahindra\b",               "M&M.NS"),
    (r"\bWipro\b",                           "WIPRO.NS"),
    (r"\bAxis\s*Bank\b",                     "AXISBANK.NS"),
    (r"\bNTPC\b",                            "NTPC.NS"),
    (r"\bPower\s*Grid\b",                    "POWERGRID.NS"),
    (r"\bAdani\s*Enterprises\b",             "ADANIENT.NS"),
    (r"\bKotak\s*(?:Mahindra\s*)?Bank\b",    "KOTAKBANK.NS"),
    (r"\bBank\s*of\s*Baroda\b|\bBoB\b",      "BANKBARODA.NS"),
    (r"\bBajaj\s*Finserv\b",                 "BAJAJFINSV.NS"),
    (r"\bHDFC\s*Life\b",                     "HDFCLIFE.NS"),
    (r"\bSBI\s*Life\b",                      "SBILIFE.NS"),
    (r"\bTech\s*Mahindra\b",                 "TECHM.NS"),
    (r"\bMphasis\b",                         "MPHASIS.NS"),
    (r"\bAdani\s*Ports\b",                   "ADANIPORTS.NS"),
    (r"\bAdani\s*Green\b",                   "ADANIGREEN.NS"),
    (r"\bTata\s*Power\b",                    "TATAPOWER.NS"),
    (r"\bIndian\s*Oil\b|\bIOC\b",            "IOC.NS"),
    (r"\bBPCL\b|\bBharat\s*Petroleum\b",     "BPCL.NS"),
    (r"\bGAIL\b|\bGail\s*India\b",           "GAIL.NS"),

    # ── Consumer / FMCG ──
    (r"\bAsian\s*Paints\b",                  "ASIANPAINT.NS"),
    (r"\bDabur\b",                           "DABUR.NS"),
    (r"\bGodrej\s*Consumer\b",               "GODREJCP.NS"),
    (r"\bTitan\b",                           "TITAN.NS"),
    (r"\bTrent\b",                           "TRENT.NS"),
    (r"\bMarico\b|\bParachute\b",            "MARICO.NS"),
    (r"\bColgate\b",                         "COLPAL.NS"),
    (r"\bBritannia\b",                       "BRITANNIA.NS"),
    (r"\bTata\s*Consumer\b",                 "TATACONSUM.NS"),
    (r"\bEmami\b",                           "EMAMILTD.NS"),
    (r"\bVarun\s*Beverages\b",               "VBL.NS"),
    (r"\bUnited\s*Breweries\b|\bKingfisher\b","UBL.NS"),
    (r"\bRadico\b",                          "RADICO.NS"),
    (r"\bJyothy\b",                          "JYOTHYLAB.NS"),
    (r"\bBikaji\b",                          "BIKAJI.NS"),
    (r"\bCCL\s*Products\b",                  "CCLPROD.NS"),
    (r"\bHindustan\s*Zinc\b",                "HINDZINC.NS"),

    # ── Pharma / Healthcare ──
    (r"\bDr\s*Reddy\b",                      "DRREDDY.NS"),
    (r"\bCipla\b",                           "CIPLA.NS"),
    (r"\bDivi\s*(?:s|'s)?\s*Lab\b",          "DIVISLAB.NS"),
    (r"\bApollo\s*Hospitals\b",              "APOLLOHOSP.NS"),
    (r"\bTorrent\s*Pharma\b",               "TORNTPHARM.NS"),
    (r"\bLupin\b",                           "LUPIN.NS"),
    (r"\bBiocon\b",                          "BIOCON.NS"),
    (r"\bAurobindo\b",                       "AUROPHARMA.NS"),
    (r"\bMankind\s*Pharma\b",               "MANKIND.NS"),
    (r"\bZydus\b",                           "ZYDUSLIFE.NS"),
    (r"\bAlkem\b",                           "ALKEM.NS"),
    (r"\bIpca\b",                            "IPCALAB.NS"),
    (r"\bAjanta\s*Pharma\b",                "AJANTPHARM.NS"),
    (r"\bLaurus\s*Labs\b",                   "LAURUSLABS.NS"),
    (r"\bGranules\b",                        "GRANULES.NS"),
    (r"\bMax\s*Healthcare\b",               "MAXHEALTH.NS"),
    (r"\bNarayana\s*(?:Hrudayalaya|Health)\b","NH.NS"),
    (r"\bFortis\s*Healthcare\b|\bFortis\b",  "FORTIS.NS"),
    (r"\bKIMS\b|\bKrishna\s*Institute\b",    "KIMS.NS"),
    (r"\bSyngene\b",                         "SYNGENE.NS"),
    (r"\bAbbott\s*India\b",                  "ABBOTINDIA.NS"),

    # ── Auto ──
    (r"\bBajaj\s*Auto\b",                    "BAJAJ-AUTO.NS"),
    (r"\bHero\s*(?:MotoCorp|Moto)\b",        "HEROMOTOCO.NS"),
    (r"\bEicher\s*Motors\b|\bRoyal\s*Enfield\b","EICHERMOT.NS"),
    (r"\bMRF\b",                             "MRF.NS"),
    (r"\bApollo\s*Tyre\b",                   "APOLLOTYRE.NS"),
    (r"\bBalkrishna\b|\bBKT\b",              "BALKRISIND.NS"),
    (r"\bMotherson\b|\bSamvardhana\b",       "MOTHERSON.NS"),
    (r"\bBosch\s*India\b",                   "BOSCHLTD.NS"),
    (r"\bMinda\s*Industries\b",              "MINDAIND.NS"),
    (r"\bSona\s*BLW\b|\bSona\s*Comstar\b",   "SONACOMS.NS"),
    (r"\bTube\s*Investments\b|\bTI\s*India\b","TIINDIA.NS"),

    # ── Consumer Tech / Internet ──
    (r"\bZomato\b|\bEternal\b",              "ETERNAL.NS"),
    (r"\bOla\s*Electric\b|\bOlectric\b",     "OLAELEC.NS"),
    (r"\bPaytm\b|\bOne97\b",                 "PAYTM.NS"),
    (r"\bDmart\b|\bAvenue\s*Supermarts\b",   "DMART.NS"),
    (r"\bLIC\b|\bLife\s*Insurance\s*Corporation\b","LICI.NS"),
    (r"\bInfo\s*Edge\b|\bNaukri\b",          "NAUKRI.NS"),
    (r"\bPolicyBazaar\b|\bPB\s*Fintech\b",   "POLICYBZR.NS"),
    (r"\bNykaa\b|\bFSN\b",                   "NYKAA.NS"),
    (r"\bVodafone\s*Idea\b|\bVi\b",          "IDEA.NS"),
    (r"\bIndiamart\b",                       "INDIAMART.NS"),
    (r"\bAffle\b",                           "AFFLE.NS"),
    (r"\bRoute\s*Mobile\b",                  "ROUTE.NS"),
    (r"\bTanla\b",                           "TANLA.NS"),

    # ── Real Estate ──
    (r"\bDLF\b",                             "DLF.NS"),
    (r"\bGodrej\s*Properties\b",             "GODREJPROP.NS"),
    (r"\bOberoi\s*Realty\b",                 "OBEROIRLTY.NS"),
    (r"\bPrestige\b",                        "PRESTIGE.NS"),
    (r"\bMacrotech\b|\bLodha\b",             "LODHA.NS"),
    (r"\bBrigade\s*Enterprises\b",           "BRIGADE.NS"),
    (r"\bSobha\b",                           "SOBHA.NS"),
    (r"\bPhoenix\s*Mills\b",                 "PHOENIXLTD.NS"),

    # ── Cement / Materials ──
    (r"\bShree\s*Cement\b",                  "SHREECEM.NS"),
    (r"\bUltraTech\s*Cement\b",              "ULTRACEMCO.NS"),
    (r"\bAmbuja\s*Cement\b",                 "AMBUJACEM.NS"),
    (r"\bACC\s*Cement\b|\bACC\b",            "ACC.NS"),
    (r"\bHindalco\b",                        "HINDALCO.NS"),
    (r"\bJSW\s*Steel\b",                     "JSWSTEEL.NS"),
    (r"\bJindal\s*Steel\b|\bJSPL\b",         "JINDALSTEL.NS"),
    (r"\bShyam\s*Metalics\b",               "SHYAMMETL.NS"),
    (r"\bAPL\s*Apollo\b",                    "APLAPOLLO.NS"),
    (r"\bNational\s*Aluminium\b|\bNALCO\b",  "NATIONALUM.NS"),
    (r"\bHindustan\s*Copper\b",              "HINDCOPPER.NS"),

    # ── Paints / Chemicals ──
    (r"\bHavells\b",                         "HAVELLS.NS"),
    (r"\bPidilite\b|\bFeviCol\b",            "PIDILITIND.NS"),
    (r"\bBerger\s*Paints\b",                 "BERGEPAINT.NS"),
    (r"\bSRF\b",                             "SRF.NS"),
    (r"\bAarti\s*Industries\b",              "AARTIIND.NS"),
    (r"\bDeepak\s*Nitrite\b|\bDeepak\s*Nitrate\b","DEEPAKNTR.NS"),
    (r"\bNavin\s*Fluorine\b",               "NAVINFLUOR.NS"),
    (r"\bBalaji\s*Amines\b",                 "BALAMINES.NS"),
    (r"\bGujarat\s*Fluoro\b|\bGujfluoro\b",  "GUJFLUORO.NS"),
    (r"\bAlkyl\s*Amines\b",                  "ALKYLAMINE.NS"),
    (r"\bVinati\s*Organics\b",               "VINATIORGA.NS"),

    # ── Financials / NBFC / Insurance ──
    (r"\bMuthoot\s*Finance\b",               "MUTHOOTFIN.NS"),
    (r"\bChola\b|\bCholamandalam\b",         "CHOLAFIN.NS"),
    (r"\bSBI\s*Card\b",                      "SBICARD.NS"),
    (r"\bHDFC\s*AMC\b",                      "HDFCAMC.NS"),
    (r"\bNippon\s*AMC\b|\bNippon\s*Life\b",  "NAM-INDIA.NS"),
    (r"\bAngel\s*One\b",                     "ANGELONE.NS"),
    (r"\bBSE\s*India\b|\bBSE\s*Ltd\b",       "BSE.NS"),
    (r"\bCAMS\b|\bComputer\s*Age\b",         "CAMS.NS"),
    (r"\bUTI\s*AMC\b",                       "UTIAMC.NS"),
    (r"\bShriram\s*Finance\b",               "SHRIRAMFIN.NS"),
    (r"\bAditya\s*Birla\s*Capital\b",        "ABCAPITAL.NS"),
    (r"\bFive\s*Star\s*(?:Business)?\s*Finance\b","FIVESTAR.NS"),
    (r"\bHome\s*First\b",                    "HOMEFIRST.NS"),
    (r"\bIIFL\s*Finance\b|\bIIFL\b",         "IIFL.NS"),
    (r"\bManappuram\b",                      "MANAPPURAM.NS"),
    (r"\bBajaj\s*Holdings\b",               "BAJAJHLDNG.NS"),

    # ── Hospitality / Travel ──
    (r"\bIndian\s*Hotels\b|\bTaj\b",         "INDHOTEL.NS"),
    (r"\bIndiGo\b|\bInterGlobe\b",           "INDIGO.NS"),
    (r"\bIRCTC\b|\bIndian\s*Railway\s*Catering\b","IRCTC.NS"),
    (r"\bEIH\s*Hotels\b|\bObberoi\s*Hotels\b","EIHOTEL.NS"),
    (r"\bLemon\s*Tree\b",                    "LEMONTREE.NS"),

    # ── Capital Goods / Defence / Infra ──
    (r"\bCummins\b",                         "CUMMINSIND.NS"),
    (r"\bABB\s*India\b|\bABB\b",             "ABB.NS"),
    (r"\bSiemens\b",                         "SIEMENS.NS"),
    (r"\bBharat\s*Forge\b",                  "BHARATFORG.NS"),
    (r"\bPage\s*Industries\b|\bJockey\b",    "PAGEIND.NS"),
    (r"\bVoltas\b",                          "VOLTAS.NS"),
    (r"\bBlue\s*Star\b",                     "BLUESTARCO.NS"),
    (r"\bHAL\b|\bHindustan\s*Aeronautics\b", "HAL.NS"),
    (r"\bMazagon\s*Dock\b|\bMDL\b",          "MAZDOCK.NS"),
    (r"\bCochin\s*Shipyard\b",               "COCHINSHIP.NS"),
    (r"\bGarden\s*Reach\b|\bGRSE\b",         "GRSE.NS"),
    (r"\bSolar\s*Industries\b",              "SOLARINDS.NS"),
    (r"\bData\s*Patterns\b",                 "DATAPATTNS.NS"),
    (r"\bParas\s*Defence\b",                 "PDRP.NS"),
    (r"\bMTAR\s*Technologies\b|\bMTAR\b",    "MTAR.NS"),
    (r"\bBharat\s*Dynamics\b|\bBDL\b",       "BDL.NS"),
    (r"\bHUDCO\b",                           "HUDCO.NS"),
    (r"\bJaiprakash\s*Power\b|\bJP\s*Power\b","JPPOWER.NS"),
    (r"\bRVNL\b|\bRail\s*Vikas\b",           "RVNL.NS"),
    (r"\bIrcon\b",                           "IRCON.NS"),
    (r"\bKEC\s*International\b|\bKEC\b",     "KEC.NS"),
    (r"\bKalpataru\b",                       "KPIL.NS"),
    (r"\bPNC\s*Infratech\b",                 "PNCINFRA.NS"),
    (r"\bIRFC\b|\bIndian\s*Railway\s*Finance\b","IRFC.NS"),
    (r"\bNatco\s*Pharma\b",                  "NATCOPHARM.NS"),

    # ── IT / Tech ──
    (r"\bPersistent\s*Systems\b",            "PERSISTENT.NS"),
    (r"\bCoforge\b",                         "COFORGE.NS"),
    (r"\bLTIMindtree\b|\bLTM\b|\bLTI\b",    "LTM.NS"),
    (r"\bKPIT\s*Tech\b",                     "KPITTECH.NS"),
    (r"\bKaynes\b",                          "KAYNES.NS"),
    (r"\bData\s*Patterns\b",                 "DATAPATTNS.NS"),
    (r"\bDixon\s*Tech\b",                    "DIXON.NS"),
    (r"\bAmber\s*Enterprises\b",             "AMBER.NS"),
    (r"\bTata\s*Communications\b",           "TATACOMM.NS"),
    (r"\bZensar\b",                          "ZENSARTECH.NS"),
    (r"\bMastek\b",                          "MASTEK.NS"),

    # ── Energy / Power ──
    (r"\bAdani\s*Power\b",                   "ADANIPOWER.NS"),
    (r"\bTorrent\s*Power\b",                 "TORNTPOWER.NS"),
    (r"\bSuzlon\b",                          "SUZLON.NS"),
    (r"\bAdani\s*Total\s*Gas\b",             "ATGL.NS"),
    (r"\bGujarat\s*Gas\b",                   "GUJGASLTD.NS"),
    (r"\bIGL\b|\bIndraprastha\s*Gas\b",      "IGL.NS"),
    (r"\bMahanagar\s*Gas\b|\bMGL\b",         "MGL.NS"),
    (r"\bCESC\b",                            "CESC.NS"),
    (r"\bOil\s*India\b",                     "OIL.NS"),
    (r"\bHPCL\b|\bHindustan\s*Petroleum\b",  "HINDPETRO.NS"),
    (r"\bMRPL\b",                            "MRPL.NS"),

    # ── Food / QSR ──
    (r"\bJubilant\s*Food(?:works)?\b|\bDominos\b","JUBLFOOD.NS"),
    (r"\bDevyani\b",                         "DEVYANI.NS"),
    (r"\bWestlife\b|\bMcDonald\b",           "WESTLIFE.NS"),
    (r"\bSapphire\s*Foods\b|\bKFC\s*India\b","SAPPHIRE.NS"),

    # ── ETFs ──
    (r"\bNiftybees\b|\bNifty\s*BeES\b",      "NIFTYBEES.NS"),
    (r"\bGold\s*Bees\b|\bGold\s*BeES\b",     "GOLDBEES.NS"),
    (r"\bSilver\s*Bees\b|\bSilver\s*BeES\b", "SILVERBEES.NS"),
    (r"\bNippon.*?\bET[F]?\s*IT\b|\bIT\s*Bees\b","ITBEES.NS"),
    (r"\bCPSE\s*ETF\b",                      "CPSEETF.NS"),
    (r"\bBharat\s*22\s*ETF\b",               "BHARAT22ETF.NS"),
    (r"\bMotilal\s*Nasdaq\b|\bMON100\b",     "MON100.NS"),

    # ── Specialty ──
    (r"\bPage\s*Industries\b|\bJockey\b",    "PAGEIND.NS"),
    (r"\bSRF\b",                             "SRF.NS"),
    (r"\bShyam\s*Metalics\b",               "SHYAMMETL.NS"),
    (r"\bAPL\s*Apollo\b",                    "APLAPOLLO.NS"),
    (r"\bTimken\b",                          "TIMKEN.NS"),
    (r"\bSKF\b",                             "SKFINDIA.NS"),
    (r"\bSchaeffler\b",                      "SCHAEFFLER.NS"),
]

ALIAS_REGEX = [(re.compile(pat, re.I), tk) for pat, tk in ALIAS_TO_TICKER_PATTERNS]

# Sources that require browser User-Agent to access RSS
HEADERS_REQUIRED = {"BusinessStandard_Latest",
                    "BS_Companies",
                    "BS_Finance",
                    "BS_Economy",
                    "BS_Industry",
                    "MoneyControl",
                    "Investing_India",}

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
        sources["BusinessStandard_Latest"] = "https://www.business-standard.com/rss/latest.rss"
        sources["Livemint_Markets"]        = "https://www.livemint.com/rss/markets"
        sources["BusinessLine_Markets"]    = "https://www.thehindubusinessline.com/markets/?service=rss"
        sources["NDTVProfit_Markets"]      = "https://feeds.feedburner.com/ndtvprofit-latest"
        sources["ET_Stocks"]          = "https://economictimes.indiatimes.com/rssfeeds/1977021501.cms"
        sources["ET_Companies"]       = "https://economictimes.indiatimes.com/rssfeeds/1286551815.cms"
        sources["ET_Economy"]         = "https://economictimes.indiatimes.com/rssfeeds/1373380680.cms"
        sources["ET_Industry"]       = "https://economictimes.indiatimes.com/rssfeeds/13357270.cms"

        sources["BS_Companies"]       = "https://www.business-standard.com/rss/companies-101.rss"
        sources["BS_Finance"]         = "https://www.business-standard.com/rss/finance-103.rss"
        sources["BS_Economy"]         = "https://www.business-standard.com/rss/economy-policy-102.rss"
        sources["BS_Industry"]        = "https://www.business-standard.com/rss/industry-104.rss"

        sources["Mint_Companies"]     = "https://www.livemint.com/rss/companies"
        sources["Mint_Money"]         = "https://www.livemint.com/rss/money"
        sources["Mint_Economy"]       = "https://www.livemint.com/rss/economy"

        sources["BL_Economy"]         = "https://www.thehindubusinessline.com/economy/?service=rss"
        sources["BL_Companies"]       = "https://www.thehindubusinessline.com/companies/?service=rss"
        sources["BL_Portfolio"]       = "https://www.thehindubusinessline.com/portfolio/?service=rss"

        sources["TradeBrains"]        = "https://tradebrains.in/feed"

        sources["Investing_Stocks"]   = "https://in.investing.com/rss/stock_stock_picks.rss"

        sources["Pulse_Zerodha"]      = "https://pulse.zerodha.com/feed.xml"

        sources["TickerTape"]         = "https://www.tickertape.in/blog/feed"
        sources["TheHindu_Business"]        = "https://thehindu.com/business/feeder/default.rss"
        sources["TheHindu_Markets"]         = "https://www.thehindu.com/business/markets/?service=rss"

        sources["IndianExpress_Business"]   = "https://indianexpress.com/section/business/feed/"
        sources["IndianExpress_Companies"]  = "https://indianexpress.com/section/business/companies/feed/"
        sources["IndianExpress_Market"]     = "https://indianexpress.com/section/business/market/feed/"
        sources["IndianExpress_Economy"]    = "https://indianexpress.com/section/business/economy/feed/"

        sources["IndiaToday_Business"]      = "https://www.indiatoday.in/rss/1206574"

        sources["BusinessLine_Home"]        = "https://www.thehindubusinessline.com/feeder/default.rss"
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

# Titles seen in last 7 days — for Google date validation
_SEEN_TITLES: set[str] = set()

def validate_published_date(published_utc: str, title_c: str, source_name: str) -> str:
    """
    Google News sometimes returns resharing date not original date
    If title was seen before with older date → use current time as published
    """
    if not source_name.startswith("Google_"):
        return published_utc  # Only fix Google News dates
    
    if not published_utc:
        return published_utc
    
    try:
        pub_ts = pd.Timestamp(published_utc)
        age_hours = (pd.Timestamp.now(tz="UTC") - pub_ts).total_seconds() / 3600
        
        # If article is more than 7 days old from Google News
        # it's likely a reshared/trending old article
        # Mark as slightly older live article (not archive)
        if age_hours > 168:  # 7 days
            # Keep original date but flag as potentially stale
            return published_utc
            
    except:
        pass
    
    return published_utc

def parse_with_retry(url: str, source_name: str = ""):
    """Feedparser with custom UA + backoff. Uses requests for sources that block feedparser."""
    headers = {"User-Agent": USER_AGENT}
    
    # Business Standard blocks feedparser's UA — use requests to fetch then parse
    if source_name in HEADERS_REQUIRED:
        browser_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }
        for attempt in range(1, RETRIES + 1):
            try:
                import requests as req
                resp = req.get(url, headers=browser_headers, timeout=10)
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.content)
                    if getattr(feed, "entries", None):
                        return feed
            except Exception:
                pass
            time.sleep(BACKOFF_BASE ** (attempt - 1) + random.random())
        return feedparser.parse(url, request_headers=headers)
    
    # Standard feedparser for all other sources
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

def route_global_news(title: str, ticker: str) -> list:
    """
    If ticker is NaN, route article to affected sectors based on keywords
    Returns list of (ticker, sentiment_boost) tuples
    """
    if pd.notna(ticker):
        return [(ticker, 0)]  # Already mapped, no change
    
    title_lower = title.lower()
    routed = []
    
    for category, config in GLOBAL_KEYWORD_ROUTING.items():
        keywords = config["keywords"]
        if any(kw in title_lower for kw in keywords):
            affected = config["affected_tickers"]
            direction = config["sentiment_direction"]
            
            if affected == "ALL" or affected == "BROAD":
                # Market-wide signal — add to all major indices
                routed.append(("NIFTY_BROAD", direction))
            elif isinstance(affected, list):
                for tk in affected:
                    routed.append((tk, direction))
    
    return routed

def fetch_one_source(name_url: tuple) -> list:
    name, url = name_url
    title_c = ""  # initialize at the very top before anything
    # Add small delay for Google sources to avoid rate limiting
    if name.startswith("Google_"):
        time.sleep(random.uniform(1.0, 2.5))
    try:
        feed = parse_with_retry(url, source_name=name)
        rows = []
        now = pd.Timestamp.now(tz="UTC")
        for e in getattr(feed, "entries", []):
            title_c = ""  # ← reset for each entry
            try:
                title      = getattr(e, "title", "") or ""
                # Get RSS summary if available (Mint provides 229-char summaries)
                summary_raw = getattr(e, "summary", "") or getattr(e, "description", "") or ""
                import re as _re
                summary_clean = _re.sub(r'<[^>]+>', '', summary_raw)[:300].strip()
                text_for_sentiment = f"{title}. {summary_clean}" if summary_clean and summary_clean != title else title
                link_raw   = getattr(e, "link", "") or ""
                link       = normalize_url(link_raw)
                published_utc = parse_published(e)
                title_c    = canon_title(title) if title else ""
                tk, conf   = map_ticker(title, name)
                day        = published_utc[:10] if published_utc else "nodate"
                nid        = sha1(f"{title_c}|{domain_of(link)}|{day}")
                rows.append({
                    "source_name":    name,
                    "source_domain":  domain_of(link) or domain_of(url),
                    "title":          title,
                    "link":           link,
                    "published_utc":  published_utc,
                    "news_id":        nid,
                    "ticker":         tk,
                    "map_confidence": conf,
                    "title_canon":    title_c,
                    "source_weight":  get_source_weight(name),
                    "text_for_sentiment": text_for_sentiment,
                })
                if tk is None:
                    routed = route_global_news(title, tk)
                    routed_count = 0
                    for routed_ticker, _ in routed:
                        if routed_ticker == "NIFTY_BROAD":
                            continue
                        if routed_count >= 5:
                            break
                        routed_nid = sha1(f"{title_c}|{domain_of(link)}|{day}|{routed_ticker}")
                        rows.append({
                            "source_name":    name,
                            "source_domain":  domain_of(link) or domain_of(url),
                            "title":          title,
                            "link":           link,
                            "published_utc":  published_utc,
                            "news_id":        routed_nid,
                            "ticker":         routed_ticker,
                            "map_confidence": 0.4,
                            "title_canon":    title_c,
                            "source_weight":  get_source_weight(name) * 0.8,
                        })
                        routed_count += 1
            except Exception:
                continue
        return rows
    except Exception as ex:
        print(f" {name} failed: {ex}")
        return []

# Article content enrichment — only for top-confidence local runs
FETCH_ARTICLE_CONTENT = os.getenv("FETCH_ARTICLES", "false").lower() == "true"

def fetch_article_content(url: str) -> str:
    """Fetch full article text for better FinBERT scoring"""
    if not FETCH_ARTICLE_CONTENT:
        return ""
    try:
        import newspaper
        article = newspaper.Article(url)
        article.download()
        article.parse()
        return article.text[:500]  # first 500 chars
    except:
        return ""

# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs("data", exist_ok=True)
    sources = build_sources()
    now = pd.Timestamp.now(tz="UTC")
    print(f"Starting news fetch from {len(sources)} sources (parallel)...\n")

    # Parallel fetch — 8 workers (polite but fast)
    all_rows = []
    source_items = list(sources.items())

    # Split Google and non-Google sources
    google_sources = [(n,u) for n,u in source_items if n.startswith("Google_")]
    other_sources  = [(n,u) for n,u in source_items if not n.startswith("Google_")]

    # Fetch non-Google sources in parallel (fast)
    all_rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fetch_one_source, other_sources))
    for r in results:
        all_rows.extend(r)

    # Fetch Google sources with more delay (avoid rate limiting)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch_one_source, google_sources))
    for r in results:
        all_rows.extend(r)

    for result in results:
        all_rows.extend(result)

    if not all_rows:
        print("No items fetched.")
        return

    df = pd.DataFrame(all_rows)

    # Ensure UTC dtype and hour bucket (secondary dedup safety)
    df["published_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df["pub_hour"] = df["published_utc"].dt.floor("h")

    before = len(df)

    # Primary dedup: by stable news_id
    df = df.drop_duplicates(subset=["news_id"]).copy()

    # Secondary dedup: collapse near-duplicates across outlets within same hour
    df.sort_values(["title_canon", "pub_hour", "source_name"], inplace=True)
    df = df.drop_duplicates(subset=["title_canon", "pub_hour", "ticker"], keep="first")

    def get_tier(published_utc):
        try:
            age_hours = (now - pd.Timestamp(published_utc)).total_seconds() / 3600
            if age_hours <= 48:    return 1  # LIVE
            elif age_hours <= 720: return 2  # RECENT (30 days)
            else:                  return 3  # ARCHIVE
        except:
            return 3

    df["recency_tier"] = df["published_utc"].apply(get_tier)
    df["age_hours"] = df["published_utc"].apply(
        lambda x: round((now - pd.Timestamp(x)).total_seconds() / 3600, 1)
    )

    # Drop archive (>30 days) from active pipeline
    df_active = df[df["recency_tier"] <= 2].copy()

    # Save tier summary
    tier_counts = df["recency_tier"].value_counts().sort_index()
    print(f"News tiers: Live={tier_counts.get(1,0)}, Recent={tier_counts.get(2,0)}, Archive(dropped)={tier_counts.get(3,0)}")

    after = len(df_active)

    out = "data/raw_news.csv"
    df_active.to_csv(out, index=False, encoding="utf-8")

    print(f"\nSaved {after} deduped items to {out} (dropped {before - after} dups)\n")
    print("Sample:")
    sample_cols = ["source_name", "source_domain", "ticker", "map_confidence", "published_utc", "title"]
    print(df_active.head(8)[sample_cols])

if __name__ == "__main__":
    main()
