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
    "IEX", "Tata Technologies", "Indian Overseas Bank", "SJVN",
    "Tata Motors Commercial Vehicles", "ICICI Bank",
    "TCS", "Bharti Airtel", "State Bank of India", "Bajaj Finance", "Larsen Toubro",
    "Hindustan Unilever", "Sun Pharma", "Maruti Suzuki", "Mahindra Mahindra", "Wipro",
    "Axis Bank", "NTPC", "Power Grid", "Adani Enterprises", 
    "Kotak Mahindra Bank", "Bank of Baroda", "Bajaj Finserv",
    "HDFC Life Insurance", "SBI Life Insurance", "Tech Mahindra", "Mphasis",
    "Adani Ports", "Adani Green Energy", "Tata Power", "Indian Oil Corporation", "BPCL", "Gail India",
    "Asian Paints", "Dabur", "Godrej Consumer", "Titan Company", "Trent",
    "Dr Reddy Labs", "Cipla", "Divis Laboratories", "Apollo Hospitals",
    "Bajaj Auto", "Hero MotoCorp", "Eicher Motors", "Zomato", "Paytm", "Dmart", "LIC India", "DLF",
    "Shree Cement", "UltraTech Cement", "IndiGo Airlines", "Max Healthcare", "Hindalco", "JSW Steel",
    "Ambuja Cements", "ACC Cement", "Havells India", "Pidilite Industries", "Berger Paints", "Marico",
    "Colgate Palmolive India", "Britannia Industries", "United Spirits", "Muthoot Finance", "Cholamandalam Finance",
    "SBI Cards", "Indian Hotels", "InterGlobe Aviation", "Torrent Pharmaceuticals", "Lupin", "Biocon",
    "Aurobindo Pharma", "Mankind Pharma", "Zydus Lifesciences", "Jindal Steel Power", "Vedanta Aluminium", "Coal India",
    "NHPC", "SJVN", "Torrent Power", "Cummins India", "ABB India", "Siemens India", "Bharat Forge",
    "MRF Tyres", "Apollo Tyres", "Balkrishna Industries", "Page Industries", "Voltas", "Whirlpool India",
    "Info Edge India", "Naukri", "PolicyBazaar", "One97 Communications", "Ambuja Cements", "ACC", "Shree Cement", "UltraTech Cement",
    "Havells India", "Pidilite Industries", "Berger Paints", "Marico", "Colgate Palmolive India", "Britannia Industries",
    "Godrej Properties", "Oberoi Realty", "Prestige Estates", "Muthoot Finance", "Cholamandalam Finance", "SBI Cards",
    "Indian Hotels", "InterGlobe Aviation", "SpiceJet", "Torrent Pharmaceuticals", "Lupin", "Biocon", 
    "Aurobindo Pharma", "Mankind Pharma", "Zydus Lifesciences", "Jindal Steel Power", "Torrent Power", "Adani Power",
    "Cummins India", "ABB India", "Siemens India", "Bharat Forge", "MRF", "Apollo Tyres", "Balkrishna Industries",
    "Page Industries", "Voltas", "Blue Star", "Info Edge India", "PolicyBazaar", "Nykaa",
    "Vodafone Idea", "Suzlon Energy", "IRCTC", "HUDCO", "Jaiprakash Power", "NHPC",
    "Varun Beverages", "United Breweries", "Radico Khaitan", "Tata Consumer Products", "Emami", "Jyothy Labs",
    "SRF", "Aarti Industries", "Deepak Nitrite", "Navin Fluorine", "Balaji Amines",
    "Dixon Technologies", "Amber Enterprises", "Kaynes Technology", "Tata Elxsi", "KPIT Technologies", "Persistent Systems",
    "Coforge", "LTM Limited", "Mphasis", "Hindustan Aeronautics", "BEL", "Mazagon Dock",
    "Cochin Shipyard", "Garden Reach Shipbuilders", "Solar Industries", "Data Patterns",
    "HDFC AMC", "Nippon AMC", "Angel One", "BSE India", "PB Fintech", "Computer Age Management",
    "Laurus Labs", "Alkem Laboratories", "Ipca Laboratories","Ajanta Pharma", "Granules India",
    "Shyam Metalics", "APL Apollo Tubes", "Ratnamani Metals", "Tube Investments",
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
    (r"\bSpiceJet\b",                       "SPICEJET.BO"),
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
    (r"\bTata\s*Motors\s*Passenger\b|\bTAMO\b|\bTMPV\b", "TMPV.NS"),
    (r"\bTata\s*Motors\s*(?:Commercial|CV)?\b|\bTMCV\b", "TMCV.NS"),
    (r"\bICICI\s*Bank\b",                   "ICICIBANK.NS"),
    (r"\bTCS\b|\bTata\s*Consultancy\b",     "TCS.NS"),
    (r"\bBharti\s*Airtel\b|\bAirtel\b",     "BHARTIARTL.NS"),
    (r"\bSBI\b|\bState\s*Bank\b",           "SBIN.NS"),
    (r"\bBajaj\s*Finance\b",                "BAJFINANCE.NS"),
    (r"\bL&T\b|\bLarsen\b",                 "LT.NS"),
    (r"\bHUL\b|\bHindustan\s*Unilever\b",   "HINDUNILVR.NS"),
    (r"\bSun\s*Pharma\b",                   "SUNPHARMA.NS"),
    (r"\bMaruti\b|\bMaruti\s*Suzuki\b",     "MARUTI.NS"),
    (r"\bM&M\b|\bMahindra\b",              "M&M.NS"),
    (r"\bWipro\b",                          "WIPRO.NS"),
    (r"\bAxis\s*Bank\b",                    "AXISBANK.NS"),
    (r"\bNTPC\b",                           "NTPC.NS"),
    (r"\bPower\s*Grid\b",                   "POWERGRID.NS"),
    (r"\bAdani\s*Enterprises\b",            "ADANIENT.NS"),
    (r"\bKotak\s*(?:Mahindra\s*)?Bank\b",               "KOTAKBANK.NS"),
    (r"\bBank\s*of\s*Baroda\b|\bBoB\b",                 "BANKBARODA.NS"),
    (r"\bBajaj\s*Finserv\b",                            "BAJAJFINSV.NS"),
    (r"\bHDFC\s*Life\b",                                "HDFCLIFE.NS"),
    (r"\bSBI\s*Life\b",                                 "SBILIFE.NS"),
    (r"\bTech\s*Mahindra\b",                            "TECHM.NS"),
    (r"\bMphasis\b",                                    "MPHASIS.NS"),
    (r"\bAdani\s*Ports\b",                              "ADANIPORTS.NS"),
    (r"\bAdani\s*Green\b",                              "ADANIGREEN.NS"),
    (r"\bTata\s*Power\b",                               "TATAPOWER.NS"),
    (r"\bIndian\s*Oil\b|\bIOC\b",                       "IOC.NS"),
    (r"\bBPCL\b|\bBharat\s*Petroleum\b",                "BPCL.NS"),
    (r"\bGAIL\b|\bGail\s*India\b",                      "GAIL.NS"),
    (r"\bAsian\s*Paints\b",                             "ASIANPAINT.NS"),
    (r"\bDabur\b",                                      "DABUR.NS"),
    (r"\bGodrej\s*Consumer\b",                          "GODREJCP.NS"),
    (r"\bTitan\b",                                      "TITAN.NS"),
    (r"\bTrent\b",                                      "TRENT.NS"),
    (r"\bDr\s*Reddy\b",                                 "DRREDDY.NS"),
    (r"\bCipla\b",                                      "CIPLA.NS"),
    (r"\bDivi\s*(?:s|'s)?\s*Lab\b",                     "DIVISLAB.NS"),
    (r"\bApollo\s*Hospitals\b",                         "APOLLOHOSP.NS"),
    (r"\bBajaj\s*Auto\b",                               "BAJAJ-AUTO.NS"),
    (r"\bHero\s*(?:MotoCorp|Moto)\b",                   "HEROMOTOCO.NS"),
    (r"\bEicher\s*Motors\b|\bRoyal\s*Enfield\b",        "EICHERMOT.NS"),
    (r"\bZomato\b|\bEternal\b",                         "ETERNAL.NS"),
    (r"\bPaytm\b|\bOne97\b",                            "PAYTM.NS"),
    (r"\bDmart\b|\bAvenue\s*Supermarts\b",              "DMART.NS"),
    (r"\bLIC\b|\bLife\s*Insurance\s*Corporation\b",     "LICI.NS"),
    (r"\bDLF\b",                                        "DLF.NS"),
    (r"\bIRFC\b|\bIndian\s*Railway\s*Finance\b",        "IRFC.NS"),
    (r"\bNatco\s*Pharma\b",                             "NATCOPHARM.NS"),
    (r"\bShree\s*Cement\b",                             "SHREECEM.NS"),
    (r"\bUltraTech\s*Cement\b",                         "ULTRACEMCO.NS"),
    (r"\bIndiGo\b|\bInterGlobe\b",                      "INDIGO.NS"),
    (r"\bMax\s*Healthcare\b",                           "MAXHEALTH.NS"),
    (r"\bHindalco\b",                                   "HINDALCO.NS"),
    (r"\bJSW\s*Steel\b",                                "JSWSTEEL.NS"),
    (r"\bAmbuja\s*Cement\b",              "AMBUJACEM.NS"),
    (r"\bACC\s*Cement\b|\bACC\b",         "ACC.NS"),
    (r"\bHavells\b",                       "HAVELLS.NS"),
    (r"\bPidilite\b|\bFeviCol\b",          "PIDILITIND.NS"),
    (r"\bBerger\s*Paints\b",              "BERGEPAINT.NS"),
    (r"\bMarico\b|\bParachute\b",          "MARICO.NS"),
    (r"\bColgate\b",                       "COLPAL.NS"),
    (r"\bBritannia\b",                     "BRITANNIA.NS"),
    (r"\bMuthoot\s*Finance\b",            "MUTHOOTFIN.NS"),
    (r"\bChola\b|\bCholamandalam\b",       "CHOLAFIN.NS"),
    (r"\bSBI\s*Card\b",                    "SBICARD.NS"),
    (r"\bIndian\s*Hotels\b|\bTaj\b",       "INDHOTEL.NS"),
    (r"\bIndiGo\b|\bInterGlobe\b",         "INDIGO.NS"),
    (r"\bTorrent\s*Pharma\b",             "TORNTPHARM.NS"),
    (r"\bLupin\b",                         "LUPIN.NS"),
    (r"\bBiocon\b",                        "BIOCON.NS"),
    (r"\bAurobindo\b",                     "AUROPHARMA.NS"),
    (r"\bMankind\s*Pharma\b",             "MANKIND.NS"),
    (r"\bZydus\b",                         "ZYDUSLIFE.NS"),
    (r"\bJindal\s*Steel\b|\bJSPL\b",       "JINDALSTEL.NS"),
    (r"\bTorrent\s*Power\b",              "TORNTPOWER.NS"),
    (r"\bCummins\b",                       "CUMMINSIND.NS"),
    (r"\bABB\s*India\b|\bABB\b",           "ABB.NS"),
    (r"\bSiemens\b",                       "SIEMENS.NS"),
    (r"\bBharat\s*Forge\b",               "BHARATFORG.NS"),
    (r"\bMRF\b",                           "MRF.NS"),
    (r"\bApollo\s*Tyre\b",                "APOLLOTYRE.NS"),
    (r"\bBalkrishna\b|\bBKT\b",            "BALKRISIND.NS"),
    (r"\bPage\s*Industries\b|\bJockey\b",  "PAGEIND.NS"),
    (r"\bVoltas\b",                        "VOLTAS.NS"),
    (r"\bInfo\s*Edge\b|\bNaukri\b",        "NAUKRI.NS"),
    (r"\bPolicyBazaar\b|\bPB\s*Fintech\b", "POLICYBZR.NS"),
    (r"\bOne97\b|\bPaytm\b",              "PAYTM.NS"),
    (r"\bVodafone\s*Idea\b|\bVi\b",        "IDEA.NS"),
    (r"\bSuzlon\b",                        "SUZLON.NS"),
    (r"\bYes\s*Bank\b",                    "YESBANK.NS"),
    (r"\bIRCTC\b|\bIndian\s*Railway\s*Catering\b", "IRCTC.NS"),
    (r"\bHUDCO\b",                         "HUDCO.NS"),
    (r"\bJaiprakash\s*Power\b|\bJP\s*Power\b", "JPPOWER.NS"),
    # Cement
    (r"\bAmbuja\s*Cement\b",              "AMBUJACEM.NS"),
    (r"\bACC\b",                           "ACC.NS"),
    (r"\bShree\s*Cement\b",               "SHREECEM.NS"),
    (r"\bUltraTech\s*Cement\b",           "ULTRACEMCO.NS"),

    # Consumer/FMCG
    (r"\bHavells\b",                       "HAVELLS.NS"),
    (r"\bPidilite\b|\bFeviCol\b",          "PIDILITIND.NS"),
    (r"\bBerger\s*Paints\b",              "BERGEPAINT.NS"),
    (r"\bMarico\b|\bParachute\b",          "MARICO.NS"),
    (r"\bColgate\b",                       "COLPAL.NS"),
    (r"\bBritannia\b",                     "BRITANNIA.NS"),
    (r"\bTata\s*Consumer\b",              "TATACONSUM.NS"),
    (r"\bEmami\b",                         "EMAMILTD.NS"),
    (r"\bVarun\s*Beverages\b",            "VBL.NS"),
    (r"\bUnited\s*Breweries\b|\bKingfisher\b", "UBL.NS"),
    (r"\bRadico\b",                        "RADICO.NS"),
    (r"\bJyothy\b",                        "JYOTHYLAB.NS"),

    # Real Estate
    (r"\bGodrej\s*Properties\b",          "GODREJPROP.NS"),
    (r"\bOberoi\s*Realty\b",              "OBEROIRLTY.NS"),
    (r"\bPrestige\b",                      "PRESTIGE.NS"),

    # Finance
    (r"\bMuthoot\s*Finance\b",            "MUTHOOTFIN.NS"),
    (r"\bChola\b|\bCholamandalam\b",       "CHOLAFIN.NS"),
    (r"\bSBI\s*Card\b",                    "SBICARD.NS"),
    (r"\bHDFC\s*AMC\b",                    "HDFCAMC.NS"),
    (r"\bNippon\s*AMC\b|\bNippon\s*Life\b","NAM-INDIA.NS"),
    (r"\bAngel\s*One\b",                   "ANGELONE.NS"),
    (r"\bBSE\s*India\b|\bBSE\s*Ltd\b",    "BSE.NS"),
    (r"\bCAMS\b|\bComputer\s*Age\b",       "CAMS.NS"),

    # Travel/Hospitality
    (r"\bIndian\s*Hotels\b|\bTaj\b",       "INDHOTEL.NS"),
    (r"\bIndiGo\b|\bInterGlobe\b",         "INDIGO.NS"),
    (r"\bIRCTC\b",                         "IRCTC.NS"),

    # Pharma
    (r"\bTorrent\s*Pharma\b",             "TORNTPHARM.NS"),
    (r"\bLupin\b",                         "LUPIN.NS"),
    (r"\bBiocon\b",                        "BIOCON.NS"),
    (r"\bAurobindo\b",                     "AUROPHARMA.NS"),
    (r"\bMankind\s*Pharma\b",             "MANKIND.NS"),
    (r"\bZydus\b",                         "ZYDUSLIFE.NS"),
    (r"\bAlkem\b",                         "ALKEM.NS"),
    (r"\bIpca\b",                          "IPCALAB.NS"),
    (r"\bAjanta\s*Pharma\b",              "AJANTPHARM.NS"),
    (r"\bLaurus\s*Labs\b",                 "LAURUSLABS.NS"),
    (r"\bGranules\b",                      "GRANULES.NS"),

    # Energy/Power
    (r"\bAdani\s*Power\b",                "ADANIPOWER.NS"),
    (r"\bTorrent\s*Power\b",              "TORNTPOWER.NS"),
    (r"\bJaiprakash\s*Power\b|\bJP\s*Power\b","JPPOWER.NS"),
    (r"\bSuzlon\b",                        "SUZLON.NS"),

    # Industrial/Capital Goods
    (r"\bCummins\b",                       "CUMMINSIND.NS"),
    (r"\bABB\s*India\b|\bABB\b",           "ABB.NS"),
    (r"\bSiemens\b",                       "SIEMENS.NS"),
    (r"\bBharat\s*Forge\b",               "BHARATFORG.NS"),
    (r"\bSolar\s*Industries\b",           "SOLARINDS.NS"),

    # Auto/Tyres
    (r"\bMRF\b",                           "MRF.NS"),
    (r"\bApollo\s*Tyre\b",                "APOLLOTYRE.NS"),
    (r"\bBalkrishna\b|\bBKT\b",            "BALKRISIND.NS"),
    (r"\bTube\s*Investments\b|\bTI\b",    "TIINDIA.NS"),

    # Consumer Durables
    (r"\bVoltas\b",                        "VOLTAS.NS"),
    (r"\bBlue\s*Star\b",                   "BLUESTARCO.NS"),
    (r"\bPage\s*Industries\b|\bJockey\b",  "PAGEIND.NS"),
    (r"\bDixon\s*Tech\b",                  "DIXON.NS"),
    (r"\bAmber\s*Enterprises\b",          "AMBER.NS"),

    # IT/Tech
    (r"\bPersistent\s*Systems\b",         "PERSISTENT.NS"),
    (r"\bCoforge\b",                       "COFORGE.NS"),
    (r"\bLTIMindtree\b|\bLTM\b|\bLTI\b", "LTM.NS"),
    (r"\bKPIT\s*Tech\b",                   "KPITTECH.NS"),
    (r"\bKaynes\b",                        "KAYNES.NS"),
    (r"\bData\s*Patterns\b",              "DATAPATTNS.NS"),

    # Defence
    (r"\bHAL\b|\bHindustan\s*Aeronautics\b","HAL.NS"),
    (r"\bMazagon\s*Dock\b|\bMDL\b",       "MAZDOCK.NS"),
    (r"\bCochin\s*Shipyard\b",            "COCHINSHIP.NS"),
    (r"\bGarden\s*Reach\b|\bGRSE\b",      "GRSE.NS"),

    # Chemicals
    (r"\bSRF\b",                           "SRF.NS"),
    (r"\bAarti\s*Industries\b",           "AARTIIND.NS"),
    (r"\bDeeepak\s*Nitrite\b|\bDeeepak\s*Nitrate\b","DEEPAKNTR.NS"),
    (r"\bNavin\s*Fluorine\b",             "NAVINFLUOR.NS"),

    # New age tech
    (r"\bNykaa\b|\bFSN\b",               "NYKAA.NS"),
    (r"\bVodafone\s*Idea\b|\bVi\b",        "IDEA.NS"),
    (r"\bHUDCO\b",                         "HUDCO.NS"),
    (r"\bShyam\s*Metalics\b",             "SHYAMMETL.NS"),
    (r"\bAPL\s*Apollo\b",                  "APLAPOLLO.NS"),
]

ALIAS_REGEX = [(re.compile(pat, re.I), tk) for pat, tk in ALIAS_TO_TICKER_PATTERNS]

# Sources that require browser User-Agent to access RSS
HEADERS_REQUIRED = {"BusinessStandard_Latest"}

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

# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs("data", exist_ok=True)
    sources = build_sources()
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    print(f"Starting news fetch from {len(sources)} sources...\n")

    for name, url in sources.items():
        try:
            feed = parse_with_retry(url, source_name=name)
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
    def get_tier(published_utc):
        try:
            age_hours = (now - pd.Timestamp(published_utc)).total_seconds() / 3600
            if age_hours <= 48:    return 1  # LIVE
            elif age_hours <= 720: return 2  # RECENT (30 days)
            else:                  return 3  # ARCHIVE
        except:
            return 3

    df["recency_tier"] = df["published_utc"].apply(get_tier)
    df["age_hours"]    = df["published_utc"].apply(
        lambda x: round((now - pd.Timestamp(x)).total_seconds() / 3600, 1)
    )

    # Drop archive (>30 days) from active pipeline
    df_active = df[df["recency_tier"] <= 2].copy()

    # Save tier summary
    tier_counts = df["recency_tier"].value_counts().sort_index()
    print(f"News tiers: Live={tier_counts.get(1,0)}, Recent={tier_counts.get(2,0)}, Archive(dropped)={tier_counts.get(3,0)}")

    after = len(df_active)

    before_active = before  # original before dedup

    out = "data/raw_news.csv"
    df_active.to_csv(out, index=False, encoding="utf-8")  # ← save df_active, not df

    print(f"\nSaved {after} deduped items to {out} (dropped {before - after} dups)\n")
    print("Sample:")
    sample_cols = ["source_name", "source_domain", "ticker", "map_confidence", "published_utc", "title"]
    print(df_active.head(8)[sample_cols])

if __name__ == "__main__":
    main()
