# -*- coding: utf-8 -*-
"""
config.py – Backtest configuration.
Edit this file to change instruments, weights, and parameters.
"""

from datetime import datetime, timedelta

# ─── Global Parameters ────────────────────────────────────────────────────────

INITIAL_INVESTMENT = 1000.0   # EUR
REBAL_FREQ         = "YE"     # Annual rebalancing; pandas offset alias
RISK_FREE_RATE     = 0.0      # Annual risk-free rate for Sharpe ratio

END_DATE        = datetime.today().strftime("%Y-%m-%d")
START_DATE_5Y   = (datetime.today() - timedelta(days=5 * 365 + 2)).strftime("%Y-%m-%d")
START_DATE_AI   = "2018-01-01"  # AI ETF approximate inception window
START_DATE_2Y   = datetime.today().replace(year=datetime.today().year - 2).strftime("%Y-%m-%d")

# Refinitiv fields to attempt in order (RI = total return index, preferred)
PRICE_FIELDS = ["RI", "TRDPRC_1", "CF_LAST"]

# ─── Disk cache ───────────────────────────────────────────────────────────────
# Set env REFRESH=1 or pass --refresh to bypass cache and re-download everything.
CACHE_ENABLED = True

# If True, try Yahoo Finance when Refinitiv has no usable series (or no session).
USE_YFINANCE_FALLBACK = True

# If True, all strategies are clipped to a single shared overlap period.
FORCE_COMMON_PERIOD = True

# ─── Strategy 3 minimum period ────────────────────────────────────────────────
# Abort Strategy 3 (AI & Digital Assets) if aligned data < this many trading days.
MIN_STRATEGY3_TRADING_DAYS = 252

# ─── FX sanity ranges ─────────────────────────────────────────────────────────
# Wide ranges to detect gross directional errors (e.g. USDEUR fed as EURUSD).
# Format: {fx_ric: (min_plausible_median, max_plausible_median)}
# These correspond to the "direct" interpretation of each Refinitiv FX RIC.
FX_SANITY_RANGES = {
    "EUR=":  (0.70, 1.60),   # EURUSD – USD per 1 EUR
    "GBP=":  (1.00, 2.00),   # GBPUSD – USD per 1 GBP
    "HKD=":  (0.08, 0.20),   # HKDUSD – USD per 1 HKD  (~1/7.8 ≈ 0.128)
}

# ─── FX Configuration ─────────────────────────────────────────────────────────
# Refinitiv FX RICs for non-EUR currencies expressed as [CCY]/USD
# EUR= → EURUSD (USD per 1 EUR, e.g. 1.08)
# GBP= → GBPUSD (USD per 1 GBP, e.g. 1.27)
# HKD= → HKDUSD (USD per 1 HKD, e.g. 0.128)
# Conversion: EUR_price = foreign_price * (CCY_USD / EUR_USD)

EUR_USD_RIC = "EUR="
CCY_USD_RICS = {
    "USD": None,    # base currency for conversion pipeline
    "GBP": "GBP=",
    "HKD": "HKD=",
}

# ─── Instrument Buckets ───────────────────────────────────────────────────────
# Each bucket lists RIC candidates in priority order.
# The fetcher tries them top-to-bottom; uses the first one that returns data.
# 'currency' is the price currency of each RIC (before EUR conversion).

BUCKETS = {

    "sp500_ew": {
        "description": "S&P 500 Equal Weight",
        "search_hint": "UCITS ETF S&P 500 Equal Weight EUR accumulating",
        "instruments": {
            "XDEW.DE":    "EUR",   # Xtrackers S&P 500 EW UCITS (Xetra)
            "EWSP.L":     "GBP",   # Invesco S&P 500 EW UCITS (LSE)
            "SP5EUDP.PA": "EUR",   # SPDR S&P 500 EW UCITS (Euronext Paris)
            "SPEQ.L":     "USD",   # Invesco S&P 500 EW UCITS USD class
        },
    },

    "msci_europe": {
        "description": "MSCI Europe",
        "search_hint": "UCITS ETF MSCI Europe EUR accumulating",
        "instruments": {
            "IMEU.AS": "EUR",   # iShares Core MSCI Europe (Amsterdam)
            "MEUD.PA": "EUR",   # Lyxor MSCI Europe Acc (Paris)
            "EUN.L":   "GBP",   # iShares MSCI Europe (LSE)
            "LCWE.PA": "EUR",   # Amundi MSCI Europe (Paris)
        },
    },

    "msci_em": {
        "description": "MSCI Emerging Markets",
        "search_hint": "UCITS ETF MSCI Emerging Markets EUR accumulating",
        "instruments": {
            "EIMI.AS": "EUR",   # iShares Core MSCI EM IMI (Amsterdam)
            "IEEM.AS": "EUR",   # iShares MSCI EM UCITS (Amsterdam)
            "EMEA.L":  "GBP",   # iShares MSCI EM (LSE)
            "EMIM.AS": "EUR",   # iShares Core MSCI EM IMI alt share class
        },
    },

    "msci_ageing": {
        "description": "MSCI World Ageing Population",
        "search_hint": "UCITS ETF MSCI World Ageing Population EUR accumulating",
        "instruments": {
            "2B7I.DE": "EUR",   # iShares Ageing Population UCITS (Xetra)
            "AGED.L":  "GBP",   # iShares Ageing Population UCITS (LSE)
            "DPAG.DE": "EUR",   # alternative ticker Xetra
            "AGNG.L":  "GBP",   # alternative LSE ticker
        },
    },

    "euro_govbond_1_5y": {
        "description": "Euro Government Bonds 1-5Y",
        "search_hint": "UCITS ETF Euro Government Bond 1-5 year EUR accumulating",
        "instruments": {
            "IBGS.AS": "EUR",   # iShares Euro Govt Bond 1-5yr (Amsterdam)
            "EXX6.DE": "EUR",   # iShares Euro Govt Bond 1-5yr (Xetra)
            "CBUG.PA": "EUR",   # Amundi Euro Govt Bond 1-3Y (Paris)
            "EUSC.DE": "EUR",   # iShares Euro Govt Bond 0-1yr (similar)
        },
    },

    "msci_india": {
        "description": "MSCI India",
        "search_hint": "UCITS ETF MSCI India EUR accumulating",
        "instruments": {
            "NDIA.L":  "USD",   # iShares MSCI India UCITS (LSE, USD)
            "CIND.L":  "GBP",   # iShares MSCI India UCITS (LSE, GBP)
            "SAIN.L":  "GBP",   # iShares MSCI India UCITS alt
            "INDIA.AS":"EUR",   # possible EUR-listed India ETF
        },
    },

    "nasdaq100": {
        "description": "Nasdaq 100",
        "search_hint": "UCITS ETF Nasdaq 100 EUR accumulating",
        "instruments": {
            "CNDX.L":  "USD",   # iShares Nasdaq-100 UCITS (LSE, USD)
            "EQQQ.DE": "EUR",   # Invesco EQQQ Nasdaq-100 UCITS (Xetra)
            "SXRV.DE": "EUR",   # iShares Nasdaq-100 EUR Hedged (Xetra)
            "QDVE.DE": "EUR",   # iShares S&P 500 Info Tech (proxy, Xetra)
        },
    },

    "msci_china": {
        "description": "MSCI China",
        "search_hint": "UCITS ETF MSCI China EUR accumulating",
        "instruments": {
            "CNYA.L":  "USD",   # iShares MSCI China UCITS (LSE, USD)
            "ICGI.L":  "GBP",   # iShares MSCI China UCITS (LSE, GBP)
            "MXCN.L":  "GBP",   # MSCI China alternative
            "HMCD.PA": "EUR",   # HSBC MSCI China (Paris)
        },
    },

    "ai_etf": {
        "description": "Artificial Intelligence & Big Data",
        "search_hint": "UCITS ETF Artificial Intelligence Big Data EUR accumulating",
        "instruments": {
            "L0QM.DE": "EUR",   # WisdomTree AI UCITS ETF (Xetra) – launched ~2019
            "XAIX.DE": "EUR",   # Xtrackers AI & Big Data UCITS (Xetra) – launched ~2019
            "2B7K.DE": "EUR",   # iShares Automation & Robotics alt Xetra
            "AIAI.L":  "GBP",   # WisdomTree AI UCITS (LSE)
            "WTAI.L":  "GBP",   # WisdomTree AI alt LSE
        },
    },

    "bitcoin_eur": {
        "description": "Bitcoin Spot EUR",
        "search_hint": "Bitcoin ETP ETN EUR physical backed Xetra",
        "instruments": {
            "BTC=BTSP": "USD",   # Bitcoin spot/index feed (Refinitiv) - long history
            "ABTC.L":  "USD",   # 21Shares Bitcoin ETP (LSE, USD) – confirmed working
            "BITC.L":  "USD",   # Bitcoin ETP (LSE, USD) – confirmed working
            "BTCE.L":  "GBP",   # ETC Group Physical Bitcoin (LSE, GBP) – confirmed working
            "BTCE.DE": "EUR",   # ETC Group Physical Bitcoin (Xetra) – fallback
            "VBTC.DE": "EUR",   # VanEck Bitcoin ETN (Xetra) – fallback
            "BTC1.DE": "EUR",   # alternative Bitcoin ETP (Xetra) – fallback
            "ZBTC.DE": "EUR",   # 21Shares Bitcoin ETP (Xetra) – fallback
        },
    },

    "hang_seng_tech": {
        "description": "Hang Seng TECH Index",
        "search_hint": "UCITS ETF Hang Seng TECH Index EUR or USD accumulating",
        "instruments": {
            "HTEC.L":  "USD",   # CSOP Hang Seng TECH UCITS (LSE, USD)
            "HSTE.L":  "USD",   # Hang Seng TECH alternative (LSE)
            "KTEC.L":  "USD",   # Krane Shares HS TECH (LSE)
            "3032.HK": "HKD",   # CSOP HS TECH ETF on HKEX
        },
    },
}

# ─── Strategy Definitions ─────────────────────────────────────────────────────
# weights must be bucket keys from BUCKETS above and must sum to 1.0.

STRATEGIES = {

    "strategy_1": {
        "name": "Cartera Global Diversificada",
        "start_date": START_DATE_2Y,
        "end_date":   END_DATE,
        "weights": {
            "sp500_ew":          0.25,
            "msci_europe":       0.25,
            "msci_em":           0.15,
            "msci_ageing":       0.15,
            "euro_govbond_1_5y": 0.20,
        },
    },

    "strategy_2": {
        "name": "Creixement Mercats Emergents",
        "start_date": START_DATE_2Y,
        "end_date":   END_DATE,
        "weights": {
            "msci_em":    0.35,
            "msci_india": 0.25,
            "nasdaq100":  0.20,
            "msci_china": 0.20,
        },
    },

    "strategy_3": {
        "name": "IA i Actius Digitals",
        "start_date": START_DATE_2Y,
        "end_date":   END_DATE,
        "weights": {
            "ai_etf":        0.50,
            "bitcoin_eur":   0.30,
            "hang_seng_tech":0.20,
        },
    },
}

# ─── Output paths (relative to project root) ──────────────────────────────────
OUTPUT_TABLES = "output/tables"
OUTPUT_CHARTS = "output/charts"
LOGS_DIR      = "logs"
DATA_RAW_DIR  = "data/raw"
DATA_PROC_DIR = "data/processed"
