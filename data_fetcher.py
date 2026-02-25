# -*- coding: utf-8 -*-
"""
data_fetcher.py â€“ Refinitiv data access with:
  - Automatic RIC fallback within each bucket.
  - Disk cache (data/raw/): bypass with REFRESH=1 env or --refresh flag.
  - FX direction sanity-check: detects and corrects inverted FX pairs.
  - Explicit FX failure: never silently skips currency conversion.
  - Total Return field priority: RI â†’ TRDPRC_1 â†’ CF_LAST.
"""

import os
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# â”€â”€ Library detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_rd  = None   # refinitiv.data handle
_ek  = None   # eikon handle
_yf  = None   # yfinance handle
_lib = None   # "rd" | "eikon" | None

def _detect_library() -> str | None:
    global _rd, _ek, _lib
    if _lib is not None:
        return _lib
    try:
        import refinitiv.data as rd
        _rd  = rd
        _lib = "rd"
        logger.info("Refinitiv library: refinitiv.data")
        return _lib
    except ImportError:
        pass
    try:
        import eikon as ek
        _ek  = ek
        _lib = "eikon"
        logger.info("Refinitiv library: eikon")
        return _lib
    except ImportError:
        pass
    _lib = None
    logger.error("Neither refinitiv.data nor eikon is installed.")
    return _lib


def _detect_yfinance() -> bool:
    """Detect optional yfinance provider for fallback data pulls."""
    global _yf
    if _yf is not None:
        return True
    try:
        import yfinance as yf
        cache_dir = os.path.join("data", "yfinance_cache")
        os.makedirs(cache_dir, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(cache_dir)
        _yf = yf
        logger.info("Fallback provider available: yfinance")
        return True
    except ImportError:
        return False


# â”€â”€ Session management â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def open_connection() -> None:
    """Open a Refinitiv session. Reads APP_KEY from environment if available."""
    lib = _detect_library()
    has_yf = _detect_yfinance()
    app_key = os.environ.get("APP_KEY", "").strip()

    if lib == "rd":
        try:
            if app_key:
                logger.info(f"Opening rd session with APP_KEY ...{app_key[-6:]}")
                _rd.open_session(app_key=app_key)
            else:
                logger.info("Opening rd desktop session (no APP_KEY - Workspace proxy).")
                _rd.open_session()
        except Exception as exc:
            if has_yf:
                logger.warning(
                    f"Refinitiv session open failed ({exc}); continuing with yfinance fallback."
                )
            else:
                raise EnvironmentError(
                    f"Cannot open Refinitiv session: {exc}\n"
                    "Install yfinance for fallback or fix Workspace/session."
                ) from exc

    elif lib == "eikon":
        try:
            if app_key:
                _ek.set_app_key(app_key)
            else:
                logger.warning("eikon: no APP_KEY set. Set APP_KEY env variable.")
            logger.info("Eikon session ready.")
        except Exception as exc:
            if has_yf:
                logger.warning(
                    f"Eikon session setup failed ({exc}); continuing with yfinance fallback."
                )
            else:
                raise EnvironmentError(
                    f"Cannot open Eikon session: {exc}\n"
                    "Install yfinance for fallback or fix APP_KEY/session."
                ) from exc

    else:
        if has_yf:
            logger.warning("Refinitiv not available; using yfinance fallback mode.")
        else:
            raise EnvironmentError(
                "Cannot open Refinitiv session: no compatible library found.\n"
                "Install: pip install refinitiv-data or yfinance"
            )


def close_connection() -> None:
    """Close the active Refinitiv session."""
    if _lib == "rd":
        try:
            _rd.close_session()
            logger.info("Refinitiv session closed.")
        except Exception as e:
            logger.warning(f"Session close warning: {e}")


# â”€â”€ Disk cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _cache_csv_path(bucket_name: str) -> str:
    return os.path.join("data", "raw", f"{bucket_name}.csv")

def _cache_meta_path(bucket_name: str) -> str:
    return os.path.join("data", "raw", f"{bucket_name}_meta.json")


def _load_series_cache(bucket_name: str) -> tuple[pd.Series | None, str | None]:
    """Load EUR price series and RIC metadata from disk cache."""
    csv_path  = _cache_csv_path(bucket_name)
    meta_path = _cache_meta_path(bucket_name)

    if not os.path.exists(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        series = df.iloc[:, 0].sort_index()
        series.name = bucket_name

        ric = "cached"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            ric = meta.get("ric", "cached")

        logger.info(
            f"  âœ” Cache HIT â€“ '{bucket_name}' ({ric}): "
            f"{len(series)} pts  {series.index[0].date()} â†’ {series.index[-1].date()}"
        )
        return series, ric
    except Exception as exc:
        logger.warning(f"  Cache load failed for '{bucket_name}': {exc}. Will re-download.")
        return None, None


def _save_series_cache(bucket_name: str, series: pd.Series, ric: str, field: str,
                       start: str, end: str) -> None:
    """Save EUR price series and metadata to disk cache."""
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    csv_path  = _cache_csv_path(bucket_name)
    meta_path = _cache_meta_path(bucket_name)

    series.to_csv(csv_path, header=True, index_label="date")
    with open(meta_path, "w") as f:
        json.dump({"ric": ric, "field": field, "start": start, "end": end,
                   "n_pts": len(series)}, f, indent=2)


# â”€â”€ Low-level data fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fetch_rd(ric: str, field: str, start: str, end: str) -> pd.Series | None:
    """Fetch one field for one RIC via refinitiv.data get_history."""
    try:
        df = _rd.get_history(universe=ric, fields=[field], start=start, end=end)
        if df is None or df.empty:
            return None

        # Handle potential MultiIndex columns (e.g. (ric, field))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)

        if field in df.columns:
            s = df[field].dropna()
        elif len(df.columns) == 1:
            s = df.iloc[:, 0].dropna()   # accept whatever column came back
        else:
            return None

        return s.sort_index() if not s.empty else None
    except Exception as e:
        logger.debug(f"    rd.get_history({ric}, {field}): {e}")
        return None


# Eikon's get_timeseries uses OHLCV column names, not Refinitiv field codes.
_EIKON_FIELD_ALIAS = {
    "RI":       "CLOSE",
    "TRDPRC_1": "CLOSE",
    "CF_LAST":  "CLOSE",
}

def _fetch_eikon(ric: str, field: str, start: str, end: str) -> pd.Series | None:
    """Fetch one field for one RIC via eikon get_timeseries."""
    eikon_field = _EIKON_FIELD_ALIAS.get(field, field)
    try:
        df = _ek.get_timeseries(
            rics=ric,
            fields=[eikon_field],
            start_date=start,
            end_date=end,
            interval="daily",
        )
        if df is None or df.empty:
            return None
        col = eikon_field if eikon_field in df.columns else df.columns[0]
        s   = df[col].dropna()
        return s.sort_index() if not s.empty else None
    except Exception as e:
        logger.debug(f"    eikon.get_timeseries({ric}, {field}): {e}")
        return None


def _raw_fetch(ric: str, field: str, start: str, end: str) -> pd.Series | None:
    """Dispatch to the available library."""
    if _lib == "rd":
        return _fetch_rd(ric, field, start, end)
    if _lib == "eikon":
        return _fetch_eikon(ric, field, start, end)
    return None


_FX_RIC_TO_YF = {
    "EUR=": "EURUSD=X",
    "GBP=": "GBPUSD=X",
    "HKD=": "HKDUSD=X",
}


def _ric_to_yf_ticker(ric: str) -> str | None:
    """Best-effort RIC -> Yahoo ticker mapping."""
    if ric in _FX_RIC_TO_YF:
        return _FX_RIC_TO_YF[ric]
    if ric.endswith("="):
        return None
    return ric


def _fetch_yfinance_series(ric: str, start: str, end: str) -> tuple[pd.Series | None, str | None]:
    """Fetch daily adjusted/close history from yfinance as fallback."""
    if not _detect_yfinance():
        return None, None

    ticker = _ric_to_yf_ticker(ric)
    if not ticker:
        return None, None

    try:
        df = _yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None, None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        for col in ("Adj Close", "Close"):
            if col in df.columns:
                s = df[col].dropna().sort_index()
                if len(s) >= 5:
                    field_used = "YF_ADJ_CLOSE" if col == "Adj Close" else "YF_CLOSE"
                    return s, field_used
        return None, None
    except Exception as exc:
        logger.debug(f"    yfinance({ticker}) failed: {exc}")
        return None, None


def _to_weekday_series(s: pd.Series | None) -> pd.Series | None:
    """Keep a clean, sorted business-day series (Mon-Fri only)."""
    if s is None or s.empty:
        return None
    out = s.copy().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out[out.index.dayofweek < 5]
    return out if not out.empty else None


def fetch_series(ric: str, start: str, end: str) -> tuple[pd.Series | None, str | None]:
    """
    Try price fields in priority order (RI â†’ TRDPRC_1 â†’ CF_LAST).

    Returns:
        (series, field_used)  or  (None, None)
    """
    from config import PRICE_FIELDS, USE_YFINANCE_FALLBACK

    # Approximate expected trading days for missing-% computation
    try:
        expected_days = len(pd.bdate_range(start, end))
    except Exception:
        expected_days = 0

    for field in PRICE_FIELDS:
        s = _to_weekday_series(_raw_fetch(ric, field, start, end))
        if s is not None and len(s) >= 20:
            pct_missing = (
                max(0.0, 1.0 - len(s) / expected_days) * 100
                if expected_days > 0 else 0.0
            )
            logger.info(
                f"    âœ“ {ric} [{field}]: {len(s)} pts, "
                f"~{pct_missing:.1f}% missing  "
                f"{s.index[0].date()} â†’ {s.index[-1].date()}"
            )
            return s, field

    if USE_YFINANCE_FALLBACK:
        s, field = _fetch_yfinance_series(ric, start, end)
        s = _to_weekday_series(s)
        if s is not None and len(s) >= 20:
            pct_missing = (
                max(0.0, 1.0 - len(s) / expected_days) * 100
                if expected_days > 0 else 0.0
            )
            logger.info(
                f"    ✓ {ric} [{field}]: {len(s)} pts, "
                f"~{pct_missing:.1f}% missing  "
                f"{s.index[0].date()} → {s.index[-1].date()} (yfinance fallback)"
            )
            return s, field

    logger.warning(f"    âœ— {ric}: no usable data for fields {PRICE_FIELDS}")
    return None, None


# â”€â”€ FX helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _check_and_orient_fx(fx_series: pd.Series, fx_ric: str) -> pd.Series:
    """
    Sanity-check the FX series against known plausible ranges.
    If the median is outside the expected range but 1/median is inside,
    the series is inverted and a WARNING is logged.
    Returns a correctly-oriented series.
    """
    from config import FX_SANITY_RANGES

    if fx_ric not in FX_SANITY_RANGES:
        return fx_series   # no known range â†’ trust the data

    lo, hi     = FX_SANITY_RANGES[fx_ric]
    median_val = float(fx_series.median())

    if lo <= median_val <= hi:
        logger.debug(f"    FX {fx_ric}: median={median_val:.4f} âˆˆ [{lo}, {hi}] âœ“")
        return fx_series

    # Try the inverse
    inv_median = 1.0 / median_val if median_val != 0 else float("nan")
    if lo <= inv_median <= hi:
        logger.warning(
            f"    FX {fx_ric}: median={median_val:.4f} is OUTSIDE expected range [{lo}, {hi}]. "
            f"Inverting series (1/x); inv_median={inv_median:.4f} is plausible. "
            f"Double-check the RIC direction in config."
        )
        return 1.0 / fx_series

    # Neither direction looks right â€“ use as-is with a loud warning
    logger.warning(
        f"    FX {fx_ric}: median={median_val:.4f} is outside [{lo}, {hi}] and "
        f"its inverse ({inv_median:.4f}) is also outside. "
        f"Using as-is â€“ verify RIC '{fx_ric}' manually."
    )
    return fx_series


def _fetch_fx(ric: str, start: str, end: str, fx_cache: dict) -> pd.Series | None:
    """Fetch an FX rate series (with in-memory cache and direction sanity check)."""
    from config import USE_YFINANCE_FALLBACK

    if ric in fx_cache:
        return fx_cache[ric]

    for field in ["CF_LAST", "TRDPRC_1", "BID"]:
        s = _to_weekday_series(_raw_fetch(ric, field, start, end))
        if s is not None and len(s) >= 5:
            s = _check_and_orient_fx(s, ric)
            logger.info(f"    FX {ric} [{field}]: {len(s)} pts")
            fx_cache[ric] = s.sort_index()
            return fx_cache[ric]

    if USE_YFINANCE_FALLBACK:
        s, field = _fetch_yfinance_series(ric, start, end)
        s = _to_weekday_series(s)
        if s is not None and len(s) >= 5:
            s = _check_and_orient_fx(s, ric)
            logger.info(f"    FX {ric} [{field}]: {len(s)} pts (yfinance fallback)")
            fx_cache[ric] = s.sort_index()
            return fx_cache[ric]

    logger.error(f"    FX fetch FAILED for RIC '{ric}'.")
    return None


def _align_fx(fx_series: pd.Series, price_series: pd.Series) -> pd.Series:
    """Reindex FX series to price_series dates, forward- then back-filling gaps."""
    return fx_series.reindex(price_series.index).ffill().bfill()


def convert_to_eur(
    price_series: pd.Series,
    currency: str,
    fx_cache: dict,
    start: str,
    end: str,
    ric_label: str = "?",
) -> pd.Series | None:
    """
    Convert price_series from `currency` to EUR.

    Returns EUR pd.Series, or None if conversion is impossible (NEVER silent skip).

    Pipeline:
      EUR  â†’ return as-is.
      USD  â†’ divide by EURUSD  (EUR= gives USD per 1 EUR).
      GBP/HKD â†’ multiply by CCY/USD, then divide by EUR/USD.
    """
    from config import EUR_USD_RIC, CCY_USD_RICS

    if currency == "EUR":
        return price_series

    # â”€â”€ 1. Fetch EURUSD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    eurusd = _fetch_fx(EUR_USD_RIC, start, end, fx_cache)
    if eurusd is None:
        logger.error(
            f"    âœ– EUR/USD rate unavailable (RIC: {EUR_USD_RIC}). "
            f"Cannot convert {currency}â†’EUR for '{ric_label}'. "
            f"Check Refinitiv connection or add a manual currency override."
        )
        return None

    eurusd_a = _align_fx(eurusd, price_series)

    if currency == "USD":
        return price_series / eurusd_a

    # â”€â”€ 2. Fetch CCY/USD cross-rate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ccy_usd_ric = CCY_USD_RICS.get(currency)
    if ccy_usd_ric is None:
        logger.error(
            f"    âœ– No FX RIC configured for currency '{currency}' (bucket '{ric_label}'). "
            f"Add it to CCY_USD_RICS in config.py."
        )
        return None

    ccy_usd = _fetch_fx(ccy_usd_ric, start, end, fx_cache)
    if ccy_usd is None:
        logger.error(
            f"    âœ– {currency}/USD rate unavailable (RIC: {ccy_usd_ric}). "
            f"Cannot convert {currency}â†’EUR for '{ric_label}'."
        )
        return None

    ccy_usd_a   = _align_fx(ccy_usd, price_series)
    price_in_usd = price_series * ccy_usd_a      # foreign â†’ USD
    return price_in_usd / eurusd_a               # USD â†’ EUR


# â”€â”€ Public utility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def suggest_search_text(bucket_name: str) -> str:
    """Return a Refinitiv search hint for a failed bucket."""
    from config import BUCKETS
    hint = BUCKETS.get(bucket_name, {}).get("search_hint", f"UCITS ETF {bucket_name} EUR accumulating")
    return f"Busca a Refinitiv: {hint}"


# â”€â”€ Bucket fetch with fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_bucket_with_fallback(
    bucket_name: str,
    bucket_cfg: dict,
    start: str,
    end: str,
    fx_cache: dict,
    refresh: bool = False,
) -> tuple[pd.Series | None, str | None]:
    """
    Try each RIC candidate in the bucket until one returns valid EUR data.
    Uses disk cache unless `refresh=True`.

    Returns:
        (series_in_eur, ric_used)  or  (None, None) if all candidates fail.
    """
    from config import CACHE_ENABLED

    instruments = bucket_cfg["instruments"]    # {ric: currency}
    description = bucket_cfg["description"]

    logger.info(f"\nâ”€â”€ Bucket '{bucket_name}' ({description}) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")

    # â”€â”€ Try disk cache first â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if CACHE_ENABLED and not refresh:
        cached_series, cached_ric = _load_series_cache(bucket_name)
        if cached_series is not None:
            return cached_series, cached_ric

    # â”€â”€ Download from Refinitiv â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    used_field = None
    for ric, currency in instruments.items():
        logger.info(f"  Trying {ric} ({currency}) â€¦")

        series, field = fetch_series(ric, start, end)
        if series is None:
            continue

        # Convert to EUR (explicit failure â†’ try next RIC)
        if currency != "EUR":
            logger.info(f"    Converting {currency} â†’ EUR for {ric}")
            series = convert_to_eur(series, currency, fx_cache, start, end, ric_label=ric)
            if series is None:
                logger.warning(f"    FX conversion failed for {ric}; trying next candidate.")
                continue

        used_field = field
        logger.info(f"  âœ” Bucket '{bucket_name}' resolved â†’ {ric} [{field}]")

        # Save to cache
        _save_series_cache(bucket_name, series, ric, field, start, end)
        return series, ric

    # â”€â”€ All candidates exhausted â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    search_txt = suggest_search_text(bucket_name)
    logger.error(
        f"\n  âœ– BUCKET FAILED: '{bucket_name}' â€“ "
        f"all {len(instruments)} RICs returned no usable data."
    )
    logger.error(f"  â†’ {search_txt}")
    logger.error(f"  â†’ Candidats provats: {list(instruments.keys())}")
    print(
        f"\n{'-'*60}\n"
        f"WARNING: Bucket '{bucket_name}' ({description}): cap RIC disponible.\n"
        f"   {search_txt}\n"
        f"   RICs provats: {', '.join(instruments.keys())}\n"
        f"{'-'*60}\n"
    )
    return None, None

