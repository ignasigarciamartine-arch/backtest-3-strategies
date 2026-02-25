# -*- coding: utf-8 -*-
"""
utils.py – Logging, directory creation, and pandas-version-safe helpers.
"""

import os
import logging
import pandas as pd


def setup_logging(log_dir: str = "logs", filename: str = "backtest.log") -> str:
    """
    Configure root logger to write to file and console simultaneously.
    Returns the path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def create_project_directories(dirs: list) -> None:
    """Create a list of directories (including parents) if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_all_project_dirs() -> list:
    """Return all directories required by the project."""
    from config import OUTPUT_TABLES, OUTPUT_CHARTS, LOGS_DIR, DATA_RAW_DIR, DATA_PROC_DIR
    return [DATA_RAW_DIR, DATA_PROC_DIR, OUTPUT_TABLES, OUTPUT_CHARTS, LOGS_DIR]


def resample_year_end(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a DataFrame to year-end frequency.
    Handles pandas >= 2.2 ('YE') and older versions ('A').

    The returned index always contains ACTUAL trading dates from df
    (last trading day of each calendar year), never synthetic Dec 31 dates.
    This guarantees that `if d in df.index` lookups in the backtester work.
    """
    for alias in ("YE", "Y", "A-DEC", "A"):
        try:
            return df.resample(alias).last()
        except ValueError:
            continue
    # Fallback: select the last actual trading date per calendar year.
    # Using the index itself avoids groupby.apply deprecation warnings in
    # pandas 2.2+ and ensures the resulting index matches df.index exactly.
    last_dates = df.index.to_series().groupby(df.index.year).last()
    return df.loc[last_dates.values]


def align_series(series_dict: dict, method: str = "ffill") -> pd.DataFrame:
    """
    Combine multiple Series (keyed by name) into one DataFrame.
    Forward-fills gaps (e.g. different holiday calendars) then drops remaining NaN rows.
    """
    df = pd.DataFrame(series_dict)
    if method == "ffill":
        df = df.ffill()
    return df.dropna()
