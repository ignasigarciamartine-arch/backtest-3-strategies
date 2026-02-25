# -*- coding: utf-8 -*-
"""
metrics.py – Performance metrics for equity curves.

All functions accept a pd.Series with DatetimeIndex (equity curve in EUR).
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ── Core metrics ──────────────────────────────────────────────────────────────

def calculate_cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate (annualised total return)."""
    n_years      = (equity.index[-1] - equity.index[0]).days / 365.25
    if n_years <= 0:
        return float("nan")
    total_return = equity.iloc[-1] / equity.iloc[0]
    return float(total_return ** (1.0 / n_years) - 1.0)


def calculate_volatility(equity: pd.Series) -> float:
    """Annualised volatility from daily log returns."""
    log_ret = np.log(equity / equity.shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative, e.g. -0.35 = -35%)."""
    rolling_peak = equity.cummax()
    drawdown     = (equity - rolling_peak) / rolling_peak
    return float(drawdown.min())


def calculate_sharpe(equity: pd.Series, rf_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio (daily arithmetic returns)."""
    daily_returns = equity.pct_change().dropna()
    rf_daily      = rf_annual / TRADING_DAYS_PER_YEAR
    excess        = daily_returns - rf_daily

    std = excess.std()
    if std == 0.0 or np.isnan(std):
        return float("nan")

    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


# ── Additional metrics ────────────────────────────────────────────────────────

def calculate_sortino(equity: pd.Series, rf_annual: float = 0.0) -> float:
    """
    Annualised Sortino ratio.
    Downside deviation = RMS of negative-excess daily returns, annualised.
    """
    daily_returns = equity.pct_change().dropna()
    rf_daily      = rf_annual / TRADING_DAYS_PER_YEAR
    excess        = daily_returns - rf_daily

    downside = excess[excess < 0]

    if len(downside) == 0:
        return float("inf")   # no losing days

    # Semi-deviation: sqrt of mean squared negative excess returns
    downside_dev = np.sqrt((downside ** 2).mean())

    if downside_dev == 0.0 or np.isnan(downside_dev):
        return float("nan")

    return float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_worst_calendar_year(equity: pd.Series) -> float:
    """
    Worst full-calendar-year return.
    Each year: first trading day of that year → last trading day of that year.
    Partial years at the start/end of the series are included as-is.
    """
    annual_returns = []
    for _year, grp in equity.groupby(equity.index.year):
        if len(grp) >= 2:
            annual_returns.append(grp.iloc[-1] / grp.iloc[0] - 1.0)

    if not annual_returns:
        return float(equity.iloc[-1] / equity.iloc[0] - 1.0)   # single-year fallback

    return float(min(annual_returns))


def calculate_pct_days_in_drawdown(equity: pd.Series) -> float:
    """
    Percentage of trading days where portfolio is below its previous all-time high.
    """
    rolling_peak = equity.cummax()
    days_in_dd   = (equity < rolling_peak).sum()
    return float(days_in_dd / len(equity) * 100.0)


# ── Aggregator ────────────────────────────────────────────────────────────────

def calculate_all_metrics(
    equity: pd.Series,
    strategy_name: str = "",
    instruments_used: str = "",
    rf_annual: float = 0.0,
) -> dict:
    """Compute all metrics and return as a flat dictionary."""
    return {
        "strategy_name":         strategy_name,
        "start_date":            equity.index[0].strftime("%Y-%m-%d"),
        "end_date":              equity.index[-1].strftime("%Y-%m-%d"),
        "initial_eur":           round(float(equity.iloc[0]), 2),
        "final_eur":             round(float(equity.iloc[-1]), 2),
        "total_return_pct":      round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
        "cagr_pct":              round(calculate_cagr(equity) * 100, 2),
        "volatility_ann_pct":    round(calculate_volatility(equity) * 100, 2),
        "max_drawdown_pct":      round(calculate_max_drawdown(equity) * 100, 2),
        "sharpe_ratio":          round(calculate_sharpe(equity, rf_annual), 3),
        "sortino_ratio":         round(calculate_sortino(equity, rf_annual), 3),
        "worst_year_pct":        round(calculate_worst_calendar_year(equity) * 100, 2),
        "pct_days_in_drawdown":  round(calculate_pct_days_in_drawdown(equity), 1),
        "instruments_used":      instruments_used,
    }
