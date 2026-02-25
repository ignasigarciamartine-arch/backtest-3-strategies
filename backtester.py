# -*- coding: utf-8 -*-
"""
backtester.py – Portfolio simulation with periodic rebalancing.

Approach:
  - Allocate initial capital by target weights on day 0.
  - Each day, mark-to-market the portfolio using daily returns.
  - On each rebalancing date (year-end by default), reset allocations to
    target weights at that day's total portfolio value.
"""

import logging

import numpy as np
import pandas as pd

from utils import resample_year_end

logger = logging.getLogger(__name__)


def compute_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from a price DataFrame.
    NaN on row 0 is dropped.
    """
    return prices_df.pct_change().dropna()


def simulate_portfolio(
    prices_df: pd.DataFrame,
    weights: dict,
    initial_investment: float = 1000.0,
    rebal_freq: str = "YE",
) -> pd.Series:
    """
    Vectorised portfolio simulation with annual rebalancing.

    Args:
        prices_df:          Daily EUR prices; columns = bucket names.
        weights:            Dict {bucket: float}; values must sum ≈ 1.0.
        initial_investment: Starting capital in EUR.
        rebal_freq:         Pandas offset alias (ignored; uses resample_year_end).

    Returns:
        pd.Series: Daily portfolio equity curve in EUR.
    """
    # ── 1. Normalise weights ──────────────────────────────────────────────────
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 0.005:
        logger.warning(f"Weights sum to {total_w:.4f}; normalising.")
        weights = {k: v / total_w for k, v in weights.items()}

    # ── 2. Align columns ──────────────────────────────────────────────────────
    cols = list(weights.keys())
    missing = [c for c in cols if c not in prices_df.columns]
    if missing:
        raise ValueError(f"Missing columns in prices_df: {missing}")

    df = prices_df[cols].copy().ffill().dropna()

    if df.empty:
        raise ValueError("prices_df is empty after ffill/dropna.")

    n_days = len(df)
    w      = np.array([weights[c] for c in cols], dtype=float)

    # ── 3. Daily returns array ────────────────────────────────────────────────
    # returns[0] is NaN → replaced with 0 (no change on first day)
    returns = df.pct_change().fillna(0.0).values   # shape: (n_days, n_assets)

    # ── 4. Rebalancing date indices ───────────────────────────────────────────
    rebal_df    = resample_year_end(df)
    rebal_dates = set(rebal_df.index)
    rebal_idx   = set()
    for d in rebal_dates:
        if d in df.index:
            loc = df.index.get_loc(d)
            rebal_idx.add(int(loc))

    logger.info(
        f"Simulation: {df.index[0].date()} → {df.index[-1].date()} | "
        f"{n_days} days | {len(rebal_idx)} rebalancing events"
    )

    # ── 5. Main loop ──────────────────────────────────────────────────────────
    alloc            = initial_investment * w          # current EUR value per asset
    portfolio_values = np.empty(n_days, dtype=float)
    portfolio_values[0] = initial_investment

    for i in range(1, n_days):
        alloc              = alloc * (1.0 + returns[i])
        total              = alloc.sum()
        portfolio_values[i] = total

        if i in rebal_idx:
            alloc = total * w
            logger.debug(
                f"  Rebal {df.index[i].date()}: "
                + " | ".join(f"{cols[j]}={alloc[j]:.1f}" for j in range(len(cols)))
            )

    equity = pd.Series(portfolio_values, index=df.index, name="equity_eur")

    logger.info(
        f"  Initial: €{initial_investment:,.2f} → "
        f"Final: €{portfolio_values[-1]:,.2f} "
        f"({(portfolio_values[-1]/initial_investment - 1)*100:+.1f}%)"
    )
    return equity
