# -*- coding: utf-8 -*-
"""
main.py – Backtest orchestrator.

Usage:
    python main.py              # use disk cache if available
    python main.py --refresh    # force re-download (bypass cache)

Environment variables:
    APP_KEY=<key>   Refinitiv app key (optional for desktop/Workspace sessions)
    REFRESH=1       Same as --refresh

Outputs:
    data/raw/                    raw EUR price CSV + metadata per bucket
    output/tables/               prices, returns, equity curves, metrics CSVs
    output/charts/equity_plot.png
    logs/backtest.log
"""

import os
import sys
import logging

# ── Ensure working directory = project root ───────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from config import (
    STRATEGIES, BUCKETS, INITIAL_INVESTMENT, REBAL_FREQ, RISK_FREE_RATE,
    OUTPUT_TABLES, OUTPUT_CHARTS, LOGS_DIR,
    FORCE_COMMON_PERIOD,
    MIN_STRATEGY3_TRADING_DAYS,
)
from utils import (
    setup_logging, create_project_directories, get_all_project_dirs, align_series,
)
from data_fetcher import (
    open_connection, close_connection,
    fetch_bucket_with_fallback, suggest_search_text,
)
from backtester import simulate_portfolio, compute_returns
from metrics import calculate_all_metrics


# ── Refresh flag: --refresh or REFRESH=1 ─────────────────────────────────────
REFRESH = (
    "--refresh" in sys.argv
    or os.environ.get("REFRESH", "").lower() in ("1", "true", "yes")
)

# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Setup ────────────────────────────────────────────────────────────────
    create_project_directories(get_all_project_dirs())
    log_path = setup_logging(LOGS_DIR)
    logger   = logging.getLogger(__name__)

    logger.info("=" * 65)
    logger.info("  BACKTEST  –  3 Strategies  –  Refinitiv Data")
    logger.info("=" * 65)
    logger.info(f"Project root : {PROJECT_ROOT}")
    logger.info(f"Log file     : {log_path}")
    logger.info(f"Refresh mode : {'ON (re-downloading)' if REFRESH else 'OFF (using cache)'}")

    # 2. Open Refinitiv session ───────────────────────────────────────────────
    try:
        open_connection()
    except EnvironmentError as exc:
        logger.critical(str(exc))
        sys.exit(1)

    # 3. Determine broadest date window per bucket ────────────────────────────
    bucket_date_windows: dict[str, dict] = {}
    for strat in STRATEGIES.values():
        for bucket in strat["weights"]:
            if bucket not in bucket_date_windows:
                bucket_date_windows[bucket] = {
                    "start": strat["start_date"],
                    "end":   strat["end_date"],
                }
            else:
                bucket_date_windows[bucket]["start"] = min(
                    bucket_date_windows[bucket]["start"], strat["start_date"]
                )
                bucket_date_windows[bucket]["end"] = max(
                    bucket_date_windows[bucket]["end"], strat["end_date"]
                )

    # 4. Download / load all required buckets ─────────────────────────────────
    fx_cache:       dict[str, pd.Series] = {}
    bucket_series:  dict[str, pd.Series] = {}
    bucket_ric_map: dict[str, str]       = {}

    logger.info("\n── PHASE 1: DATA DOWNLOAD ──────────────────────────────────")
    for bucket_name, dates in bucket_date_windows.items():
        series, ric = fetch_bucket_with_fallback(
            bucket_name,
            BUCKETS[bucket_name],
            dates["start"],
            dates["end"],
            fx_cache,
            refresh=REFRESH,
        )
        if series is not None:
            bucket_series[bucket_name]  = series
            bucket_ric_map[bucket_name] = ric
        else:
            logger.error(
                f"Bucket '{bucket_name}' unavailable – "
                f"strategies using it will be skipped.\n"
                f"  {suggest_search_text(bucket_name)}"
            )

    # 5. Run backtest per strategy ─────────────────────────────────────────────
    common_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    common_dates_all: pd.DatetimeIndex | None = None
    if FORCE_COMMON_PERIOD and bucket_series:
        common_start = max(s.index.min() for s in bucket_series.values())
        common_end   = min(s.index.max() for s in bucket_series.values())
        if common_start < common_end:
            common_window = (pd.Timestamp(common_start), pd.Timestamp(common_end))
            # Exact same trading dates for every strategy (global intersection).
            idx_sets = [
                s.loc[common_window[0]:common_window[1]].index
                for s in bucket_series.values()
            ]
            if idx_sets:
                common_idx = idx_sets[0]
                for idx in idx_sets[1:]:
                    common_idx = common_idx.intersection(idx)
                common_dates_all = common_idx.sort_values()
            logger.info(
                f"Common testing window enabled: "
                f"{common_window[0].strftime('%Y-%m-%d')} -> {common_window[1].strftime('%Y-%m-%d')}"
            )
        else:
            logger.error(
                "Common testing window enabled, but there is no overlap across available buckets."
            )

    all_equity_curves: dict[str, pd.Series] = {}
    all_metrics:       dict[str, dict]      = {}

    logger.info("\n── PHASE 2: BACKTEST SIMULATION ────────────────────────────")
    for strat_id, strat_cfg in STRATEGIES.items():
        logger.info(f"\n  Strategy: {strat_cfg['name']} ({strat_id})")
        weights    = strat_cfg["weights"]
        start, end = strat_cfg["start_date"], strat_cfg["end_date"]

        if common_window is not None:
            start_ts = max(pd.Timestamp(start), common_window[0])
            end_ts   = min(pd.Timestamp(end), common_window[1])
            if start_ts >= end_ts:
                logger.error(
                    f"  Skipping {strat_id}: no data left after applying common window."
                )
                continue
            start, end = start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

        # Check all required buckets are available
        missing = [b for b in weights if b not in bucket_series]
        if missing:
            logger.error(
                f"  Skipping {strat_id}: missing buckets {missing}\n"
                + "\n".join(f"    {suggest_search_text(b)}" for b in missing)
            )
            continue

        # Build aligned price DataFrame over this strategy's window
        raw_slices = {
            bucket: bucket_series[bucket].loc[start:end]
            for bucket in weights
        }
        prices_df = align_series(raw_slices)
        if common_dates_all is not None:
            prices_df = prices_df.reindex(common_dates_all).dropna()

        # ── Minimum data guard ───────────────────────────────────────────────
        if prices_df.empty or len(prices_df) < 30:
            logger.error(
                f"  Skipping {strat_id}: only {len(prices_df)} aligned trading days."
            )
            continue

        # Strategy 3 specific: require at least 1 year of aligned data
        if strat_id == "strategy_3" and len(prices_df) < MIN_STRATEGY3_TRADING_DAYS:
            logger.error(
                f"  Skipping strategy_3: only {len(prices_df)} aligned trading days "
                f"(minimum required: {MIN_STRATEGY3_TRADING_DAYS} ≈ 1 year).\n"
                f"  The AI ETF, Bitcoin ETP or Hang Seng TECH instrument may have "
                f"insufficient history.\n"
                f"  → Check config.py BUCKETS['ai_etf'] / 'bitcoin_eur' / 'hang_seng_tech' "
                f"for earlier alternatives."
            )
            print(
                f"\nWARNING: Strategy 3 aborted: only {len(prices_df)} trading days of common data "
                f"(need >= {MIN_STRATEGY3_TRADING_DAYS}). "
                f"Strategies 1 and 2 continue normally.\n"
            )
            continue

        actual_start = prices_df.index[0].strftime("%Y-%m-%d")
        actual_end   = prices_df.index[-1].strftime("%Y-%m-%d")
        logger.info(
            f"  Period (actual): {actual_start} → {actual_end} "
            f"({len(prices_df)} trading days)"
        )

        # Save processed data
        prices_df.to_csv(
            os.path.join(OUTPUT_TABLES, f"prices_{strat_id}.csv"), index_label="date"
        )
        compute_returns(prices_df).to_csv(
            os.path.join(OUTPUT_TABLES, f"returns_{strat_id}.csv"), index_label="date"
        )

        # Simulate
        try:
            equity = simulate_portfolio(prices_df, weights, INITIAL_INVESTMENT, REBAL_FREQ)
        except Exception as exc:
            logger.error(f"  Simulation error for {strat_id}: {exc}")
            continue

        all_equity_curves[strat_id] = equity

        # Metrics
        instruments_str = ", ".join(
            f"{b}={bucket_ric_map.get(b, '?')}" for b in weights
        )
        all_metrics[strat_id] = calculate_all_metrics(
            equity,
            strategy_name=strat_cfg["name"],
            instruments_used=instruments_str,
            rf_annual=RISK_FREE_RATE,
        )

    # 6. Save combined outputs ─────────────────────────────────────────────────
    if all_equity_curves:
        equity_df = pd.DataFrame(all_equity_curves)
        equity_df.index.name = "date"
        equity_df.to_csv(os.path.join(OUTPUT_TABLES, "equity_curves.csv"))
        logger.info(f"Saved: {OUTPUT_TABLES}/equity_curves.csv")

    if all_metrics:
        # Full metrics (all fields)
        full_df = pd.DataFrame(all_metrics).T
        full_df.index.name = "strategy_id"
        full_df.to_csv(os.path.join(OUTPUT_TABLES, "summary_metrics.csv"))
        logger.info(f"Saved: {OUTPUT_TABLES}/summary_metrics.csv")

        # Clean comparison CSV (publication-ready subset)
        _save_comparison_clean(all_metrics)
        logger.info(f"Saved: {OUTPUT_TABLES}/strategy_comparison_clean.csv")

    # 7. Plot ──────────────────────────────────────────────────────────────────
    if all_equity_curves:
        _plot_equity_curves(all_equity_curves, all_metrics, STRATEGIES)

    # 8. Close session ─────────────────────────────────────────────────────────
    close_connection()

    # 9. Terminal summary + instructions ───────────────────────────────────────
    _print_summary(all_metrics)
    _print_instructions()

    logger.info("\nBacktest complete. Results in output/")


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_equity_curves(
    equity_curves: dict[str, pd.Series],
    all_metrics:   dict[str, dict],
    strategies:    dict,
) -> None:
    """Corbes d'equitat base-100 amb CAGR a la llegenda i periode real al subtitol."""
    logger = logging.getLogger(__name__)

    fig, ax = plt.subplots(figsize=(15, 7))

    colors = ["#1565C0", "#2E7D32", "#BF360C"]
    styles = ["-", "--", "-."]

    period_parts = []

    for idx, (strat_id, equity) in enumerate(equity_curves.items()):
        base100 = equity / equity.iloc[0] * 100.0
        name    = strategies[strat_id]["name"]
        cagr    = all_metrics[strat_id].get("cagr_pct", float("nan"))
        label   = f"{name}  (CAGR: {cagr:+.2f}%)"
        color   = colors[idx % len(colors)]
        style   = styles[idx % len(styles)]

        ax.plot(base100.index, base100.values,
                label=label, color=color, linewidth=2, linestyle=style)

        s = equity.index[0].strftime("%Y-%m-%d")
        e = equity.index[-1].strftime("%Y-%m-%d")
        period_parts.append(f"{name.split()[0]}: {s}→{e}")

    # Linia de referencia a 100
    ax.axhline(100.0, color="grey", linewidth=0.7, linestyle=":", alpha=0.7, label="Base 100")

    # Titol + subtitol
    subtitle  = "  |  ".join(period_parts)
    ax.set_title(
        f"Corbes d'Equitat de Cartera (Base 100, inicial €{INITIAL_INVESTMENT:,.0f})\n{subtitle}",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Valor Indexat (Base 100)", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.18, linestyle="--")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_CHARTS, "equity_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {out_path}")


# ── Clean comparison CSV ───────────────────────────────────────────────────────

def _save_comparison_clean(all_metrics: dict) -> None:
    """Save publication-ready subset of metrics."""
    rows = []
    for strat_id, m in all_metrics.items():
        rows.append({
            "strategy":           m["strategy_name"],
            "start_used":         m["start_date"],
            "end_used":           m["end_date"],
            "final_value_eur":    m["final_eur"],
            "total_return_pct":   m["total_return_pct"],
            "cagr_pct":           m["cagr_pct"],
            "vol_ann_pct":        m["volatility_ann_pct"],
            "max_drawdown_pct":   m["max_drawdown_pct"],
            "sharpe_ratio":       m["sharpe_ratio"],
            "sortino_ratio":      m.get("sortino_ratio",        float("nan")),
            "worst_year_pct":     m.get("worst_year_pct",       float("nan")),
            "pct_days_in_dd":     m.get("pct_days_in_drawdown", float("nan")),
        })

    df = pd.DataFrame(rows).set_index("strategy")
    df.to_csv(os.path.join(OUTPUT_TABLES, "strategy_comparison_clean.csv"))


# ── Terminal summary ───────────────────────────────────────────────────────────

def _print_summary(all_metrics: dict) -> None:
    """Print results in the requested exact format."""
    SEP = "=" * 30
    print(f"\n{SEP}")
    print("BACKTEST SUMMARY")
    print(SEP)

    strat_labels = {
        "strategy_1": "Estratègia 1",
        "strategy_2": "Estratègia 2",
        "strategy_3": "Estratègia 3",
    }

    for strat_id, m in all_metrics.items():
        label = strat_labels.get(strat_id, strat_id.title())
        s, e  = m["start_date"], m["end_date"]

        sortino  = m.get("sortino_ratio",        float("nan"))
        worst    = m.get("worst_year_pct",       float("nan"))
        pct_dd   = m.get("pct_days_in_drawdown", float("nan"))

        sortino_str = f"{sortino:.2f}" if np.isfinite(sortino) else "n/a"
        worst_str   = f"{worst:.2f}%"  if np.isfinite(worst)  else "n/a"
        pct_dd_str  = f"{pct_dd:.1f}%" if np.isfinite(pct_dd) else "n/a"

        print(f"\n{label} ({s} -> {e})")
        print(f"Valor final: {m['final_eur']:,.2f} EUR")
        print(f"Total return: {m['total_return_pct']:.2f}%")
        print(f"CAGR: {m['cagr_pct']:.2f}%")
        print(f"Vol: {m['volatility_ann_pct']:.2f}%")
        print(f"MaxDD: {m['max_drawdown_pct']:.2f}%")
        print(f"Sharpe (rf=0): {m['sharpe_ratio']:.2f}")
        print(f"Sortino (rf=0): {sortino_str}")
        print(f"Worst year: {worst_str}")
        print(f"% days in drawdown: {pct_dd_str}")

    print(f"\n{SEP}\n")


# ── Instructions & troubleshooting ────────────────────────────────────────────

def _print_instructions() -> None:
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("INSTRUCCIONS D'EXECUCIÓ")
    print(SEP)
    print("  pip install -r requirements.txt")
    print("  export APP_KEY=<refinitiv_app_key>   # opcional si Workspace obert")
    print("  python main.py                        # usa cache si existeix")
    print("  python main.py --refresh              # força re-descàrrega")
    print()
    print("TROUBLESHOOTING:")
    print("  · Si un bucket falla -> edita BUCKETS a config.py")
    print("    o cerca: 'UCITS ETF [nom] EUR accumulating' a Refinitiv.")
    print("  · Si FX falla -> comprova CCY_USD_RICS i EUR_USD_RIC a config.py")
    print("    o comprova que Refinitiv Workspace estigui obert.")
    print("  · Si Estratègia 3 falla per periode curt -> el Bitcoin ETP o")
    print("    Hang Seng TECH pot tenir < 1 any. Busca alternatives al config.")
    print("  · Per forçar re-descàrrega: python main.py --refresh")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
