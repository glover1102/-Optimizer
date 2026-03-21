"""
Walk-forward validation engine.

Splits historical data into N windows, optimises on the in-sample portion,
evaluates on the out-of-sample portion, and reports stability metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.backtester import run_backtest

logger = logging.getLogger(__name__)

_OVERFIT_THRESHOLD = 0.15  # IS win-rate exceeds OOS by this fraction → flag


def run_walk_forward(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    best_params: dict[str, Any],
    n_windows: int = 5,
    oos_fraction: float = 0.2,
) -> dict[str, Any]:
    """
    Run walk-forward validation using *best_params*.

    Each window has an in-sample (IS) portion for reference and an
    out-of-sample (OOS) portion for evaluation.  We always evaluate the
    same *best_params* (derived from full-sample optimisation) so the
    walk-forward tests generalisation rather than per-window re-optimisation.

    Parameters
    ----------
    high, low, close : full price arrays
    best_params      : parameter dict to evaluate in each OOS window
    n_windows        : number of walk-forward windows
    oos_fraction     : fraction of each window reserved for OOS evaluation

    Returns
    -------
    {
        "windows": list[dict],           # per-window results
        "avg_oos_win_rate": float,
        "avg_is_win_rate": float,
        "stability_score": float,        # 0–1 (higher = more stable)
        "consistency": float,            # 1 – (std / mean) of OOS win-rates
        "overfitting_detected": bool,
        "walk_forward_score": float,     # composite score 0–100
    }
    """
    n = len(close)
    window_size = n // n_windows

    if window_size < 100:
        logger.warning("Insufficient data for %d walk-forward windows", n_windows)
        return _empty_wf()

    oos_wins: list[float] = []
    is_wins: list[float] = []
    windows: list[dict] = []

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else n
        split = int(start + (end - start) * (1 - oos_fraction))

        is_h = high[start:split]
        is_l = low[start:split]
        is_c = close[start:split]

        oos_h = high[split:end]
        oos_l = low[split:end]
        oos_c = close[split:end]

        is_result = run_backtest(is_h, is_l, is_c, **best_params)
        oos_result = run_backtest(oos_h, oos_l, oos_c, **best_params)

        is_wr = is_result["win_rate"] if is_result["total_signals"] >= 5 else None
        oos_wr = oos_result["win_rate"] if oos_result["total_signals"] >= 3 else None

        windows.append({
            "window": w + 1,
            "is_bars": split - start,
            "oos_bars": end - split,
            "is_signals": is_result["total_signals"],
            "oos_signals": oos_result["total_signals"],
            "is_win_rate": is_wr,
            "oos_win_rate": oos_wr,
        })

        if is_wr is not None:
            is_wins.append(is_wr)
        if oos_wr is not None:
            oos_wins.append(oos_wr)

    if not oos_wins:
        return _empty_wf()

    avg_oos = float(np.mean(oos_wins))
    avg_is = float(np.mean(is_wins)) if is_wins else avg_oos

    # Stability: 1 − coefficient of variation of OOS win-rates
    if len(oos_wins) > 1:
        std = float(np.std(oos_wins))
        cv = std / max(avg_oos, 0.01)
        consistency = max(0.0, 1.0 - cv)
    else:
        consistency = 0.5

    # Overfitting flag
    overfitting = (avg_is - avg_oos) > _OVERFIT_THRESHOLD

    # Walk-forward score (0–100)
    wf_score = avg_oos * 100 * consistency
    if overfitting:
        wf_score *= 0.7  # penalty

    stability_score = consistency

    return {
        "windows": windows,
        "avg_oos_win_rate": round(avg_oos, 4),
        "avg_is_win_rate": round(avg_is, 4),
        "stability_score": round(stability_score, 4),
        "consistency": round(consistency, 4),
        "overfitting_detected": overfitting,
        "walk_forward_score": round(wf_score, 2),
    }


def _empty_wf() -> dict[str, Any]:
    return {
        "windows": [],
        "avg_oos_win_rate": 0.0,
        "avg_is_win_rate": 0.0,
        "stability_score": 0.0,
        "consistency": 0.0,
        "overfitting_detected": False,
        "walk_forward_score": 0.0,
    }
