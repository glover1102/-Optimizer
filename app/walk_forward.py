"""Walk-forward validation for the KLS+MoM strategy."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.backtester import run_backtest

logger = logging.getLogger(__name__)
_OVERFIT_THRESHOLD = 0.15


def run_walk_forward(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None,
    best_params: dict[str, Any],
    *,
    timestamps=None,
    timeframe: str | None = None,
    open_: np.ndarray | None = None,
    n_windows: int = 5,
    oos_fraction: float = 0.2,
) -> dict[str, Any]:
    """Evaluate fixed KLS+MoM parameters across rolling OOS windows."""
    n = len(close)
    window_size = n // n_windows
    if window_size < 40:
        logger.warning("Insufficient data for %d walk-forward windows", n_windows)
        return _empty_wf()

    oos_wins: list[float] = []
    is_wins: list[float] = []
    windows: list[dict] = []
    open_arr = open_ if open_ is not None else np.concatenate(([close[0]], close[:-1]))
    volume_arr = np.ones_like(close) if volume is None else volume

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else n
        split = int(start + (end - start) * (1 - oos_fraction))
        slices = {
            "is": slice(start, split),
            "oos": slice(split, end),
        }
        results: dict[str, dict[str, Any]] = {}
        for label, slc in slices.items():
            ts_slice = None if timestamps is None else timestamps[slc]
            results[label] = run_backtest(
                high[slc],
                low[slc],
                close[slc],
                volume_arr[slc],
                open_=open_arr[slc],
                timestamps=ts_slice,
                timeframe=timeframe,
                **best_params,
            )
        is_wr = results["is"]["win_rate"] if results["is"]["total_signals"] >= 2 else None
        oos_wr = results["oos"]["win_rate"] if results["oos"]["total_signals"] >= 1 else None
        windows.append({
            "window": w + 1,
            "is_bars": split - start,
            "oos_bars": end - split,
            "is_signals": results["is"]["total_signals"],
            "oos_signals": results["oos"]["total_signals"],
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
    if len(oos_wins) > 1:
        std = float(np.std(oos_wins))
        cv = std / max(avg_oos, 0.01)
        consistency = max(0.0, 1.0 - cv)
    else:
        consistency = 0.5
    overfitting = (avg_is - avg_oos) > _OVERFIT_THRESHOLD
    wf_score = avg_oos * 100 * consistency
    if overfitting:
        wf_score *= 0.7
    return {
        "windows": windows,
        "avg_oos_win_rate": round(avg_oos, 4),
        "avg_is_win_rate": round(avg_is, 4),
        "stability_score": round(consistency, 4),
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
