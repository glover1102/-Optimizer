"""KLS+MoM strategy backtester."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.kls_mom import atr as _atr, empty_backtest_result, simulate_kls_strategy, summarize_backtest


def run_backtest(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None = None,
    *,
    open_: np.ndarray | None = None,
    timestamps=None,
    timeframe: str | None = None,
    **params,
) -> dict[str, Any]:
    """Run a KLS+MoM backtest while preserving the legacy result contract."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if len(close) < 5:
        return _empty_result()

    simulation = simulate_kls_strategy(
        high,
        low,
        close,
        volume,
        open_=open_,
        timestamps=timestamps,
        timeframe=timeframe,
        params=params,
    )
    return summarize_backtest(simulation)


def _empty_result() -> dict[str, Any]:
    return empty_backtest_result()
