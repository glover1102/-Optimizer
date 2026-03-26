"""
Vectorized QTAlgo backtester.

Replicates the exact QTAlgo Pine Script pivot-detection logic using NumPy
arrays, with a minimal number of Python loops.

Pivot Detection Logic (matching Pine Script):
 1. Calculate ATR for the given period.
 2. Pivot highs: bars where high + ATR*offset is the maximum in
    left_bars before AND right_bars after.
 3. Pivot lows:  bars where low  - ATR*offset is the minimum in
    left_bars before AND right_bars after.
 4. Track running pivotHigh / pivotLow (last confirmed pivot).
 5. Trend: close > pivotHigh → 1 (bull), close < pivotLow → -1 (bear).
 6. Signal: when trend changes by 2 (buy: -1→1, sell: 1→-1).
 7. Entry = close at signal bar.
 8. SL = pivotLow (buys) or pivotHigh (sells).
 9. TP1 = entry ± atr*5,  TP2 = entry ± atr*10,  TP3 = entry ± atr*15.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Compute ATR using Wilder's smoothing (matches Pine Script ta.atr)."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )

    atr = np.empty(n)
    atr[:period] = np.nan
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum over *window* bars (inclusive of current bar)."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_slice = arr[i - window + 1 : i + 1]
        valid = window_slice[~np.isnan(window_slice)]
        if len(valid) > 0:
            out[i] = valid.max()
    return out


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum over *window* bars (inclusive of current bar)."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_slice = arr[i - window + 1 : i + 1]
        valid = window_slice[~np.isnan(window_slice)]
        if len(valid) > 0:
            out[i] = valid.min()
    return out


def run_backtest(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    left_bars: int = 10,
    right_bars: int = 10,
    offset: float = 2.0,
    atr_multiplier: float = 1.0,
    atr_period: int = 14,
) -> dict[str, Any]:
    """
    Run a single backtest with the given parameters.

    Parameters
    ----------
    high, low, close : 1-D float arrays of equal length
    left_bars        : lookback bars for pivot confirmation
    right_bars       : look-forward bars for pivot confirmation
    offset           : ATR multiplier applied to the pivot level
    atr_multiplier   : ATR multiplier for TP calculation
    atr_period       : period for the ATR calculation

    Returns
    -------
    dict with keys: total_signals, wins, tp2_hits, tp3_hits, sl_hits,
                    win_rate, tp2_rate, tp3_rate, sl_rate
    """
    n = len(close)

    if n < max(left_bars + right_bars + atr_period, 50):
        return _empty_result()

    atr = _atr(high, low, close, atr_period)

    # ── Pivot level arrays ────────────────────────────────────────────────
    # Adjusted high / low with ATR offset (matches Pine Script pivot logic)
    adj_high = high + atr * offset
    adj_low = low - atr * offset

    # Vectorised: for each bar i, is it a pivot high?
    # It must be the max of adj_high in [i-left_bars .. i] AND
    # the max of adj_high in [i .. i+right_bars].
    # We check both windows and compare centre value.

    # Rolling max looking left (including centre)
    left_max_h = _rolling_max(adj_high, left_bars + 1)
    # Rolling max looking right (we shift the array for look-forward)
    right_max_h = np.full(n, np.nan)
    for i in range(n):
        right_end = min(i + right_bars + 1, n)
        window_slice = adj_high[i:right_end]
        valid = window_slice[~np.isnan(window_slice)]
        if len(valid) > 0:
            right_max_h[i] = valid.max()

    is_pivot_high = (adj_high == left_max_h) & (adj_high == right_max_h)

    left_min_l = _rolling_min(adj_low, left_bars + 1)
    right_min_l = np.full(n, np.nan)
    for i in range(n):
        right_end = min(i + right_bars + 1, n)
        window_slice = adj_low[i:right_end]
        valid = window_slice[~np.isnan(window_slice)]
        if len(valid) > 0:
            right_min_l[i] = valid.min()

    is_pivot_low = (adj_low == left_min_l) & (adj_low == right_min_l)

    # ── Trend + signal state machine (requires sequential state) ─────────
    trend = np.zeros(n, dtype=np.int8)
    pivot_high_val = np.full(n, np.nan)
    pivot_low_val = np.full(n, np.nan)

    running_ph = np.nan
    running_pl = np.nan

    for i in range(n):
        if is_pivot_high[i]:
            running_ph = high[i]
        if is_pivot_low[i]:
            running_pl = low[i]
        pivot_high_val[i] = running_ph
        pivot_low_val[i] = running_pl

        if not np.isnan(running_ph) and close[i] > running_ph:
            trend[i] = 1
        elif not np.isnan(running_pl) and close[i] < running_pl:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1] if i > 0 else 0

    # ── Detect signal bars (trend change of ±2) ──────────────────────────
    trend_diff = np.diff(trend.astype(np.int16), prepend=trend[0])
    buy_signals = np.where(trend_diff == 2)[0]
    sell_signals = np.where(trend_diff == -2)[0]

    all_signals = np.concatenate([buy_signals, sell_signals])
    directions = np.concatenate([
        np.ones(len(buy_signals), dtype=np.int8),
        -np.ones(len(sell_signals), dtype=np.int8),
    ])

    if len(all_signals) == 0:
        return _empty_result()

    # Sort by bar index
    order = np.argsort(all_signals)
    all_signals = all_signals[order]
    directions = directions[order]

    # ── Simulate trades ───────────────────────────────────────────────────
    wins = tp2_hits = tp3_hits = sl_hits = 0
    total_signals = len(all_signals)
    trades: list[dict] = []

    for idx, (sig_bar, direction) in enumerate(zip(all_signals, directions)):
        entry = close[sig_bar]
        atr_val = atr[sig_bar]
        if np.isnan(atr_val) or atr_val <= 0:
            total_signals -= 1
            continue

        if direction == 1:  # buy
            sl = pivot_low_val[sig_bar]
            if np.isnan(sl):
                total_signals -= 1
                continue
        else:  # sell
            sl = pivot_high_val[sig_bar]
            if np.isnan(sl):
                total_signals -= 1
                continue

        tp1 = entry + direction * atr_val * 5
        tp2 = entry + direction * atr_val * 10
        tp3 = entry + direction * atr_val * 15

        # Next signal bar (or end of data)
        next_sig = all_signals[idx + 1] if idx + 1 < total_signals else n
        trade_bars = close[sig_bar + 1 : next_sig]

        if len(trade_bars) == 0:
            total_signals -= 1
            continue

        tp3_hit = tp2_hit = tp1_hit = sl_hit = False

        for price in trade_bars:
            if direction == 1:
                if price >= tp3:
                    tp3_hit = True
                    tp2_hit = True
                    tp1_hit = True
                    break
                if price >= tp2:
                    tp2_hit = True
                    tp1_hit = True
                    break
                if price >= tp1:
                    tp1_hit = True
                    break
                if price <= sl:
                    sl_hit = True
                    break
            else:
                if price <= tp3:
                    tp3_hit = True
                    tp2_hit = True
                    tp1_hit = True
                    break
                if price <= tp2:
                    tp2_hit = True
                    tp1_hit = True
                    break
                if price <= tp1:
                    tp1_hit = True
                    break
                if price >= sl:
                    sl_hit = True
                    break

        if tp1_hit or tp2_hit or tp3_hit:
            wins += 1
        if tp2_hit or tp3_hit:
            tp2_hits += 1
        if tp3_hit:
            tp3_hits += 1
        if sl_hit:
            sl_hits += 1

        result_str = (
            "tp3_hit" if tp3_hit else
            "tp2_hit" if tp2_hit else
            "tp1_hit" if tp1_hit else
            "sl_hit" if sl_hit else
            "open"
        )
        trades.append({
            "bar": int(sig_bar),
            "direction": "BUY" if direction == 1 else "SELL",
            "entry": float(entry),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "result": result_str,
        })

    if total_signals <= 0:
        return _empty_result()

    win_rate = wins / total_signals
    tp2_rate = tp2_hits / total_signals
    tp3_rate = tp3_hits / total_signals
    sl_rate = sl_hits / total_signals

    return {
        "total_signals": total_signals,
        "wins": wins,
        "tp2_hits": tp2_hits,
        "tp3_hits": tp3_hits,
        "sl_hits": sl_hits,
        "win_rate": round(win_rate, 4),
        "tp2_rate": round(tp2_rate, 4),
        "tp3_rate": round(tp3_rate, 4),
        "sl_rate": round(sl_rate, 4),
        "trades": trades,
    }


def _empty_result() -> dict[str, Any]:
    return {
        "total_signals": 0,
        "wins": 0,
        "tp2_hits": 0,
        "tp3_hits": 0,
        "sl_hits": 0,
        "win_rate": 0.0,
        "tp2_rate": 0.0,
        "tp3_rate": 0.0,
        "sl_rate": 0.0,
        "trades": [],
    }
