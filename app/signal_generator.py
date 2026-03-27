"""
Signal generator — ports the QTAlgo Buy&Sell PineScript logic to Python.

Implements:
  - Pivot trend detection (matches backtester.py logic)
  - Regime filter (HMA-based trend/volume scoring)
  - RSI filter
  - WaveTrend filter
  - EMA trend filter
  - Golden Line (composite MVWAP/CVWAP/EMA)
  - Price position filter
  - Entry modes: Pivot, Crossover, Hybrid
  - ATR / R:R target calculation
  - Signal strength (0-4)

Public API:
  generate_signal(symbol, timeframe, params=None) -> dict
  generate_signals_batch(symbols, timeframe, params=None) -> list[dict]
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level indicator helpers
# ─────────────────────────────────────────────────────────────────────────────


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR via Wilder smoothing — matches backtester.py."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    k = 2.0 / (period + 1.0)
    # Find first non-nan index
    start = 0
    while start < len(arr) and np.isnan(arr[start]):
        start += 1
    if start + period > len(arr):
        return out
    out[start + period - 1] = np.nanmean(arr[start : start + period])
    for i in range(start + period, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1 : i + 1]
        if not np.any(np.isnan(window)):
            out[i] = window.mean()
    return out


def _wma(arr: np.ndarray, period: int) -> np.ndarray:
    """Weighted moving average (linearly weighted)."""
    n = len(arr)
    out = np.full(n, np.nan)
    weights = np.arange(1, period + 1, dtype=np.float64)
    w_sum = weights.sum()
    for i in range(period - 1, n):
        window = arr[i - period + 1 : i + 1]
        if not np.any(np.isnan(window)):
            out[i] = (window * weights).sum() / w_sum
    return out


def _hma(arr: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(1, period // 2)
    sqrt_p = max(1, int(round(math.sqrt(period))))
    wma_half = _wma(arr, half)
    wma_full = _wma(arr, period)
    diff = 2.0 * wma_half - wma_full
    return _wma(diff, sqrt_p)


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Standard RSI."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[period] = gain[1 : period + 1].mean()
    avg_loss[period] = loss[1 : period + 1].mean()
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    valid = ~np.isnan(avg_gain)
    out[valid] = 100.0 - 100.0 / (1.0 + rs[valid])
    return out


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        s = arr[i - window + 1 : i + 1]
        v = s[~np.isnan(s)]
        if len(v) > 0:
            out[i] = v.max()
    return out


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        s = arr[i - window + 1 : i + 1]
        v = s[~np.isnan(s)]
        if len(v) > 0:
            out[i] = v.min()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pivot trend detection (mirrors backtester.py)
# ─────────────────────────────────────────────────────────────────────────────


def _compute_pivot_trend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    left_bars: int,
    right_bars: int,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (trend, pivot_high_val, pivot_low_val, is_signal_bar) arrays.
    trend: 1=bull, -1=bear, 0=neutral
    """
    n = len(close)
    adj_high = high + atr * offset
    adj_low = low - atr * offset

    left_max_h = _rolling_max(adj_high, left_bars + 1)
    right_max_h = np.full(n, np.nan)
    for i in range(n):
        s = adj_high[i : min(i + right_bars + 1, n)]
        v = s[~np.isnan(s)]
        if len(v) > 0:
            right_max_h[i] = v.max()
    is_pivot_high = (adj_high == left_max_h) & (adj_high == right_max_h)

    left_min_l = _rolling_min(adj_low, left_bars + 1)
    right_min_l = np.full(n, np.nan)
    for i in range(n):
        s = adj_low[i : min(i + right_bars + 1, n)]
        v = s[~np.isnan(s)]
        if len(v) > 0:
            right_min_l[i] = v.min()
    is_pivot_low = (adj_low == left_min_l) & (adj_low == right_min_l)

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

    trend_diff = np.diff(trend.astype(np.int16), prepend=trend[0])
    is_signal_bar = (trend_diff == 2) | (trend_diff == -2)

    return trend, pivot_high_val, pivot_low_val, is_signal_bar


# ─────────────────────────────────────────────────────────────────────────────
# Regime filter (HMA-based)
# ─────────────────────────────────────────────────────────────────────────────


def _compute_regime_filter(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    regime_length: int,
    trend_threshold: float,
    volume_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (trend_score, volume_score) arrays.
    Scores > threshold indicate positive regime alignment.
    """
    hlc3 = (high + low + close) / 3.0

    hma_hlc3 = _hma(hlc3, 15)
    hma_vol = _hma(volume, 15)

    n = len(close)
    trend_score = np.zeros(n)
    volume_score = np.zeros(n)

    for i in range(regime_length, n):
        window_hlc3 = hma_hlc3[i - regime_length : i]
        window_vol = hma_vol[i - regime_length : i]
        current_hlc3 = hma_hlc3[i]
        current_vol = hma_vol[i]

        if np.isnan(current_hlc3) or np.isnan(current_vol):
            continue

        ts = np.sum(~np.isnan(window_hlc3) & (window_hlc3 < current_hlc3))
        vs = np.sum(~np.isnan(window_vol) & (window_vol < current_vol))
        trend_score[i] = float(ts)
        volume_score[i] = float(vs)

    return trend_score, volume_score


# ─────────────────────────────────────────────────────────────────────────────
# WaveTrend oscillator
# ─────────────────────────────────────────────────────────────────────────────


def _compute_wavetrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    channel_len: int = 10,
    avg_len: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (wt1, wt2) WaveTrend oscillator arrays."""
    hlc3 = (high + low + close) / 3.0
    esa = _ema(hlc3, channel_len)
    d = _ema(np.abs(hlc3 - esa), channel_len)
    ci = np.where(d == 0, 0.0, (hlc3 - esa) / (0.015 * d))
    wt1 = _ema(ci, avg_len)
    wt2 = _sma(wt1, 4)
    return wt1, wt2


# ─────────────────────────────────────────────────────────────────────────────
# VWAP helpers (rolling approximation)
# ─────────────────────────────────────────────────────────────────────────────


def _rolling_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Rolling VWAP over *period* bars as MVWAP approximation."""
    hlc3 = (high + low + close) / 3.0
    tp_vol = hlc3 * volume
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        tv = tp_vol[i - period + 1 : i + 1]
        vv = volume[i - period + 1 : i + 1]
        sv = vv.sum()
        if sv > 0:
            out[i] = tv.sum() / sv
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Golden Line
# ─────────────────────────────────────────────────────────────────────────────


def _compute_golden_line(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 1.0,
    w4: float = 1.0,
    ema1_period: int = 9,
    ema2_period: int = 50,
) -> np.ndarray:
    """
    Golden Line composite:
    GL = (MVWAP*w1 + CVWAP*w2 + EMA1*w3 + EMA2*w4) / total_weight
    MVWAP = rolling VWAP 21, CVWAP = rolling VWAP 9
    """
    mvwap = _rolling_vwap(high, low, close, volume, 21)
    cvwap = _rolling_vwap(high, low, close, volume, 9)
    ema1 = _ema(close, ema1_period)
    ema2 = _ema(close, ema2_period)
    total_weight = w1 + w2 + w3 + w4
    if total_weight == 0:
        total_weight = 4.0
    gl = (mvwap * w1 + cvwap * w2 + ema1 * w3 + ema2 * w4) / total_weight
    return gl


# ─────────────────────────────────────────────────────────────────────────────
# Signal strength calculation
# ─────────────────────────────────────────────────────────────────────────────


def _signal_strength(
    trend_score: float,
    volume_score: float,
    trend_threshold: float,
    volume_threshold: float,
) -> int:
    """Returns signal strength 0-4."""
    trend_ok = trend_score >= trend_threshold
    volume_ok = volume_score >= volume_threshold
    if trend_ok and volume_ok:
        return 4
    if trend_ok:
        return 3
    if volume_ok:
        return 2
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Target price calculation
# ─────────────────────────────────────────────────────────────────────────────


def _calculate_targets(
    direction: int,
    entry: float,
    sl: float,
    atr_val: float,
    params: dict,
) -> tuple[float, float, float]:
    """Returns (tp1, tp2, tp3)."""
    use_rr = params.get("use_rr_targets", False)
    if use_rr:
        sl_dist = abs(entry - sl)
        rr1 = params.get("rr_tp1", 1.0)
        rr2 = params.get("rr_tp2", 2.0)
        rr3 = params.get("rr_tp3", 3.0)
        tp1 = entry + direction * sl_dist * rr1
        tp2 = entry + direction * sl_dist * rr2
        tp3 = entry + direction * sl_dist * rr3
    else:
        target = params.get("atr_target", 0.0)
        tp1 = entry + direction * atr_val * (5 + target)
        tp2 = entry + direction * atr_val * (10 + target * 2)
        tp3 = entry + direction * atr_val * (15 + target * 3)
    return tp1, tp2, tp3


# ─────────────────────────────────────────────────────────────────────────────
# Core signal generation
# ─────────────────────────────────────────────────────────────────────────────


def _generate_signal_from_arrays(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    params: dict,
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate signal from OHLCV numpy arrays.
    Returns a signal dict with action, strength, prices, etc.
    """
    n = len(close)
    p = params  # shorthand

    left_bars = int(p.get("left_bars", 10))
    right_bars = int(p.get("right_bars", 10))
    offset = float(p.get("offset", 2.0))
    atr_period = int(p.get("atr_period", 14))
    regime_length = int(p.get("regime_length", 20))
    trend_threshold = float(p.get("trend_threshold", 3.0))
    volume_threshold = float(p.get("volume_threshold", 2.0))
    entry_mode = str(p.get("entry_mode", "Pivot"))
    filters_active: list[str] = []

    min_bars = max(left_bars + right_bars + atr_period, 50)
    if n < min_bars:
        return _hold_signal(timestamp, "insufficient data")

    # ── ATR ──────────────────────────────────────────────────────────────
    atr = _atr(high, low, close, atr_period)

    # ── Pivot trend ───────────────────────────────────────────────────────
    trend, pivot_high_val, pivot_low_val, is_signal_bar = _compute_pivot_trend(
        high, low, close, atr, left_bars, right_bars, offset
    )

    # ── Regime filter ─────────────────────────────────────────────────────
    use_regime = bool(p.get("use_regime_filter", True))
    trend_score_arr = np.zeros(n)
    volume_score_arr = np.zeros(n)
    if use_regime:
        filters_active.append("regime")
        trend_score_arr, volume_score_arr = _compute_regime_filter(
            high, low, close, volume, regime_length, trend_threshold, volume_threshold
        )

    # ── RSI filter ────────────────────────────────────────────────────────
    use_rsi = bool(p.get("use_rsi_filter", False))
    rsi_arr = np.full(n, np.nan)
    if use_rsi:
        filters_active.append("rsi")
        rsi_period = int(p.get("rsi_length", 14))
        rsi_arr = _rsi(close, rsi_period)

    # ── WaveTrend filter ──────────────────────────────────────────────────
    use_wt = bool(p.get("use_wt_filter", False))
    wt1 = np.full(n, np.nan)
    wt2 = np.full(n, np.nan)
    if use_wt:
        filters_active.append("wavetrend")
        wt1, wt2 = _compute_wavetrend(
            high, low, close,
            channel_len=int(p.get("wt_channel_len", 10)),
            avg_len=int(p.get("wt_avg_len", 21)),
        )

    # ── EMA trend filter ──────────────────────────────────────────────────
    use_ema_trend = bool(p.get("use_ema_trend_filter", False))
    ema_trend = np.full(n, np.nan)
    if use_ema_trend:
        filters_active.append("ema_trend")
        ema_trend = _ema(close, int(p.get("ema_trend_period", 300)))

    # ── Golden Line ───────────────────────────────────────────────────────
    use_gl = bool(p.get("use_golden_line", False))
    gl = np.full(n, np.nan)
    if use_gl:
        filters_active.append("golden_line")
        gl = _compute_golden_line(
            high, low, close, volume,
            w1=float(p.get("gl_w1", 1.0)),
            w2=float(p.get("gl_w2", 1.0)),
            w3=float(p.get("gl_w3", 1.0)),
            w4=float(p.get("gl_w4", 1.0)),
            ema1_period=int(p.get("gl_ema1_period", 9)),
            ema2_period=int(p.get("gl_ema2_period", 50)),
        )
    # EMA used by Golden Line directional filter (ema_trend_period)
    ema_for_gl = ema_trend if use_ema_trend else _ema(close, int(p.get("ema_trend_period", 300)))

    # ── Price position filter ─────────────────────────────────────────────
    use_price_pos = bool(p.get("use_price_position_filter", False))
    if use_price_pos:
        filters_active.append("price_position")

    # ─────────────────────────────────────────────────────────────────────
    # Scan last N bars for the latest valid signal
    # We look at the last bar and work backwards to find a recent signal
    # ─────────────────────────────────────────────────────────────────────
    scan_window = min(n, max(left_bars + right_bars + 5, 30))
    start_scan = n - scan_window

    last_signal_bar: Optional[int] = None
    last_direction: int = 0
    last_gl_crossover_bar: Optional[int] = None
    last_gl_crossover_dir: int = 0

    # Track Golden Line crossover signals (for Crossover/Hybrid mode)
    gl_crossover_cooldown = 0
    gl_min_sep_atr = float(p.get("gl_min_separation_atr", 0.5))
    gl_cooldown_bars = int(p.get("gl_cooldown_bars", 3))
    gl_confirm_window = int(p.get("gl_confirm_window", 2))

    prev_gl_above = None  # True if GL > ema_for_gl at previous bar
    for i in range(max(1, start_scan - gl_cooldown_bars), n):
        if np.isnan(gl[i]) or np.isnan(ema_for_gl[i]):
            prev_gl_above = None
            continue
        curr_above = gl[i] > ema_for_gl[i]
        if prev_gl_above is not None and curr_above != prev_gl_above:
            # Crossover detected
            direction = 1 if curr_above else -1
            # Whipsaw protection: check ATR separation
            if not np.isnan(atr[i]) and atr[i] > 0:
                sep = abs(gl[i] - ema_for_gl[i])
                if sep < gl_min_sep_atr * atr[i]:
                    prev_gl_above = curr_above
                    continue
            last_gl_crossover_bar = i
            last_gl_crossover_dir = direction
        prev_gl_above = curr_above

    # Find last pivot signal bar in scan window
    for i in range(n - 1, start_scan - 1, -1):
        if is_signal_bar[i]:
            trend_diff = trend[i] - (trend[i - 1] if i > 0 else trend[i])
            direction = 1 if trend[i] == 1 else -1
            last_signal_bar = i
            last_direction = direction
            break

    # ─────────────────────────────────────────────────────────────────────
    # Determine which signal to use based on entry mode
    # ─────────────────────────────────────────────────────────────────────
    signal_bar: Optional[int] = None
    direction: int = 0
    is_confluence = False

    if entry_mode == "Pivot":
        signal_bar = last_signal_bar
        direction = last_direction
    elif entry_mode == "Crossover":
        if last_gl_crossover_bar is not None:
            signal_bar = last_gl_crossover_bar
            direction = last_gl_crossover_dir
    elif entry_mode == "Hybrid":
        # Both sources; confluence if both agree at same or nearby bar
        pivot_valid = last_signal_bar is not None
        gl_valid = last_gl_crossover_bar is not None
        if pivot_valid and gl_valid:
            pivot_bar: int = last_signal_bar  # type: ignore[assignment]
            gl_bar: int = last_gl_crossover_bar  # type: ignore[assignment]
            pivot_dir = last_direction
            gl_dir = last_gl_crossover_dir
            bars_apart = abs(pivot_bar - gl_bar)
            if pivot_dir == gl_dir and bars_apart <= gl_confirm_window:
                is_confluence = True
                signal_bar = max(pivot_bar, gl_bar)
                direction = pivot_dir
            else:
                # Use whichever is more recent
                if gl_bar >= pivot_bar:
                    signal_bar = gl_bar
                    direction = gl_dir
                else:
                    signal_bar = pivot_bar
                    direction = pivot_dir
        elif pivot_valid:
            signal_bar = last_signal_bar
            direction = last_direction
        elif gl_valid:
            signal_bar = last_gl_crossover_bar
            direction = last_gl_crossover_dir

    if signal_bar is None or direction == 0:
        return _hold_signal(timestamp, "no signal detected")

    i = signal_bar  # alias for readability
    atr_val = atr[i]
    if np.isnan(atr_val) or atr_val <= 0:
        return _hold_signal(timestamp, "invalid ATR at signal bar")

    # ─────────────────────────────────────────────────────────────────────
    # Apply filters at the signal bar
    # ─────────────────────────────────────────────────────────────────────

    # Regime filter
    require_trend = bool(p.get("require_trend_alignment", True))
    require_volume = bool(p.get("require_volume_confirmation", False))
    ts_val = trend_score_arr[i]
    vs_val = volume_score_arr[i]
    strength = 0

    if use_regime:
        trend_ok = ts_val >= trend_threshold
        volume_ok = vs_val >= volume_threshold
        if require_trend and not trend_ok:
            return _hold_signal(timestamp, "blocked by regime trend filter")
        if require_volume and not volume_ok:
            return _hold_signal(timestamp, "blocked by regime volume filter")
        strength = _signal_strength(ts_val, vs_val, trend_threshold, volume_threshold)
        # In Pivot mode require strength >= 2
        if entry_mode == "Pivot" and strength < 2:
            return _hold_signal(timestamp, f"strength {strength} < 2 required")
    else:
        strength = 2  # default when no regime filter

    # RSI filter
    if use_rsi:
        rsi_val = rsi_arr[i]
        ob = float(p.get("rsi_overbought", 70.0))
        os_ = float(p.get("rsi_oversold", 30.0))
        if not np.isnan(rsi_val):
            if direction == 1 and rsi_val >= ob:
                return _hold_signal(timestamp, "blocked by RSI overbought")
            if direction == -1 and rsi_val <= os_:
                return _hold_signal(timestamp, "blocked by RSI oversold")

    # WaveTrend filter
    if use_wt:
        wt1_val = wt1[i]
        wt_ob = float(p.get("wt_ob_level", 60))
        wt_os = float(p.get("wt_os_level", -60))
        wt_cross = bool(p.get("wt_require_cross", False))
        if not np.isnan(wt1_val):
            if direction == 1 and wt1_val >= wt_ob:
                return _hold_signal(timestamp, "blocked by WT overbought")
            if direction == -1 and wt1_val <= wt_os:
                return _hold_signal(timestamp, "blocked by WT oversold")
            if wt_cross and not np.isnan(wt2[i]):
                if direction == 1 and wt1_val <= wt2[i]:
                    return _hold_signal(timestamp, "blocked by WT cross confirmation")
                if direction == -1 and wt1_val >= wt2[i]:
                    return _hold_signal(timestamp, "blocked by WT cross confirmation")

    # EMA trend filter
    if use_ema_trend:
        ema_val = ema_trend[i]
        if not np.isnan(ema_val):
            if direction == 1 and close[i] < ema_val:
                return _hold_signal(timestamp, "blocked by EMA trend filter (bearish)")
            if direction == -1 and close[i] > ema_val:
                return _hold_signal(timestamp, "blocked by EMA trend filter (bullish)")

    # Golden Line directional filter (Pivot mode)
    if use_gl and entry_mode == "Pivot":
        gl_val = gl[i]
        ema_gl = ema_for_gl[i]
        if not np.isnan(gl_val) and not np.isnan(ema_gl):
            if direction == 1 and gl_val <= ema_gl:
                return _hold_signal(timestamp, "blocked by Golden Line (bearish)")
            if direction == -1 and gl_val >= ema_gl:
                return _hold_signal(timestamp, "blocked by Golden Line (bullish)")

    # Price position filter
    if use_price_pos and use_gl and use_ema_trend:
        price = close[i]
        gl_val = gl[i]
        ema_val = ema_for_gl[i]
        if not np.isnan(gl_val) and not np.isnan(ema_val):
            if direction == -1 and price > gl_val and price > ema_val:
                return _hold_signal(timestamp, "blocked by price position filter (above both lines)")
            if direction == 1 and price < gl_val and price < ema_val:
                return _hold_signal(timestamp, "blocked by price position filter (below both lines)")

    # ─────────────────────────────────────────────────────────────────────
    # Calculate entry, SL, targets
    # ─────────────────────────────────────────────────────────────────────
    entry_price = float(close[i])

    if direction == 1:  # BUY
        sl_price = pivot_low_val[i]
        if np.isnan(sl_price):
            sl_price = entry_price - atr_val * 2  # fallback
    else:  # SELL
        sl_price = pivot_high_val[i]
        if np.isnan(sl_price):
            sl_price = entry_price + atr_val * 2  # fallback

    tp1, tp2, tp3 = _calculate_targets(direction, entry_price, sl_price, atr_val, p)

    # Confidence: blend strength/4 with confluence bonus
    confidence = round((strength / 4.0) * 0.7 + (0.3 if is_confluence else 0.0), 3)

    # Regime label
    ts_pct = ts_val / max(regime_length, 1)
    regime_label = "bullish" if ts_pct > 0.6 else ("bearish" if ts_pct < 0.4 else "neutral")

    action = "BUY" if direction == 1 else "SELL"

    return {
        "action": action,
        "strength": strength,
        "entry_price": round(entry_price, 8),
        "sl_price": round(float(sl_price), 8),
        "tp1_price": round(float(tp1), 8),
        "tp2_price": round(float(tp2), 8),
        "tp3_price": round(float(tp3), 8),
        "regime": regime_label,
        "filters_active": filters_active,
        "entry_mode": entry_mode,
        "is_confluence": is_confluence,
        "confidence": confidence,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "signal_bar": int(i),
        "atr": round(float(atr_val), 8),
    }


def _hold_signal(timestamp: Optional[str], reason: str = "") -> dict[str, Any]:
    return {
        "action": "HOLD",
        "strength": 0,
        "entry_price": None,
        "sl_price": None,
        "tp1_price": None,
        "tp2_price": None,
        "tp3_price": None,
        "regime": "unknown",
        "filters_active": [],
        "entry_mode": "Pivot",
        "is_confluence": False,
        "confidence": 0.0,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def generate_signal(
    symbol: str,
    timeframe: str,
    params: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Fetch latest data and generate the current signal for *symbol*/*timeframe*.

    Uses optimized parameters from the database when available, falling back
    to DEFAULT_SIGNAL_PARAMS merged with any caller-supplied *params*.
    """
    from app.config import DEFAULT_SIGNAL_PARAMS
    from app.data_fetcher import fetch_ohlcv

    # Merge: defaults < db_params < caller params
    merged = dict(DEFAULT_SIGNAL_PARAMS)

    # Load optimized params from DB if available
    try:
        from app.database import get_db, is_db_available
        from app.models import OptimizationResult

        if is_db_available():
            db_gen = get_db()
            db = next(db_gen)
            try:
                result = (
                    db.query(OptimizationResult)
                    .filter(
                        OptimizationResult.symbol == symbol,
                        OptimizationResult.timeframe == timeframe,
                        OptimizationResult.is_current == True,
                    )
                    .first()
                )
                if result:
                    if result.left_bars is not None:
                        merged["left_bars"] = result.left_bars
                    if result.right_bars is not None:
                        merged["right_bars"] = result.right_bars
                    if result.offset is not None:
                        merged["offset"] = result.offset
                    if result.atr_multiplier is not None:
                        merged["atr_multiplier"] = result.atr_multiplier
                    if result.atr_period is not None:
                        merged["atr_period"] = result.atr_period
            finally:
                db_gen.close()
    except Exception as exc:
        logger.debug("Could not load DB params for %s %s: %s", symbol, timeframe, exc)

    if params:
        merged.update(params)

    try:
        df = fetch_ohlcv(symbol, timeframe)
        if df is None or len(df) < 50:
            return {**_hold_signal(None, "no data"), "symbol": symbol, "timeframe": timeframe}

        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

        ts = df.index[-1].isoformat() if hasattr(df.index[-1], "isoformat") else str(df.index[-1])
        sig = _generate_signal_from_arrays(high, low, close, volume, merged, timestamp=ts)
    except Exception as exc:
        logger.error("Signal generation error for %s %s: %s", symbol, timeframe, exc)
        sig = _hold_signal(None, f"error: {exc}")

    sig["symbol"] = symbol
    sig["timeframe"] = timeframe
    return sig


def generate_signals_batch(
    symbols: list[str],
    timeframe: str,
    params: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Generate signals for multiple symbols at the given timeframe."""
    results = []
    for sym in symbols:
        try:
            sig = generate_signal(sym, timeframe, params)
            results.append(sig)
        except Exception as exc:
            logger.error("Batch signal error for %s %s: %s", sym, timeframe, exc)
            results.append({**_hold_signal(None, str(exc)), "symbol": sym, "timeframe": timeframe})
    return results
