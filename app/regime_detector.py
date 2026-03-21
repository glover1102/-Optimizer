"""
Market regime detection.

Determines whether the current market is:
 - "trending"  : ADX > 25, clear directional movement
 - "ranging"   : ADX < 20, Bollinger Band width squeeze
 - "volatile"  : ATR ratio > 1.5× its 50-period average
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
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


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute ADX (Wilder's smoothing, matching standard implementations)."""
    n = len(high)
    if n < period * 2:
        return np.full(n, np.nan)

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.empty(n - 1)
    tr[:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )

    alpha = 1.0 / period
    tr_s = np.empty(n - 1)
    pdm_s = np.empty(n - 1)
    mdm_s = np.empty(n - 1)

    tr_s[period - 1] = tr[:period].sum()
    pdm_s[period - 1] = plus_dm[:period].sum()
    mdm_s[period - 1] = minus_dm[:period].sum()

    for i in range(period, n - 1):
        tr_s[i] = tr_s[i - 1] * (1 - alpha) + tr[i]
        pdm_s[i] = pdm_s[i - 1] * (1 - alpha) + plus_dm[i]
        mdm_s[i] = mdm_s[i - 1] * (1 - alpha) + minus_dm[i]

    pdi = np.where(tr_s > 0, 100 * pdm_s / tr_s, 0.0)
    mdi = np.where(tr_s > 0, 100 * mdm_s / tr_s, 0.0)

    dx = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)

    adx_vals = np.full(n - 1, np.nan)
    adx_vals[period * 2 - 2] = dx[period - 1 : period * 2 - 1].mean()
    for i in range(period * 2 - 1, n - 1):
        adx_vals[i] = adx_vals[i - 1] * (1 - alpha) + dx[i] * alpha

    result = np.full(n, np.nan)
    result[1:] = adx_vals
    return result


def _bollinger_width(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Return Bollinger Band width as a fraction of the middle band."""
    n = len(close)
    width = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        mid = window.mean()
        std = window.std(ddof=1)
        if mid > 0:
            width[i] = (2 * 2 * std) / mid  # 2-sigma band / mid
    return width


def detect_regime(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    adx_period: int = 14,
    atr_period: int = 14,
    atr_avg_period: int = 50,
) -> dict[str, Any]:
    """
    Detect the current market regime.

    Returns
    -------
    {
        "regime": "trending" | "ranging" | "volatile",
        "confidence": float (0.0–1.0),
        "adx": float,
        "atr_ratio": float,
        "bb_width": float,
    }
    """
    n = len(close)
    if n < max(adx_period * 3, atr_avg_period + atr_period, 60):
        return {"regime": "ranging", "confidence": 0.0, "adx": np.nan, "atr_ratio": np.nan, "bb_width": np.nan}

    adx_arr = _adx(high, low, close, adx_period)
    current_adx = float(adx_arr[~np.isnan(adx_arr)][-1]) if not np.all(np.isnan(adx_arr)) else np.nan

    atr_arr = _atr(high, low, close, atr_period)
    valid_atr = atr_arr[~np.isnan(atr_arr)]
    if len(valid_atr) < atr_avg_period:
        atr_ratio = np.nan
    else:
        current_atr = valid_atr[-1]
        avg_atr = valid_atr[-atr_avg_period:].mean()
        atr_ratio = float(current_atr / avg_atr) if avg_atr > 0 else np.nan

    bb_arr = _bollinger_width(close)
    valid_bb = bb_arr[~np.isnan(bb_arr)]
    current_bb = float(valid_bb[-1]) if len(valid_bb) > 0 else np.nan
    avg_bb = float(valid_bb[-20:].mean()) if len(valid_bb) >= 20 else np.nan

    # Regime decision
    if not np.isnan(atr_ratio) and atr_ratio > 1.5:
        regime = "volatile"
        confidence = min(1.0, (atr_ratio - 1.5) / 0.5)
    elif not np.isnan(current_adx) and current_adx > 25:
        regime = "trending"
        confidence = min(1.0, (current_adx - 25) / 25)
    else:
        regime = "ranging"
        adx_conf = max(0.0, (20 - (current_adx or 20)) / 20) if not np.isnan(current_adx) else 0.5
        bb_conf = (
            max(0.0, 1.0 - current_bb / avg_bb)
            if not np.isnan(current_bb) and not np.isnan(avg_bb) and avg_bb > 0
            else 0.5
        )
        confidence = (adx_conf + bb_conf) / 2

    return {
        "regime": regime,
        "confidence": round(float(confidence), 3),
        "adx": round(current_adx, 2) if not np.isnan(current_adx) else None,
        "atr_ratio": round(atr_ratio, 3) if not np.isnan(atr_ratio) else None,
        "bb_width": round(current_bb, 4) if not np.isnan(current_bb) else None,
    }
