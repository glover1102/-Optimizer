"""Shared KLS+MoM strategy engine used by the signal generator and backtester."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
import math
from typing import Any

import numpy as np
import pandas as pd


_OR_EXT_FACTORS = (0.5, 1.0, 1.5, 2.0)


@dataclass
class TradeRecord:
    direction: int
    entry_bar: int
    entry_time: str
    entry_price: float
    sl_price: float
    tp_prices: dict[str, float]
    trigger_type: str
    trigger_name: str
    broken_level: float
    strength: int
    confidence: float
    regime: str
    filters_active: list[str]
    filters_passed: dict[str, Any]
    session_date: str
    be_moved: bool = False
    highest_tp_hit: int = 0
    outcome: str | None = None
    exit_bar: int | None = None
    exit_time: str | None = None
    exit_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "entry_bar": self.entry_bar,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_prices": dict(self.tp_prices),
            "trigger_type": self.trigger_type,
            "trigger_name": self.trigger_name,
            "broken_level": self.broken_level,
            "strength": self.strength,
            "confidence": self.confidence,
            "regime": self.regime,
            "filters_active": list(self.filters_active),
            "filters_passed": dict(self.filters_passed),
            "session_date": self.session_date,
            "be_moved": self.be_moved,
            "highest_tp_hit": self.highest_tp_hit,
            "outcome": self.outcome,
            "exit_bar": self.exit_bar,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
        }


@dataclass
class ORState:
    enabled: bool
    duration_bars: int
    ext_enabled: bool
    high: float = math.nan
    low: float = math.nan
    locked: bool = False


def timeframe_to_minutes(timeframe: str | None, default: int = 60) -> int:
    if not timeframe:
        return default
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return max(1, int(tf[:-1]))
    if tf.endswith("h"):
        return max(1, int(tf[:-1])) * 60
    if tf.endswith("d"):
        return max(1, int(tf[:-1])) * 1440
    if tf.endswith("w"):
        return max(1, int(tf[:-1])) * 10080
    return default


def infer_timeframe_minutes(timestamps: pd.DatetimeIndex | None, fallback: int = 60) -> int:
    if timestamps is None or len(timestamps) < 2:
        return fallback
    diffs = timestamps.to_series().diff().dropna()
    if diffs.empty:
        return fallback
    seconds = float(diffs.median().total_seconds())
    if seconds <= 0:
        return fallback
    return max(1, int(round(seconds / 60.0)))


def normalize_timestamps(
    timestamps: Any | None,
    n: int,
    timeframe_minutes: int,
) -> pd.DatetimeIndex:
    if timestamps is None:
        return pd.date_range(
            start="2024-01-01T00:00:00Z",
            periods=n,
            freq=pd.Timedelta(minutes=timeframe_minutes),
            tz="UTC",
        )
    idx = pd.to_datetime(timestamps, utc=True)
    if len(idx) != n:
        raise ValueError("timestamps length must match price arrays")
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    return pd.DatetimeIndex(idx)


def derive_open(close: np.ndarray, open_: np.ndarray | None = None) -> np.ndarray:
    if open_ is not None:
        return np.asarray(open_, dtype=np.float64)
    out = np.empty_like(close, dtype=np.float64)
    out[0] = close[0]
    out[1:] = close[:-1]
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = np.empty(len(close), dtype=np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    return tr


def sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if period <= 0 or len(arr) < period:
        return out
    csum = np.cumsum(np.insert(arr.astype(np.float64), 0, 0.0))
    vals = (csum[period:] - csum[:-period]) / period
    out[period - 1 :] = vals
    return out


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if period <= 0 or len(arr) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def rma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if period <= 0 or len(arr) < period:
        return out
    alpha = 1.0 / period
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def wma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if period <= 0 or len(arr) < period:
        return out
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = float(weights.sum())
    for i in range(period - 1, len(arr)):
        out[i] = float(np.dot(arr[i - period + 1 : i + 1], weights) / denom)
    return out


def moving_average(arr: np.ndarray, period: int, kind: str) -> np.ndarray:
    kind = kind.upper()
    if kind == "EMA":
        return ema(arr, period)
    if kind == "WMA":
        return wma(arr, period)
    if kind == "RMA":
        return rma(arr, period)
    return sma(arr, period)


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, smoothing: str = "RMA") -> np.ndarray:
    return moving_average(true_range(high, low, close), period, smoothing)


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) < period + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    mask = ~np.isnan(avg_gain) & ~np.isnan(avg_loss)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss != 0)
    out[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    return out


def macd_hist(close: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(np.nan_to_num(macd_line, nan=0.0), signal)
    hist = macd_line - signal_line
    hist[np.isnan(macd_line) | np.isnan(signal_line)] = np.nan
    return hist


def session_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=np.float64)
    pv = 0.0
    vv = 0.0
    prev_day = None
    typical = (high + low + close) / 3.0
    for i, ts in enumerate(timestamps):
        day = ts.date()
        if day != prev_day:
            pv = 0.0
            vv = 0.0
            prev_day = day
        pv += typical[i] * max(volume[i], 0.0)
        vv += max(volume[i], 0.0)
        out[i] = typical[i] if vv <= 0 else pv / vv
    return out


def approximate_structure_bias(close: np.ndarray, timeframe_minutes: int, target_minutes: int) -> np.ndarray:
    """Approximate HTF structure on chart bars because request.security is unavailable."""
    out = np.zeros(len(close), dtype=np.int8)
    bars = max(2, int(round(target_minutes / max(timeframe_minutes, 1))))
    base = sma(close, max(3, bars * 2))
    slope = np.full(len(close), np.nan)
    slope[bars:] = base[bars:] - base[:-bars]
    for i in range(len(close)):
        if np.isnan(base[i]) or np.isnan(slope[i]):
            continue
        if close[i] > base[i] and slope[i] > 0:
            out[i] = 1
        elif close[i] < base[i] and slope[i] < 0:
            out[i] = -1
    return out


def resolve_directional(direction: int, value: int, filt_mode: str) -> bool:
    if filt_mode == "Strict":
        return value == direction
    return value != -direction


def enabled_tp_names(params: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for name in ("TP1", "TP2", "TP3", "TP4"):
        if params.get(f"enable_{name.lower()}", True):
            names.append(name)
    return names


def prepare_kls_context(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    timestamps: Any | None,
    params: dict[str, Any],
    timeframe: str | None = None,
) -> dict[str, Any]:
    n = len(close)
    timeframe_minutes = int(params.get("timeframe_minutes") or timeframe_to_minutes(timeframe, 60))
    idx = normalize_timestamps(timestamps, n, timeframe_minutes)
    timeframe_minutes = infer_timeframe_minutes(idx, timeframe_minutes)

    atr_len = int(params.get("atr_length", 14))
    atr_smoothing = str(params.get("atr_smoothing", "RMA"))
    atr_arr = atr(high, low, close, atr_len, atr_smoothing)
    session_vwap_arr = session_vwap(high, low, close, volume, idx)

    ema_periods = [p for p, enabled in [
        (int(params.get("ema_9", 9)), params.get("ema_9_enabled", True)),
        (int(params.get("ema_21", 21)), params.get("ema_21_enabled", True)),
        (int(params.get("ema_50", 50)), params.get("ema_50_enabled", True)),
        (int(params.get("ema_100", 100)), params.get("ema_100_enabled", False)),
        (int(params.get("ema_200", 200)), params.get("ema_200_enabled", False)),
        (int(params.get("ema_300", 300)), params.get("ema_300_enabled", False)),
    ] if enabled]
    ema_map = {period: ema(close, period) for period in ema_periods}

    vol_ma = moving_average(volume, int(params.get("fvol_len", 20)), str(params.get("fvol_ma_type", "SMA")))
    rsi_arr = rsi(close, int(params.get("rsi_length", 14)))
    macd_arr = macd_hist(close, int(params.get("macd_fast", 12)), int(params.get("macd_slow", 26)), int(params.get("macd_signal", 9)))
    atr_baseline = sma(np.nan_to_num(atr_arr, nan=0.0), int(params.get("fatr_baseline", 20)))

    struct_biases = {
        "15m": approximate_structure_bias(close, timeframe_minutes, 15),
        "1h": approximate_structure_bias(close, timeframe_minutes, 60),
        "4h": approximate_structure_bias(close, timeframe_minutes, 240),
    }

    days = idx.normalize()
    new_sessions = np.zeros(n, dtype=bool)
    new_sessions[0] = True
    new_sessions[1:] = days[1:] != days[:-1]

    return {
        "timestamps": idx,
        "timeframe_minutes": timeframe_minutes,
        "atr": atr_arr,
        "session_vwap": session_vwap_arr,
        "ema_map": ema_map,
        "volume_ma": vol_ma,
        "rsi": rsi_arr,
        "macd_hist": macd_arr,
        "atr_baseline": atr_baseline,
        "structure_biases": struct_biases,
        "new_sessions": new_sessions,
    }


def _collect_filters(i: int, context: dict[str, Any], close: np.ndarray, volume: np.ndarray, params: dict[str, Any]) -> tuple[bool, bool, str, list[str], dict[str, Any], int]:
    filt_mode = str(params.get("filt_mode", "Strict"))
    filters_active: list[str] = []
    filters_passed: dict[str, Any] = {}
    directional_scores: list[int] = []
    agreeing = 0

    struct_dirs: list[int] = []
    if params.get("use_structure_15m", True):
        struct_dirs.append(int(context["structure_biases"]["15m"][i]))
    if params.get("use_structure_1h", False):
        struct_dirs.append(int(context["structure_biases"]["1h"][i]))
    if params.get("use_structure_4h", False):
        struct_dirs.append(int(context["structure_biases"]["4h"][i]))
    if struct_dirs:
        filters_active.append("structure")
        struct_dir = 1 if all(v == 1 for v in struct_dirs) else (-1 if all(v == -1 for v in struct_dirs) else 0)
        directional_scores.append(struct_dir)
        filters_passed["structure"] = struct_dir

    if params.get("use_fmom", True):
        filters_active.append("ema_stack")
        ema_vals = [arr[i] for arr in context["ema_map"].values() if not np.isnan(arr[i])]
        mom_dir = 0
        if ema_vals:
            if close[i] > max(ema_vals):
                mom_dir = 1
            elif close[i] < min(ema_vals):
                mom_dir = -1
            if params.get("fmom_strict_fan", False) and len(ema_vals) > 1:
                ordered = [context["ema_map"][k][i] for k in sorted(context["ema_map"].keys()) if not np.isnan(context["ema_map"][k][i])]
                if mom_dir == 1 and ordered != sorted(ordered, reverse=True):
                    mom_dir = 0
                if mom_dir == -1 and ordered != sorted(ordered):
                    mom_dir = 0
        directional_scores.append(mom_dir)
        filters_passed["ema_stack"] = mom_dir

    if params.get("use_frsi", True):
        filters_active.append("rsi")
        neutral = float(params.get("rsi_neutral", 5.0))
        rsi_val = context["rsi"][i]
        rsi_dir = 0
        if not np.isnan(rsi_val):
            if rsi_val > 50.0 + neutral:
                rsi_dir = 1
            elif rsi_val < 50.0 - neutral:
                rsi_dir = -1
        directional_scores.append(rsi_dir)
        filters_passed["rsi"] = rsi_dir

    if params.get("use_fmacd", True):
        filters_active.append("macd")
        macd_val = context["macd_hist"][i]
        macd_dir = 0 if np.isnan(macd_val) or macd_val == 0 else (1 if macd_val > 0 else -1)
        directional_scores.append(macd_dir)
        filters_passed["macd"] = macd_dir

    if params.get("use_fvwap", True):
        filters_active.append("vwap")
        vwap_val = context["session_vwap"][i]
        vwap_dir = 0
        if not np.isnan(vwap_val):
            if close[i] > vwap_val:
                vwap_dir = 1
            elif close[i] < vwap_val:
                vwap_dir = -1
        directional_scores.append(vwap_dir)
        filters_passed["vwap"] = vwap_dir

    if params.get("use_fvol", True):
        filters_active.append("volume")
        ma_val = context["volume_ma"][i]
        vol_ok = bool(not np.isnan(ma_val) and volume[i] > ma_val * float(params.get("fvol_mult", 1.0)))
        filters_passed["volume"] = vol_ok
    else:
        vol_ok = True

    if params.get("use_fatr", False):
        filters_active.append("atr_state")
        atr_val = context["atr"][i]
        base = context["atr_baseline"][i]
        atr_ok = False
        state = "unknown"
        if not np.isnan(atr_val) and not np.isnan(base) and base > 0:
            ratio = atr_val / base
            if ratio < 0.8:
                state = "Low"
            elif ratio > 1.2:
                state = "High"
            else:
                state = "Stable"
            allow = str(params.get("fatr_allow", "Only Stable"))
            atr_ok = (
                (allow == "Only Stable" and state == "Stable")
                or (allow == "Stable or Low" and state in {"Stable", "Low"})
                or (allow == "Stable or High" and state in {"Stable", "High"})
            )
        filters_passed["atr_state"] = state
    else:
        atr_ok = True

    def directional_ok(direction: int) -> bool:
        for value in directional_scores:
            if not resolve_directional(direction, value, filt_mode):
                return False
        return True

    long_ok = directional_ok(1) and vol_ok and atr_ok
    short_ok = directional_ok(-1) and vol_ok and atr_ok

    for value in directional_scores:
        if value == 1:
            agreeing += 1
    if vol_ok and params.get("use_fvol", True):
        agreeing += 1
    if atr_ok and params.get("use_fatr", False):
        agreeing += 1

    net = sum(directional_scores)
    regime = "bullish" if net > 0 else ("bearish" if net < 0 else "neutral")
    strength = max(0, min(4, agreeing))
    return long_ok, short_ok, regime, filters_active, filters_passed, strength


def _final_target_name(params: dict[str, Any]) -> str:
    enabled = enabled_tp_names(params)
    if not enabled:
        return "TP1"
    if params.get("clear_on_tp", False):
        chosen = str(params.get("clear_tp_sel", "TP1"))
        return chosen if chosen in enabled else enabled[0]
    runner = str(params.get("runner_tgt", "TP4"))
    if runner == "Last enabled TP":
        return enabled[-1]
    return runner if runner in enabled else enabled[-1]


def _exit_trade(trade: TradeRecord, bar: int, timestamp: pd.Timestamp, price: float, outcome: str) -> None:
    trade.exit_bar = bar
    trade.exit_time = timestamp.isoformat()
    trade.exit_price = float(price)
    trade.outcome = outcome


def simulate_kls_strategy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None = None,
    *,
    open_: np.ndarray | None = None,
    timestamps: Any | None = None,
    timeframe: str | None = None,
    params: dict[str, Any] | None = None,
    include_debug: bool = False,
) -> dict[str, Any]:
    params = dict(params or {})
    from app.config import DEFAULT_SIGNAL_PARAMS

    merged = dict(DEFAULT_SIGNAL_PARAMS)
    merged.update(params)

    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    volume = np.ones_like(close, dtype=np.float64) if volume is None else np.asarray(volume, dtype=np.float64)
    open_arr = derive_open(close, None if open_ is None else np.asarray(open_, dtype=np.float64))

    if len(close) < 5:
        return {"trades": [], "current_trade": None, "context": None, "debug": {}}

    context = prepare_kls_context(open_arr, high, low, close, volume, timestamps, merged, timeframe)
    idx = context["timestamps"]
    timeframe_minutes = context["timeframe_minutes"]
    atr_arr = context["atr"]

    or_states_template = [
        ORState(bool(merged.get(f"or{i}_enabled", i == 4)), max(1, int(math.ceil(float(merged.get(f"or{i}_minutes", default)) / timeframe_minutes))), bool(merged.get(f"or{i}_ext_enabled", i in (2, 3, 4))))
        for i, default in enumerate((0, 5, 360, 480, 720)) if i > 0
    ]

    prev_day = None
    prev_week = None
    current_day = None
    current_week = None
    day_open = day_high = day_low = day_close = math.nan
    week_high = week_low = math.nan
    session_open = math.nan
    session_bar = 0
    or_states: list[ORState] = []
    armed: dict[str, dict[str, bool]] = {}

    active_trade: TradeRecord | None = None
    trades: list[TradeRecord] = []
    final_target = _final_target_name(merged)
    tp_rr = {
        "TP1": float(merged.get("tp1_rr", 0.75)),
        "TP2": float(merged.get("tp2_rr", 1.5)),
        "TP3": float(merged.get("tp3_rr", 2.25)),
        "TP4": float(merged.get("tp4_rr", 3.0)),
    }
    enabled_tps = enabled_tp_names(merged)
    be_trigger = str(merged.get("be_after_tp", "TP1"))
    be_shift = float(merged.get("be_off_ticks", 0.0))
    resolution = str(merged.get("resolve_mode", "First touch"))

    debug_levels: list[dict[str, float]] = []

    for i, ts in enumerate(idx):
        day = ts.date()
        iso = ts.isocalendar()
        week = (int(iso.year), int(iso.week))
        new_session = day != current_day

        if new_session:
            if current_day is not None:
                prev_day = {"open": day_open, "high": day_high, "low": day_low, "close": day_close}
            if current_week is not None and week != current_week:
                prev_week = {"high": week_high, "low": week_low}
            current_day = day
            if current_week is None or week != current_week:
                current_week = week
                week_high = high[i]
                week_low = low[i]
            session_open = open_arr[i]
            day_open = open_arr[i]
            day_high = high[i]
            day_low = low[i]
            day_close = close[i]
            session_bar = 0
            or_states = [ORState(s.enabled, s.duration_bars, s.ext_enabled) for s in or_states_template]
            armed = {}
            if active_trade is not None and active_trade.outcome is None:
                _exit_trade(active_trade, i, ts, close[i], "session_close")
                trades.append(active_trade)
                active_trade = None
        else:
            day_high = max(day_high, high[i])
            day_low = min(day_low, low[i])
            day_close = close[i]
            week_high = max(week_high, high[i])
            week_low = min(week_low, low[i])
            session_bar += 1

        levels: dict[str, float] = {}
        if merged.get("enable_pdl", True) and prev_day is not None:
            levels["PDL"] = float(prev_day["low"])
        if merged.get("enable_pdo", True) and prev_day is not None:
            levels["PDO"] = float(prev_day["open"])
        if merged.get("enable_pdc", True) and prev_day is not None:
            levels["PDC"] = float(prev_day["close"])
        if merged.get("enable_pdh", True) and prev_day is not None:
            levels["PDH"] = float(prev_day["high"])
        if merged.get("enable_pwh", False) and prev_week is not None:
            levels["PWH"] = float(prev_week["high"])
        if merged.get("enable_pwl", False) and prev_week is not None:
            levels["PWL"] = float(prev_week["low"])
        if merged.get("enable_open", False):
            levels["OPEN"] = float(session_open)

        for n_or, state in enumerate(or_states, start=1):
            if not state.enabled:
                continue
            if session_bar < state.duration_bars:
                state.high = high[i] if np.isnan(state.high) else max(state.high, high[i])
                state.low = low[i] if np.isnan(state.low) else min(state.low, low[i])
                if session_bar == state.duration_bars - 1:
                    state.locked = True
            if state.locked:
                if merged.get("or_as_trigger", True):
                    levels[f"OR{n_or}_HIGH"] = float(state.high)
                    levels[f"OR{n_or}_LOW"] = float(state.low)
                if state.ext_enabled and merged.get("or_ext_as_trigger", True):
                    width = float(state.high - state.low)
                    if width > 0:
                        for factor in _OR_EXT_FACTORS:
                            if merged.get(f"or_ext_{str(factor).replace('.', '_')}_enabled", factor in (1.5, 2.0)):
                                suffix = str(factor).replace(".", "_")
                                levels[f"OR{n_or}_UP_{suffix}"] = float(state.high + width * factor)
                                levels[f"OR{n_or}_DN_{suffix}"] = float(state.low - width * factor)

        debug_levels.append(dict(levels) if include_debug else {})

        if i == 0:
            continue

        for name, price in levels.items():
            armed.setdefault(name, {"long": True, "short": True})
            if merged.get("re_arm", True):
                if close[i] < price:
                    armed[name]["long"] = True
                if close[i] > price:
                    armed[name]["short"] = True

        crossed_up = [(name, price) for name, price in levels.items() if close[i - 1] <= price < close[i] and armed.get(name, {}).get("long", True)]
        crossed_down = [(name, price) for name, price in levels.items() if close[i - 1] >= price > close[i] and armed.get(name, {}).get("short", True)]

        if active_trade is not None and active_trade.outcome is None and i > active_trade.entry_bar:
            final_price = active_trade.tp_prices[final_target]
            current_sl = active_trade.sl_price
            direction = active_trade.direction
            hard_stop = (low[i] <= current_sl) if direction == 1 else (high[i] >= current_sl)
            hard_target = (high[i] >= final_price) if direction == 1 else (low[i] <= final_price)

            if resolution == "First touch" and hard_stop and hard_target:
                _exit_trade(active_trade, i, ts, current_sl, "sl_hit")
                trades.append(active_trade)
                active_trade = None
            else:
                hit_names: list[str] = []
                for name in enabled_tps:
                    price = active_trade.tp_prices[name]
                    touched = (high[i] >= price) if direction == 1 else (low[i] <= price)
                    closed = (close[i] >= price) if direction == 1 else (close[i] <= price)
                    if name.endswith(str(active_trade.highest_tp_hit)):
                        pass
                    if ((resolution == "First touch" and touched) or (resolution != "First touch" and closed)):
                        level_num = int(name[-1])
                        if level_num > active_trade.highest_tp_hit:
                            active_trade.highest_tp_hit = level_num
                            hit_names.append(name)
                if hit_names and be_trigger in hit_names and not active_trade.be_moved:
                    active_trade.sl_price = active_trade.entry_price + be_shift if direction == 1 else active_trade.entry_price - be_shift
                    active_trade.be_moved = True

                final_reached = active_trade.highest_tp_hit >= int(final_target[-1])
                if final_reached:
                    _exit_trade(active_trade, i, ts, active_trade.tp_prices[final_target], f"{final_target.lower()}_hit")
                    trades.append(active_trade)
                    active_trade = None
                else:
                    if resolution == "First touch":
                        stop_now = (low[i] <= active_trade.sl_price) if direction == 1 else (high[i] >= active_trade.sl_price)
                    else:
                        stop_now = (close[i] <= active_trade.sl_price) if direction == 1 else (close[i] >= active_trade.sl_price)
                    if stop_now:
                        outcome = "breakeven" if active_trade.be_moved and active_trade.sl_price == active_trade.entry_price else "sl_hit"
                        _exit_trade(active_trade, i, ts, active_trade.sl_price, outcome)
                        trades.append(active_trade)
                        active_trade = None

        if active_trade is None and (crossed_up or crossed_down):
            long_ok, short_ok, regime, filters_active, filters_passed, strength = _collect_filters(i, context, close, volume, merged)
            confidence = round(min(1.0, strength / 4.0), 3)
            chosen = None
            direction = 0
            if crossed_up and long_ok:
                chosen = max(crossed_up, key=lambda item: item[1])
                direction = 1
            elif crossed_down and short_ok:
                chosen = min(crossed_down, key=lambda item: item[1])
                direction = -1

            if chosen and not np.isnan(atr_arr[i]) and atr_arr[i] > 0:
                trigger_name, broken_level = chosen
                entry_price = float(broken_level if merged.get("entry_at_level", False) else close[i])
                risk = float(atr_arr[i] * float(merged.get("sl_mult", 1.5)))
                sl_price = entry_price - risk if direction == 1 else entry_price + risk
                tp_prices = {name: entry_price + direction * risk * tp_rr[name] for name in enabled_tps}
                trigger_type = "KeyLevel"
                if trigger_name.startswith("OR") and "UP" not in trigger_name and "DN" not in trigger_name and (trigger_name.endswith("HIGH") or trigger_name.endswith("LOW")):
                    trigger_type = "OR"
                elif trigger_name.startswith("OR"):
                    trigger_type = "OR-Ext"
                active_trade = TradeRecord(
                    direction=direction,
                    entry_bar=i,
                    entry_time=ts.isoformat(),
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_prices=tp_prices,
                    trigger_type=trigger_type,
                    trigger_name=trigger_name,
                    broken_level=float(broken_level),
                    strength=strength,
                    confidence=confidence,
                    regime=regime,
                    filters_active=filters_active,
                    filters_passed=filters_passed,
                    session_date=str(day),
                )
                if direction == 1:
                    armed[trigger_name]["long"] = False
                else:
                    armed[trigger_name]["short"] = False

    if active_trade is not None and active_trade.outcome is None:
        # Keep the current trade open for signal generation.
        pass

    latest_trade = active_trade or (trades[-1] if trades else None)
    current_trade = active_trade.to_dict() if active_trade is not None else None

    return {
        "trades": [t.to_dict() for t in trades],
        "current_trade": current_trade,
        "latest_trade": latest_trade.to_dict() if latest_trade is not None else None,
        "context": context,
        "debug": {"levels": debug_levels} if include_debug else {},
    }


def empty_backtest_result() -> dict[str, Any]:
    return {
        "total_signals": 0,
        "wins": 0,
        "tp1_hits": 0,
        "tp2_hits": 0,
        "tp3_hits": 0,
        "tp4_hits": 0,
        "sl_hits": 0,
        "win_rate": 0.0,
        "tp1_rate": 0.0,
        "tp2_rate": 0.0,
        "tp3_rate": 0.0,
        "tp4_rate": 0.0,
        "sl_rate": 0.0,
        "trades": [],
    }


def summarize_backtest(simulation: dict[str, Any]) -> dict[str, Any]:
    trades = list(simulation["trades"])
    total = len(trades)
    if total == 0:
        return empty_backtest_result()
    wins = sum(1 for t in trades if t.get("highest_tp_hit", 0) >= 1)
    tp1_hits = wins
    tp2_hits = sum(1 for t in trades if t.get("highest_tp_hit", 0) >= 2)
    tp3_hits = sum(1 for t in trades if t.get("highest_tp_hit", 0) >= 3)
    tp4_hits = sum(1 for t in trades if t.get("highest_tp_hit", 0) >= 4)
    sl_hits = sum(1 for t in trades if t.get("outcome") == "sl_hit")
    return {
        "total_signals": total,
        "wins": wins,
        "tp1_hits": tp1_hits,
        "tp2_hits": tp2_hits,
        "tp3_hits": tp3_hits,
        "tp4_hits": tp4_hits,
        "sl_hits": sl_hits,
        "win_rate": round(wins / total, 4),
        "tp1_rate": round(tp1_hits / total, 4),
        "tp2_rate": round(tp2_hits / total, 4),
        "tp3_rate": round(tp3_hits / total, 4),
        "tp4_rate": round(tp4_hits / total, 4),
        "sl_rate": round(sl_hits / total, 4),
        "trades": trades,
    }
