"""Signal generator for the KLS+MoM strategy."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from app.kls_mom import simulate_kls_strategy

logger = logging.getLogger(__name__)


def _hold_signal(timestamp: Optional[str], reason: str = "", entry_mode: str = "KeyLevel") -> dict[str, Any]:
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
        "entry_mode": entry_mode,
        "is_confluence": False,
        "confidence": 0.0,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }


def _trade_to_signal(trade: dict[str, Any], symbol: str | None, timeframe: str | None) -> dict[str, Any]:
    direction = int(trade["direction"])
    action = "BUY" if direction == 1 else "SELL"
    tp_prices = trade.get("tp_prices", {})
    filters = trade.get("filters_active", [])
    return {
        "action": action,
        "strength": int(trade.get("strength", 0)),
        "entry_price": round(float(trade.get("entry_price")), 8),
        "sl_price": round(float(trade.get("sl_price")), 8),
        "tp1_price": round(float(tp_prices.get("TP1")), 8) if tp_prices.get("TP1") is not None else None,
        "tp2_price": round(float(tp_prices.get("TP2")), 8) if tp_prices.get("TP2") is not None else None,
        "tp3_price": round(float(tp_prices.get("TP3")), 8) if tp_prices.get("TP3") is not None else None,
        "tp4_price": round(float(tp_prices.get("TP4")), 8) if tp_prices.get("TP4") is not None else None,
        "regime": trade.get("regime", "neutral"),
        "filters_active": filters,
        "entry_mode": trade.get("trigger_type", "KeyLevel"),
        "is_confluence": len(filters) >= 3,
        "confidence": float(trade.get("confidence", 0.0)),
        "timestamp": trade.get("entry_time") or datetime.now(timezone.utc).isoformat(),
        "broken_level": trade.get("broken_level"),
        "broken_level_name": trade.get("trigger_name"),
        "symbol": symbol,
        "timeframe": timeframe,
    }


def _generate_signal_from_arrays(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    params: dict,
    timestamps=None,
    timeframe: str | None = None,
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """Generate the current KLS+MoM signal from OHLCV arrays."""
    if len(close) < 5:
        return _hold_signal(timestamp, "insufficient data")

    sim = simulate_kls_strategy(
        high,
        low,
        close,
        volume,
        open_=open_,
        timestamps=timestamps,
        timeframe=timeframe,
        params=params,
    )
    trade = sim.get("current_trade")
    if trade is None:
        return _hold_signal(timestamp, "no active bracket")
    return _trade_to_signal(trade, None, timeframe)


def generate_signal(symbol: str, timeframe: str, params: Optional[dict] = None) -> dict[str, Any]:
    """Fetch latest data and generate a current KLS+MoM signal."""
    from app.config import DEFAULT_SIGNAL_PARAMS
    from app.data_fetcher import fetch_ohlcv

    merged = dict(DEFAULT_SIGNAL_PARAMS)
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
                    for key in DEFAULT_SIGNAL_PARAMS:
                        if hasattr(result, key):
                            value = getattr(result, key)
                            if value is not None:
                                merged[key] = value
            finally:
                db_gen.close()
    except Exception as exc:
        logger.debug("Could not load DB params for %s %s: %s", symbol, timeframe, exc)

    if params:
        merged.update(params)

    try:
        df = fetch_ohlcv(symbol, timeframe)
        if df is None or len(df) < 20:
            return {**_hold_signal(None, "no data"), "symbol": symbol, "timeframe": timeframe}
        sig = _generate_signal_from_arrays(
            df["open"].to_numpy(dtype=np.float64),
            df["high"].to_numpy(dtype=np.float64),
            df["low"].to_numpy(dtype=np.float64),
            df["close"].to_numpy(dtype=np.float64),
            df["volume"].to_numpy(dtype=np.float64),
            merged,
            timestamps=df.index,
            timeframe=timeframe,
            timestamp=df.index[-1].isoformat() if hasattr(df.index[-1], "isoformat") else str(df.index[-1]),
        )
    except Exception as exc:
        logger.error("Signal generation error for %s %s: %s", symbol, timeframe, exc)
        sig = _hold_signal(None, f"error: {exc}")

    sig["symbol"] = symbol
    sig["timeframe"] = timeframe
    return sig


def generate_signals_batch(symbols: list[str], timeframe: str, params: Optional[dict] = None) -> list[dict[str, Any]]:
    """Generate signals for multiple symbols at the given timeframe."""
    out: list[dict[str, Any]] = []
    for sym in symbols:
        try:
            out.append(generate_signal(sym, timeframe, params))
        except Exception as exc:
            logger.error("Batch signal error for %s %s: %s", sym, timeframe, exc)
            out.append({**_hold_signal(None, str(exc)), "symbol": sym, "timeframe": timeframe})
    return out
