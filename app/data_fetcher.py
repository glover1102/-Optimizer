"""
Multi-source OHLCV data fetcher.

- ccxt (Binance) for crypto symbols
- yfinance for forex, stocks, indices and futures
- In-memory caching to avoid redundant downloads within a single run
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Simple in-memory cache: key → (fetched_at, DataFrame)
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour

# Timeframe mappings
_YFINANCE_TF_MAP = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "1h",   # yfinance has no 4h; we resample from 1h
    "1d": "1d",
}

_CCXT_TF_MAP = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Crypto symbols served by ccxt
_CRYPTO_BASE = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"}


def _cache_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}:{timeframe}"


def _is_crypto(symbol: str) -> bool:
    return symbol in _CRYPTO_BASE or (
        symbol.endswith("USDT") and not symbol.endswith("=X") and "=" not in symbol
    )


# ── yfinance helpers ──────────────────────────────────────────────────────────


def _yf_period_for_tf(timeframe: str) -> str:
    """Return the 'period' argument for yfinance based on timeframe."""
    if timeframe in ("5m", "15m"):
        return "60d"
    if timeframe in ("1h", "4h"):
        return "730d"
    return "5y"


def _fetch_yfinance(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf  # lazy import

        yf_tf = _YFINANCE_TF_MAP.get(timeframe, "1d")
        period = _yf_period_for_tf(timeframe)

        time.sleep(0.5)  # Be nice to yfinance API

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=yf_tf, auto_adjust=True)

        if df.empty:
            logger.warning(
                "yfinance returned empty data for %s %s (period=%s); trying date-range fallback",
                symbol, timeframe, period,
            )
            # Known yfinance workaround: explicit start/end can succeed when period does not
            end = datetime.now()
            if timeframe in ("5m", "15m"):
                start = end - timedelta(days=59)
            elif timeframe in ("1h", "4h"):
                start = end - timedelta(days=729)
            else:
                start = end - timedelta(days=365 * 5)
            df = ticker.history(start=start, end=end, interval=yf_tf, auto_adjust=True)

        if df.empty:
            logger.warning("yfinance returned empty data for %s %s (both period and date-range)", symbol, timeframe)
            return None

        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.dropna()

        # Resample 1h → 4h if needed
        if timeframe == "4h":
            df = df.resample("4h").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

        logger.info("yfinance: fetched %d bars for %s %s", len(df), symbol, timeframe)
        return df

    except Exception as exc:
        logger.error("yfinance error for %s %s: %s", symbol, timeframe, exc)
        return None


# ── ccxt helpers ──────────────────────────────────────────────────────────────


def _fetch_ccxt(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    try:
        import ccxt  # lazy import
        from app.config import BINANCE_API_KEY, BINANCE_SECRET

        exchange = ccxt.binance({
            "apiKey": BINANCE_API_KEY or None,
            "secret": BINANCE_SECRET or None,
            "enableRateLimit": True,
        })

        ccxt_tf = _CCXT_TF_MAP.get(timeframe, "1h")
        # Convert BTCUSDT → BTC/USDT
        ccxt_symbol = symbol[:-4] + "/" + symbol[-4:] if symbol.endswith("USDT") else symbol

        limit = 1000
        ohlcv = exchange.fetch_ohlcv(ccxt_symbol, ccxt_tf, limit=limit)

        if not ohlcv:
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").dropna()
        logger.info("ccxt: fetched %d bars for %s %s", len(df), symbol, timeframe)
        return df

    except Exception as exc:
        logger.error("ccxt error for %s %s: %s", symbol, timeframe, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_ohlcv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for *symbol* at *timeframe*.

    Returns a DataFrame with columns [open, high, low, close, volume]
    indexed by UTC datetime, or None on failure.
    Uses an in-memory cache keyed by (symbol, timeframe).
    """
    key = _cache_key(symbol, timeframe)
    now = time.monotonic()

    if key in _cache:
        fetched_at, cached_df = _cache[key]
        if now - fetched_at < _CACHE_TTL_SECONDS:
            return cached_df

    if _is_crypto(symbol):
        df = _fetch_ccxt(symbol, timeframe)
        if df is None:
            # Fallback: try yfinance with "-USD" convention
            yf_sym = symbol.replace("USDT", "-USD")
            df = _fetch_yfinance(yf_sym, timeframe)
    else:
        df = _fetch_yfinance(symbol, timeframe)

    if df is not None and not df.empty:
        _cache[key] = (now, df)

    return df


def get_numpy_arrays(
    symbol: str, timeframe: str
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convenience wrapper that returns (high, low, close) as float64 NumPy arrays,
    or None if data cannot be fetched.
    """
    df = fetch_ohlcv(symbol, timeframe)
    if df is None or len(df) < 50:
        return None
    return (
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
    )


def clear_cache() -> None:
    """Clear the in-memory data cache."""
    _cache.clear()
