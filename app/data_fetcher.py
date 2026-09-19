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

# Conservative cap for range fetches to avoid unbounded downloads.
_MAX_RANGE_BARS = {
    "5m": 20_000,
    "15m": 20_000,
    "1h": 10_000,
    "4h": 8_000,
    "1d": 5_000,
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


def _fetch_yfinance_range(symbol: str, timeframe: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf  # lazy import

        yf_tf = _YFINANCE_TF_MAP.get(timeframe, "1d")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=yf_tf, auto_adjust=True)
        if df.empty:
            return None
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.dropna()
        if timeframe == "4h":
            df = df.resample("4h").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
        return df
    except Exception as exc:
        logger.error("yfinance range error for %s %s: %s", symbol, timeframe, exc)
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


def _fetch_ccxt_range(symbol: str, timeframe: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    try:
        import ccxt  # lazy import
        from app.config import BINANCE_API_KEY, BINANCE_SECRET

        exchange = ccxt.binance({
            "apiKey": BINANCE_API_KEY or None,
            "secret": BINANCE_SECRET or None,
            "enableRateLimit": True,
        })
        ccxt_tf = _CCXT_TF_MAP.get(timeframe, "1h")
        ccxt_symbol = symbol[:-4] + "/" + symbol[-4:] if symbol.endswith("USDT") else symbol
        max_bars = _MAX_RANGE_BARS.get(timeframe, 10_000)

        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_rows: list[list[float]] = []
        last_ts = since_ms
        limit = 1000

        while last_ts < end_ms and len(all_rows) < max_bars:
            chunk = exchange.fetch_ohlcv(ccxt_symbol, ccxt_tf, since=last_ts, limit=limit)
            if not chunk:
                break
            all_rows.extend(chunk)
            newest = int(chunk[-1][0])
            if newest <= last_ts:
                break
            last_ts = newest + 1
            if len(chunk) < limit:
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        df = df[~df.index.duplicated(keep="last")].sort_index().dropna()
        if len(df) > max_bars:
            df = df.iloc[-max_bars:]
        return df
    except Exception as exc:
        logger.error("ccxt range error for %s %s: %s", symbol, timeframe, exc)
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


def fetch_ohlcv_range(symbol: str, timeframe: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a bounded datetime range (UTC).

    Returns the same DataFrame shape as fetch_ohlcv:
    columns [open, high, low, close, volume] with UTC DatetimeIndex.
    """
    start_utc = pd.Timestamp(start)
    if start_utc.tzinfo is None:
        start_utc = start_utc.tz_localize("UTC")
    else:
        start_utc = start_utc.tz_convert("UTC")
    end_utc = pd.Timestamp(end)
    if end_utc.tzinfo is None:
        end_utc = end_utc.tz_localize("UTC")
    else:
        end_utc = end_utc.tz_convert("UTC")
    if end_utc <= start_utc:
        return None

    if _is_crypto(symbol):
        df = _fetch_ccxt_range(symbol, timeframe, start_utc.to_pydatetime(), end_utc.to_pydatetime())
        if df is None or df.empty:
            yf_sym = symbol.replace("USDT", "-USD")
            df = _fetch_yfinance_range(yf_sym, timeframe, start_utc.to_pydatetime(), end_utc.to_pydatetime())
    else:
        df = _fetch_yfinance_range(symbol, timeframe, start_utc.to_pydatetime(), end_utc.to_pydatetime())

    if df is None or df.empty:
        return None
    max_bars = _MAX_RANGE_BARS.get(timeframe, 10_000)
    if len(df) > max_bars:
        df = df.iloc[-max_bars:]
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


def get_numpy_arrays_with_volume(
    symbol: str, timeframe: str
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (high, low, close, volume) as float64 NumPy arrays,
    or None if data cannot be fetched.
    """
    df = fetch_ohlcv(symbol, timeframe)
    if df is None or len(df) < 50:
        return None
    return (
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        df["volume"].to_numpy(dtype=np.float64),
    )


def clear_cache() -> None:
    """Clear the in-memory data cache."""
    _cache.clear()
