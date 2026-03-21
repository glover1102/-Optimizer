"""Tests for the data fetcher (mocked to avoid network calls)."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from app.data_fetcher import fetch_ohlcv, get_numpy_arrays, clear_cache, _is_crypto, _cache_key


def _sample_df(n: int = 100) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for mocking."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    data = {
        "open": np.random.uniform(100, 110, n),
        "high": np.random.uniform(110, 120, n),
        "low": np.random.uniform(90, 100, n),
        "close": np.random.uniform(100, 110, n),
        "volume": np.random.uniform(1000, 5000, n),
    }
    return pd.DataFrame(data, index=idx)


class TestIsCrypto:
    def test_btcusdt_is_crypto(self):
        assert _is_crypto("BTCUSDT") is True

    def test_ethusdt_is_crypto(self):
        assert _is_crypto("ETHUSDT") is True

    def test_eurusd_not_crypto(self):
        assert _is_crypto("EURUSD=X") is False

    def test_aapl_not_crypto(self):
        assert _is_crypto("AAPL") is False

    def test_futures_not_crypto(self):
        assert _is_crypto("MNQ=F") is False


class TestCacheKey:
    def test_cache_key_format(self):
        key = _cache_key("BTCUSDT", "1h")
        assert key == "BTCUSDT:1h"


class TestFetchOHLCV:
    def test_crypto_uses_ccxt(self):
        clear_cache()
        df = _sample_df(200)
        with patch("app.data_fetcher._fetch_ccxt", return_value=df) as mock_ccxt:
            result = fetch_ohlcv("BTCUSDT", "1h")
            mock_ccxt.assert_called_once_with("BTCUSDT", "1h")
            assert result is not None

    def test_forex_uses_yfinance(self):
        clear_cache()
        df = _sample_df(200)
        with patch("app.data_fetcher._fetch_yfinance", return_value=df) as mock_yf:
            result = fetch_ohlcv("EURUSD=X", "1h")
            mock_yf.assert_called_once_with("EURUSD=X", "1h")
            assert result is not None

    def test_stocks_use_yfinance(self):
        clear_cache()
        df = _sample_df(200)
        with patch("app.data_fetcher._fetch_yfinance", return_value=df) as mock_yf:
            result = fetch_ohlcv("AAPL", "1d")
            mock_yf.assert_called_once()
            assert result is not None

    def test_futures_use_yfinance(self):
        clear_cache()
        df = _sample_df(200)
        with patch("app.data_fetcher._fetch_yfinance", return_value=df) as mock_yf:
            result = fetch_ohlcv("MNQ=F", "1h")
            mock_yf.assert_called_once()
            assert result is not None

    def test_cache_hit_avoids_refetch(self):
        clear_cache()
        df = _sample_df(200)
        with patch("app.data_fetcher._fetch_yfinance", return_value=df) as mock_yf:
            fetch_ohlcv("AAPL", "1d")
            fetch_ohlcv("AAPL", "1d")
            assert mock_yf.call_count == 1  # Second call should be from cache

    def test_returns_none_on_failure(self):
        clear_cache()
        with patch("app.data_fetcher._fetch_yfinance", return_value=None):
            result = fetch_ohlcv("INVALID_SYMBOL", "1h")
            assert result is None

    def test_ccxt_fallback_to_yfinance_on_failure(self):
        clear_cache()
        df = _sample_df(200)
        with patch("app.data_fetcher._fetch_ccxt", return_value=None), \
             patch("app.data_fetcher._fetch_yfinance", return_value=df) as mock_yf:
            result = fetch_ohlcv("BTCUSDT", "1h")
            # Should fall back to yfinance
            mock_yf.assert_called_once()
            assert result is not None


class TestGetNumpyArrays:
    def test_returns_tuple_of_arrays(self):
        clear_cache()
        df = _sample_df(100)
        with patch("app.data_fetcher.fetch_ohlcv", return_value=df):
            result = get_numpy_arrays("AAPL", "1d")
            assert result is not None
            high, low, close = result
            assert isinstance(high, np.ndarray)
            assert isinstance(low, np.ndarray)
            assert isinstance(close, np.ndarray)

    def test_returns_none_when_insufficient_data(self):
        clear_cache()
        df = _sample_df(10)  # Too few bars
        with patch("app.data_fetcher.fetch_ohlcv", return_value=df):
            result = get_numpy_arrays("AAPL", "1d")
            assert result is None

    def test_returns_none_when_fetch_fails(self):
        clear_cache()
        with patch("app.data_fetcher.fetch_ohlcv", return_value=None):
            result = get_numpy_arrays("INVALID", "1h")
            assert result is None
