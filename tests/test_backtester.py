"""Tests for the vectorized QTAlgo backtester."""

import numpy as np
import pytest

from app.backtester import run_backtest, _atr, _empty_result


def _synthetic_trending(n: int = 500, seed: int = 42) -> tuple:
    """Generate synthetic trending price data."""
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0.1, 1.0, n)) + 100
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return high, low, close


def _synthetic_ranging(n: int = 500, seed: int = 7) -> tuple:
    """Generate synthetic mean-reverting price data."""
    rng = np.random.default_rng(seed)
    close = np.sin(np.linspace(0, 8 * np.pi, n)) * 5 + 100 + rng.normal(0, 0.3, n)
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    return high, low, close


class TestATR:
    def test_atr_length(self):
        h, l, c = _synthetic_trending(200)
        atr = _atr(h, l, c, period=14)
        assert len(atr) == 200

    def test_atr_nan_prefix(self):
        h, l, c = _synthetic_trending(200)
        atr = _atr(h, l, c, period=14)
        # First (period-1) values should be NaN
        assert np.all(np.isnan(atr[:13]))
        assert not np.isnan(atr[13])

    def test_atr_positive(self):
        h, l, c = _synthetic_trending(200)
        atr = _atr(h, l, c, period=14)
        valid = atr[~np.isnan(atr)]
        assert np.all(valid > 0)

    def test_atr_short_series(self):
        h = np.array([1.0, 2.0, 3.0])
        l = np.array([0.5, 1.5, 2.5])
        c = np.array([0.8, 1.8, 2.8])
        atr = _atr(h, l, c, period=14)
        assert len(atr) == 3


class TestRunBacktest:
    def test_returns_dict_keys(self):
        h, l, c = _synthetic_trending(300)
        result = run_backtest(h, l, c)
        expected_keys = {
            "total_signals", "wins", "tp2_hits", "tp3_hits", "sl_hits",
            "win_rate", "tp2_rate", "tp3_rate", "sl_rate",
        }
        assert expected_keys == set(result.keys())

    def test_rates_between_0_and_1(self):
        h, l, c = _synthetic_trending(500)
        result = run_backtest(h, l, c)
        for key in ("win_rate", "tp2_rate", "tp3_rate", "sl_rate"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"

    def test_signals_non_negative(self):
        h, l, c = _synthetic_trending(500)
        result = run_backtest(h, l, c)
        assert result["total_signals"] >= 0

    def test_wins_lte_total_signals(self):
        h, l, c = _synthetic_trending(500)
        result = run_backtest(h, l, c)
        assert result["wins"] <= result["total_signals"]

    def test_tp2_lte_wins(self):
        h, l, c = _synthetic_trending(500)
        result = run_backtest(h, l, c)
        assert result["tp2_hits"] <= result["wins"]

    def test_tp3_lte_tp2(self):
        h, l, c = _synthetic_trending(500)
        result = run_backtest(h, l, c)
        assert result["tp3_hits"] <= result["tp2_hits"]

    def test_insufficient_data_returns_empty(self):
        h = np.ones(10)
        l = np.ones(10) * 0.9
        c = np.ones(10) * 0.95
        result = run_backtest(h, l, c)
        assert result == _empty_result()

    def test_custom_params(self):
        h, l, c = _synthetic_trending(600)
        result = run_backtest(h, l, c, left_bars=5, right_bars=5, offset=1.0, atr_multiplier=1.0, atr_period=10)
        assert isinstance(result["win_rate"], float)

    def test_ranging_data(self):
        h, l, c = _synthetic_ranging(500)
        result = run_backtest(h, l, c)
        assert isinstance(result["total_signals"], int)

    def test_large_params(self):
        h, l, c = _synthetic_trending(1000)
        result = run_backtest(h, l, c, left_bars=30, right_bars=30, offset=4.0, atr_period=30)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_small_params(self):
        h, l, c = _synthetic_trending(600)
        result = run_backtest(h, l, c, left_bars=3, right_bars=3, offset=0.5, atr_period=5)
        assert 0.0 <= result["win_rate"] <= 1.0


class TestEmptyResult:
    def test_empty_result_structure(self):
        result = _empty_result()
        assert result["total_signals"] == 0
        assert result["win_rate"] == 0.0
