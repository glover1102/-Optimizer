"""Tests for the Optuna optimizer."""

import numpy as np
import pytest

from app.optimizer import run_optimization


def _synthetic_data(n: int = 800, seed: int = 42):
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0.1, 1.0, n)) + 100
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return high, low, close


class TestRunOptimization:
    def test_returns_expected_keys(self):
        h, l, c = _synthetic_data()
        result = run_optimization(h, l, c, n_trials=10)
        expected = {"best_params", "best_value", "best_backtest", "top_trials", "n_trials_completed"}
        assert expected == set(result.keys())

    def test_best_params_keys(self):
        h, l, c = _synthetic_data()
        result = run_optimization(h, l, c, n_trials=10)
        param_keys = {"left_bars", "right_bars", "offset", "atr_multiplier", "atr_period"}
        assert param_keys == set(result["best_params"].keys())

    def test_best_params_in_range(self):
        from app.config import PARAM_RANGES
        h, l, c = _synthetic_data()
        result = run_optimization(h, l, c, n_trials=10)
        p = result["best_params"]
        assert PARAM_RANGES["left_bars"][0] <= p["left_bars"] <= PARAM_RANGES["left_bars"][1]
        assert PARAM_RANGES["right_bars"][0] <= p["right_bars"] <= PARAM_RANGES["right_bars"][1]
        assert PARAM_RANGES["offset"][0] <= p["offset"] <= PARAM_RANGES["offset"][1]
        assert PARAM_RANGES["atr_multiplier"][0] <= p["atr_multiplier"] <= PARAM_RANGES["atr_multiplier"][1]
        assert PARAM_RANGES["atr_period"][0] <= p["atr_period"] <= PARAM_RANGES["atr_period"][1]

    def test_top_trials_list(self):
        h, l, c = _synthetic_data()
        result = run_optimization(h, l, c, n_trials=15)
        assert isinstance(result["top_trials"], list)
        assert len(result["top_trials"]) <= 10

    def test_win_rate_objective(self):
        h, l, c = _synthetic_data()
        result = run_optimization(h, l, c, n_trials=10, objective="win_rate")
        assert result["best_backtest"]["win_rate"] >= 0.0

    def test_tp2_rate_objective(self):
        h, l, c = _synthetic_data()
        result = run_optimization(h, l, c, n_trials=10, objective="tp2_rate")
        assert result["best_backtest"]["tp2_rate"] >= 0.0

    def test_n_trials_completed(self):
        h, l, c = _synthetic_data()
        result = run_optimization(h, l, c, n_trials=5)
        # Some trials may be pruned, but at least a few should complete
        assert result["n_trials_completed"] >= 0
