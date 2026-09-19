"""Tests for the KLS+MoM Optuna optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import DEFAULT_SIGNAL_PARAMS, PARAM_RANGES
from app.optimizer import run_optimization


def _optimization_data(days: int = 10):
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    timestamps: list[pd.Timestamp] = []
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    for day in range(days):
        base = start + pd.Timedelta(days=day)
        pattern = [
            (10.0, 10.3, 9.8, 10.0, 100.0),
            (10.0, 10.4, 9.9, 10.1, 120.0),
            (10.1, 11.4, 10.0, 11.2, 280.0),
            (11.2, 12.2, 11.0, 12.0, 240.0),
            (12.0, 12.1, 11.7, 11.9, 150.0),
        ]
        for offset, row in enumerate(pattern):
            o, h, l, c, v = row
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(c)
            volumes.append(v)
            timestamps.append(base + pd.Timedelta(hours=offset))
    return (
        np.array(opens, dtype=float),
        np.array(highs, dtype=float),
        np.array(lows, dtype=float),
        np.array(closes, dtype=float),
        np.array(volumes, dtype=float),
        pd.DatetimeIndex(timestamps, tz="UTC"),
    )


class TestRunOptimization:
    def test_returns_expected_keys(self):
        open_, h, l, c, v, ts = _optimization_data()
        result = run_optimization(h, l, c, v, open_=open_, timestamps=ts, timeframe="1h", n_trials=4)
        assert {"best_params", "best_value", "best_backtest", "top_trials", "n_trials_completed"} == set(result.keys())

    def test_best_params_include_kls_fields(self):
        open_, h, l, c, v, ts = _optimization_data()
        result = run_optimization(h, l, c, v, open_=open_, timestamps=ts, timeframe="1h", n_trials=4)
        param_keys = {"atr_length", "sl_mult", "tp1_rr", "tp2_rr", "tp3_rr", "tp4_rr", "resolve_mode", "filt_mode"}
        assert param_keys.issubset(set(result["best_params"].keys()))

    def test_best_params_respect_ranges(self):
        open_, h, l, c, v, ts = _optimization_data()
        result = run_optimization(h, l, c, v, open_=open_, timestamps=ts, timeframe="1h", n_trials=4)
        p = result["best_params"]
        assert PARAM_RANGES["atr_length"][0] <= p["atr_length"] <= PARAM_RANGES["atr_length"][1]
        assert PARAM_RANGES["sl_mult"][0] <= p["sl_mult"] <= PARAM_RANGES["sl_mult"][1]
        assert PARAM_RANGES["tp2_rr"][0] <= p["tp2_rr"] <= PARAM_RANGES["tp2_rr"][1]
        assert p["resolve_mode"] in PARAM_RANGES["categorical"]["resolve_mode"]
        assert p["filt_mode"] in PARAM_RANGES["categorical"]["filt_mode"]

    def test_top_trials_list(self):
        open_, h, l, c, v, ts = _optimization_data()
        result = run_optimization(h, l, c, v, open_=open_, timestamps=ts, timeframe="1h", n_trials=5)
        assert isinstance(result["top_trials"], list)
        assert len(result["top_trials"]) <= 10

    def test_default_fallback_shape(self):
        h = np.ones(10)
        l = np.ones(10) * 0.9
        c = np.ones(10) * 0.95
        result = run_optimization(h, l, c, np.ones(10), timeframe="1h", n_trials=2)
        assert result["best_params"] == DEFAULT_SIGNAL_PARAMS
        assert result["best_backtest"]["total_signals"] == 0

    def test_objectives_execute(self):
        open_, h, l, c, v, ts = _optimization_data()
        win = run_optimization(h, l, c, v, open_=open_, timestamps=ts, timeframe="1h", n_trials=3, objective="win_rate")
        tp2 = run_optimization(h, l, c, v, open_=open_, timestamps=ts, timeframe="1h", n_trials=3, objective="tp2_rate")
        assert win["best_backtest"]["win_rate"] >= 0.0
        assert tp2["best_backtest"]["tp2_rate"] >= 0.0
