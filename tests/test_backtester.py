"""Tests for the KLS+MoM backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtester import _atr, _empty_result, run_backtest
from app.config import DEFAULT_SIGNAL_PARAMS
from app.kls_mom import simulate_kls_strategy


def _timestamps(n: int, start: str = "2024-01-01T00:00:00Z", freq: str = "1h"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


def _session_fixture():
    open_ = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 104.5], dtype=float)
    high = np.array([101.0, 102.0, 103.0, 104.0, 105.0, 106.0], dtype=float)
    low = np.array([99.0, 100.0, 101.0, 102.0, 103.0, 104.0], dtype=float)
    close = np.array([100.5, 101.5, 102.5, 103.5, 104.5, 105.5], dtype=float)
    volume = np.full(len(close), 100.0)
    ts = pd.DatetimeIndex([
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T01:00:00Z"),
        pd.Timestamp("2024-01-01T02:00:00Z"),
        pd.Timestamp("2024-01-02T00:00:00Z"),
        pd.Timestamp("2024-01-02T01:00:00Z"),
        pd.Timestamp("2024-01-02T02:00:00Z"),
    ])
    return open_, high, low, close, volume, ts


def _or_long_fixture():
    open_ = np.array([10.0, 10.0, 10.2, 11.2, 11.7], dtype=float)
    high = np.array([10.4, 10.5, 11.6, 12.1, 12.0], dtype=float)
    low = np.array([9.8, 9.9, 10.1, 11.0, 11.4], dtype=float)
    close = np.array([10.0, 10.1, 11.2, 11.8, 11.9], dtype=float)
    volume = np.array([100.0, 110.0, 300.0, 180.0, 150.0], dtype=float)
    ts = _timestamps(len(close))
    return open_, high, low, close, volume, ts


def _repeating_sessions(days: int = 8):
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    stamps: list[pd.Timestamp] = []
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    for day in range(days):
        base = start + pd.Timedelta(days=day)
        pattern = [
            (10.0, 10.3, 9.8, 10.0, 100.0),
            (10.0, 10.4, 9.9, 10.1, 120.0),
            (10.1, 11.3, 10.1, 11.1, 300.0),
            (11.1, 12.1, 11.0, 11.9, 220.0),
            (11.9, 12.0, 11.6, 11.8, 140.0),
        ]
        for offset, row in enumerate(pattern):
            o, h, l, c, v = row
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(c)
            volumes.append(v)
            stamps.append(base + pd.Timedelta(hours=offset))
    return (
        np.array(opens, dtype=float),
        np.array(highs, dtype=float),
        np.array(lows, dtype=float),
        np.array(closes, dtype=float),
        np.array(volumes, dtype=float),
        pd.DatetimeIndex(stamps, tz="UTC"),
    )


class TestATR:
    def test_atr_length(self):
        _, h, l, c, _, _ = _repeating_sessions(5)
        atr = _atr(h, l, c, 14)
        assert len(atr) == len(c)

    def test_atr_nan_prefix(self):
        _, h, l, c, _, _ = _repeating_sessions(5)
        atr = _atr(h, l, c, 5)
        assert np.all(np.isnan(atr[:4]))
        assert not np.isnan(atr[4])


class TestKLSBacktest:
    def test_session_key_levels_use_previous_day_values(self):
        open_, high, low, close, volume, ts = _session_fixture()
        params = {**DEFAULT_SIGNAL_PARAMS, "or_as_trigger": False, "or_ext_as_trigger": False}
        sim = simulate_kls_strategy(high, low, close, volume, open_=open_, timestamps=ts, params=params, include_debug=True)
        levels = sim["debug"]["levels"][3]
        assert levels["PDO"] == 100.0
        assert levels["PDH"] == 103.0
        assert levels["PDL"] == 99.0
        assert levels["PDC"] == 102.5

    def test_or_break_produces_long_entry(self):
        open_, high, low, close, volume, ts = _or_long_fixture()
        params = {
            **DEFAULT_SIGNAL_PARAMS,
            "enable_pdl": False,
            "enable_pdo": False,
            "enable_pdc": False,
            "enable_pdh": False,
            "use_structure_15m": False,
            "use_fmom": False,
            "use_fvol": False,
            "use_frsi": False,
            "use_fmacd": False,
            "use_fvwap": False,
            "or1_enabled": True,
            "or1_minutes": 120,
            "or_as_trigger": True,
            "or_ext_as_trigger": False,
            "atr_length": 2,
        }
        sim = simulate_kls_strategy(high, low, close, volume, open_=open_, timestamps=ts, params=params)
        trade = sim["current_trade"]
        assert trade is not None
        assert trade["direction"] == 1
        assert trade["trigger_type"] == "OR"
        assert trade["trigger_name"] == "OR1_HIGH"

    def test_first_touch_both_hit_resolves_conservatively_to_stop(self):
        open_, high, low, close, volume, ts = _or_long_fixture()
        high[3] = 11.9
        high[-1] = 13.0
        low[-1] = 9.0
        close[-1] = 11.7
        params = {
            **DEFAULT_SIGNAL_PARAMS,
            "enable_pdl": False,
            "enable_pdo": False,
            "enable_pdc": False,
            "enable_pdh": False,
            "use_structure_15m": False,
            "use_fmom": False,
            "use_fvol": False,
            "use_frsi": False,
            "use_fmacd": False,
            "use_fvwap": False,
            "or1_enabled": True,
            "or1_minutes": 120,
            "or_as_trigger": True,
            "or_ext_as_trigger": False,
            "clear_on_tp": True,
            "clear_tp_sel": "TP1",
            "resolve_mode": "First touch",
            "atr_length": 2,
            "sl_mult": 1.0,
        }
        result = run_backtest(high, low, close, volume, open_=open_, timestamps=ts, **params)
        assert result["total_signals"] == 1
        assert result["sl_hits"] == 1
        assert result["wins"] == 0

    def test_close_mode_counts_tp2_hit(self):
        open_, high, low, close, volume, ts = _or_long_fixture()
        close[-1] = 13.0
        high[-1] = 13.2
        params = {
            **DEFAULT_SIGNAL_PARAMS,
            "enable_pdl": False,
            "enable_pdo": False,
            "enable_pdc": False,
            "enable_pdh": False,
            "use_structure_15m": False,
            "use_fmom": False,
            "use_fvol": False,
            "use_frsi": False,
            "use_fmacd": False,
            "use_fvwap": False,
            "or1_enabled": True,
            "or1_minutes": 120,
            "or_as_trigger": True,
            "or_ext_as_trigger": False,
            "clear_on_tp": True,
            "clear_tp_sel": "TP2",
            "resolve_mode": "Close",
            "atr_length": 2,
            "sl_mult": 1.0,
        }
        result = run_backtest(high, low, close, volume, open_=open_, timestamps=ts, **params)
        assert result["wins"] == 1
        assert result["tp2_hits"] == 1
        assert result["sl_hits"] == 0

    def test_breakeven_move_preserves_win_without_stop_loss_hit(self):
        open_, high, low, close, volume, ts = _or_long_fixture()
        high = np.append(high, [12.4, 11.9])
        low = np.append(low, [11.2, 11.1])
        close = np.append(close, [12.0, 11.2])
        open_ = np.append(open_, [11.9, 12.0])
        volume = np.append(volume, [150.0, 130.0])
        ts = ts.append(pd.DatetimeIndex([ts[-1] + pd.Timedelta(hours=1), ts[-1] + pd.Timedelta(hours=2)]))
        params = {
            **DEFAULT_SIGNAL_PARAMS,
            "enable_pdl": False,
            "enable_pdo": False,
            "enable_pdc": False,
            "enable_pdh": False,
            "use_structure_15m": False,
            "use_fmom": False,
            "use_fvol": False,
            "use_frsi": False,
            "use_fmacd": False,
            "use_fvwap": False,
            "or1_enabled": True,
            "or1_minutes": 120,
            "or_as_trigger": True,
            "or_ext_as_trigger": False,
            "resolve_mode": "First touch",
            "atr_length": 2,
            "sl_mult": 1.0,
            "be_after_tp": "TP1",
            "clear_on_tp": False,
        }
        result = run_backtest(high, low, close, volume, open_=open_, timestamps=ts, **params)
        assert result["wins"] == 1
        assert result["tp1_hits"] == 1
        assert result["sl_hits"] == 0
        assert result["trades"][0]["outcome"] in {"breakeven", "tp4_hit", "tp3_hit", "tp2_hit", "tp1_hit"}

    def test_volume_filter_blocks_entry(self):
        open_, high, low, close, volume, ts = _or_long_fixture()
        volume[2] = 50.0
        params = {
            **DEFAULT_SIGNAL_PARAMS,
            "enable_pdl": False,
            "enable_pdo": False,
            "enable_pdc": False,
            "enable_pdh": False,
            "use_structure_15m": False,
            "use_fmom": False,
            "use_fvol": True,
            "fvol_len": 2,
            "fvol_mult": 2.0,
            "use_frsi": False,
            "use_fmacd": False,
            "use_fvwap": False,
            "or1_enabled": True,
            "or1_minutes": 120,
            "or_as_trigger": True,
            "or_ext_as_trigger": False,
            "atr_length": 2,
        }
        sim = simulate_kls_strategy(high, low, close, volume, open_=open_, timestamps=ts, params=params)
        assert sim["current_trade"] is None
        assert sim["trades"] == []

    def test_run_backtest_preserves_contract_keys(self):
        open_, high, low, close, volume, ts = _repeating_sessions(6)
        result = run_backtest(high, low, close, volume, open_=open_, timestamps=ts, **DEFAULT_SIGNAL_PARAMS)
        expected = {
            "total_signals", "wins", "tp1_hits", "tp2_hits", "tp3_hits", "tp4_hits", "sl_hits",
            "win_rate", "tp1_rate", "tp2_rate", "tp3_rate", "tp4_rate", "sl_rate", "trades",
        }
        assert expected == set(result.keys())
        assert result["total_signals"] >= result["wins"]

    def test_insufficient_data_returns_empty(self):
        h = np.ones(4)
        l = np.ones(4) * 0.9
        c = np.ones(4) * 0.95
        assert run_backtest(h, l, c) == _empty_result()
