"""Tests for simulator engine and objective helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, OptimizationResult, SimulationRun
from app.optimizer import compute_expectancy_r, compute_profit_factor, run_optimization
from app.simulator import apply_simulation_results, run_simulation


def _synthetic_df(n: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base = 100 + np.sin(np.linspace(0, 16 * np.pi, n)) * 3
    close = base + np.sin(np.linspace(0, 4 * np.pi, n))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    volume = np.full(n, 1000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _setup_db(tmp_path, monkeypatch):
    from app import database

    db_path = tmp_path / "sim_test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)

    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(database, "_session_factory", factory)
    monkeypatch.setattr(database, "SessionLocal", factory)
    monkeypatch.setattr(database, "_db_available", True)
    return factory


def test_simulation_completes_and_stores_results(tmp_path, monkeypatch):
    factory = _setup_db(tmp_path, monkeypatch)

    def _fetch(symbol, timeframe, start, end):
        if symbol == "MISS":
            return None
        return _synthetic_df()

    monkeypatch.setattr("app.simulator.fetch_ohlcv_range", _fetch)

    session = factory()
    run = SimulationRun(
        status="queued",
        symbols=json.dumps(["AAPL", "MSFT", "MISS"]),
        timeframe="1h",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 20, tzinfo=timezone.utc),
        objective="risk_adjusted",
        n_trials=2,
        swept_params=json.dumps(["sl_mult", "use_frsi", "rsi_length"]),
        locked_params=json.dumps({"min_trades": 1}),
        progress_total=6,
        results=json.dumps({}),
    )
    session.add(run)
    session.commit()
    run_id = run.id
    session.close()

    run_simulation(run_id)

    session = factory()
    saved = session.query(SimulationRun).filter(SimulationRun.id == run_id).first()
    assert saved is not None
    assert saved.status == "completed"
    assert saved.progress_current == saved.progress_total

    results = json.loads(saved.results)
    assert "AAPL" in results and "MSFT" in results and "MISS" in results
    assert "best_params" in results["AAPL"]
    assert "in_sample" in results["AAPL"]
    assert "out_of_sample" in results["AAPL"]
    assert "walk_forward" in results["AAPL"]
    assert "error" in results["MISS"]
    session.close()


def test_simulation_cancelled_when_requested(tmp_path, monkeypatch):
    factory = _setup_db(tmp_path, monkeypatch)
    monkeypatch.setattr("app.simulator.fetch_ohlcv_range", lambda *args, **kwargs: _synthetic_df())

    session = factory()
    run = SimulationRun(
        status="queued",
        symbols=json.dumps(["AAPL", "MSFT"]),
        timeframe="1h",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 10, tzinfo=timezone.utc),
        objective="risk_adjusted",
        n_trials=2,
        swept_params=json.dumps(["sl_mult"]),
        locked_params=json.dumps({"min_trades": 1}),
        cancel_requested=True,
        results=json.dumps({}),
    )
    session.add(run)
    session.commit()
    run_id = run.id
    session.close()

    run_simulation(run_id)

    session = factory()
    saved = session.query(SimulationRun).filter(SimulationRun.id == run_id).first()
    assert saved is not None
    assert saved.status == "cancelled"
    session.close()


def test_apply_simulation_results_creates_current_rows(tmp_path, monkeypatch):
    factory = _setup_db(tmp_path, monkeypatch)

    session = factory()
    run = SimulationRun(
        status="completed",
        symbols=json.dumps(["AAPL"]),
        timeframe="1h",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 10, tzinfo=timezone.utc),
        objective="risk_adjusted",
        n_trials=1,
        swept_params="[]",
        locked_params="{}",
        results=json.dumps(
            {
                "AAPL": {
                    "best_params": {"sl_mult": 1.4, "tp1_rr": 0.9, "use_frsi": False},
                    "in_sample": {"signals": 12, "win_rate": 0.55, "tp2_rate": 0.3, "sl_rate": 0.2},
                    "walk_forward": {"score": 45.0, "consistency": 0.7},
                }
            }
        ),
    )
    session.add(run)
    session.commit()
    run_id = run.id
    session.close()

    result = apply_simulation_results(run_id)
    assert result["applied"] == [{"symbol": "AAPL", "timeframe": "1h"}]

    session = factory()
    row = session.query(OptimizationResult).filter(OptimizationResult.symbol == "AAPL", OptimizationResult.timeframe == "1h", OptimizationResult.is_current == True).first()
    assert row is not None
    assert row.sl_mult == 1.4
    session.close()


def test_objective_helpers_and_min_trade_pruning():
    params = {"tp1_rr": 0.8, "tp2_rr": 1.5}
    result = {
        "trades": [
            {"highest_tp_hit": 2, "outcome": "tp2_hit"},
            {"highest_tp_hit": 0, "outcome": "sl_hit"},
            {"highest_tp_hit": 1, "outcome": "tp1_hit"},
        ]
    }
    expectancy = compute_expectancy_r(result, params)
    profit_factor = compute_profit_factor(result, params)
    assert expectancy > 0
    assert profit_factor > 1

    arr = np.array([100.0, 101.0, 102.0, 103.0, 102.5, 102.2])
    opt = run_optimization(arr + 1.0, arr - 1.0, arr, np.ones_like(arr), timeframe="1h", n_trials=2, min_trades=999)
    assert opt["n_trials_completed"] == 0
