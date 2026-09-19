from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import optuna
from sqlalchemy import update

from app.backtester import run_backtest
from app.config import DEFAULT_SIGNAL_PARAMS, PARAM_RANGES
from app.data_fetcher import fetch_ohlcv_range
from app.optimizer import run_optimization
from app.walk_forward import run_walk_forward

logger = logging.getLogger(__name__)

_FILTER_TOGGLES = [
    "use_structure_15m",
    "use_structure_1h",
    "use_structure_4h",
    "use_fmom",
    "use_fvol",
    "use_frsi",
    "use_fmacd",
    "use_fvwap",
    "use_fatr",
]

SIM_SWEEP_GENES: list[dict[str, Any]] = [
    {"key": "sl_mult", "label": "SL ATR Mult", "type": "float", "range": [1.0, 3.0], "step": 0.05},
    {"key": "tp1_rr", "label": "TP1 R:R", "type": "float", "range": [0.5, 1.5], "step": 0.05},
    {"key": "tp2_rr", "label": "TP2 R:R", "type": "float", "range": [1.0, 2.5], "step": 0.05},
    {"key": "tp3_rr", "label": "TP3 R:R", "type": "float", "range": [1.5, 3.5], "step": 0.05},
    {"key": "tp4_rr", "label": "TP4 R:R", "type": "float", "range": [2.0, 4.5], "step": 0.05},
    {"key": "be_after_tp", "label": "Move BE After", "type": "categorical", "choices": ["Off", "TP1", "TP2", "TP3", "TP4"]},
    {"key": "resolve_mode", "label": "Resolve Mode", "type": "categorical", "choices": ["Close", "First touch"]},
    {"key": "entry_at_level", "label": "Entry at level", "type": "bool"},
    {"key": "re_arm", "label": "Re-arm", "type": "bool"},
    {"key": "runner_tgt", "label": "Runner target", "type": "categorical", "choices": ["TP4", "Last enabled TP"]},
    {"key": "clear_on_tp", "label": "Clear on TP", "type": "bool"},
    {"key": "clear_tp_sel", "label": "Clear TP", "type": "categorical", "choices": ["TP1", "TP2", "TP3", "TP4"]},
    {"key": "enable_pdl", "label": "PDL", "type": "bool"},
    {"key": "enable_pdo", "label": "PDO", "type": "bool"},
    {"key": "enable_pdc", "label": "PDC", "type": "bool"},
    {"key": "enable_pdh", "label": "PDH", "type": "bool"},
    {"key": "enable_pwh", "label": "PWH", "type": "bool"},
    {"key": "enable_pwl", "label": "PWL", "type": "bool"},
    {"key": "enable_open", "label": "OPEN", "type": "bool"},
    {"key": "or_as_trigger", "label": "OR as trigger", "type": "bool"},
    {"key": "or_ext_as_trigger", "label": "OR Ext as trigger", "type": "bool"},
    {"key": "or1_enabled", "label": "OR1 enabled", "type": "bool"},
    {"key": "or1_minutes", "label": "OR1 minutes", "type": "int", "range": [5, 240], "step": 5},
    {"key": "or2_enabled", "label": "OR2 enabled", "type": "bool"},
    {"key": "or2_minutes", "label": "OR2 minutes", "type": "int", "range": [60, 720], "step": 5},
    {"key": "or3_enabled", "label": "OR3 enabled", "type": "bool"},
    {"key": "or3_minutes", "label": "OR3 minutes", "type": "int", "range": [120, 960], "step": 5},
    {"key": "or4_enabled", "label": "OR4 enabled", "type": "bool"},
    {"key": "or4_minutes", "label": "OR4 minutes", "type": "int", "range": [240, 1440], "step": 5},
    {"key": "or_ext_0_5_enabled", "label": "OR ext 0.5", "type": "bool"},
    {"key": "or_ext_1_0_enabled", "label": "OR ext 1.0", "type": "bool"},
    {"key": "or_ext_1_5_enabled", "label": "OR ext 1.5", "type": "bool"},
    {"key": "or_ext_2_0_enabled", "label": "OR ext 2.0", "type": "bool"},
    {"key": "filt_mode", "label": "Filter mode", "type": "categorical", "choices": ["Strict", "Not against"]},
    {"key": "use_structure_15m", "label": "Use structure 15m", "type": "bool"},
    {"key": "use_structure_1h", "label": "Use structure 1h", "type": "bool"},
    {"key": "use_structure_4h", "label": "Use structure 4h", "type": "bool"},
    {"key": "use_fmom", "label": "Use EMA stack", "type": "bool"},
    {"key": "fmom_strict_fan", "label": "EMA strict fan", "type": "bool", "when": "use_fmom"},
    {"key": "ema_9_enabled", "label": "EMA 9", "type": "bool", "when": "use_fmom"},
    {"key": "ema_21_enabled", "label": "EMA 21", "type": "bool", "when": "use_fmom"},
    {"key": "ema_50_enabled", "label": "EMA 50", "type": "bool", "when": "use_fmom"},
    {"key": "ema_100_enabled", "label": "EMA 100", "type": "bool", "when": "use_fmom"},
    {"key": "ema_200_enabled", "label": "EMA 200", "type": "bool", "when": "use_fmom"},
    {"key": "ema_300_enabled", "label": "EMA 300", "type": "bool", "when": "use_fmom"},
    {"key": "use_fvol", "label": "Use volume filter", "type": "bool"},
    {"key": "fvol_len", "label": "Volume MA length", "type": "int", "range": [5, 80], "step": 1, "when": "use_fvol"},
    {"key": "fvol_mult", "label": "Volume multiplier", "type": "float", "range": [0.6, 2.0], "step": 0.05, "when": "use_fvol"},
    {"key": "fvol_ma_type", "label": "Volume MA type", "type": "categorical", "choices": ["SMA", "EMA", "WMA", "RMA"], "when": "use_fvol"},
    {"key": "use_frsi", "label": "Use RSI filter", "type": "bool"},
    {"key": "rsi_length", "label": "RSI length", "type": "int", "range": [5, 30], "step": 1, "when": "use_frsi"},
    {"key": "rsi_neutral", "label": "RSI neutral zone", "type": "float", "range": [1.0, 12.0], "step": 0.5, "when": "use_frsi"},
    {"key": "use_fmacd", "label": "Use MACD filter", "type": "bool"},
    {"key": "macd_fast", "label": "MACD fast", "type": "int", "range": [6, 20], "step": 1, "when": "use_fmacd"},
    {"key": "macd_slow", "label": "MACD slow", "type": "int", "range": [18, 50], "step": 1, "when": "use_fmacd"},
    {"key": "macd_signal", "label": "MACD signal", "type": "int", "range": [5, 20], "step": 1, "when": "use_fmacd"},
    {"key": "use_fvwap", "label": "Use VWAP filter", "type": "bool"},
    {"key": "use_fatr", "label": "Use ATR state gate", "type": "bool"},
    {"key": "fatr_baseline", "label": "ATR baseline", "type": "int", "range": [5, 80], "step": 1, "when": "use_fatr"},
    {"key": "fatr_allow", "label": "ATR allow mode", "type": "categorical", "choices": ["Only Stable", "Stable or Low", "Stable or High"], "when": "use_fatr"},
]

def _json_load(v: str | None, default: Any):
    if not v:
        return default
    try:
        return json.loads(v)
    except Exception:
        return default

def _safe_round(value: float | int | None) -> float | None:
    if value is None:
        return None
    if np.isinf(value):
        return 999999.0
    return round(float(value), 6)


def _build_base_params(locked_params: dict[str, Any]) -> dict[str, Any]:
    base = dict(DEFAULT_SIGNAL_PARAMS)
    base["filt_mode"] = "Not against"
    for key in _FILTER_TOGGLES:
        base[key] = False
    base.update(locked_params)
    return base


def _resolve_gene_value(trial: optuna.Trial, gene: dict[str, Any], swept: set[str], values: dict[str, Any], locked: dict[str, Any]):
    key = gene["key"]
    parent = gene.get("when")
    if parent and not bool(values.get(parent, locked.get(parent, DEFAULT_SIGNAL_PARAMS.get(parent)))):
        values[key] = locked.get(key, values.get(key, DEFAULT_SIGNAL_PARAMS.get(key)))
        return

    should_sweep = key in swept
    current = locked.get(key, values.get(key, DEFAULT_SIGNAL_PARAMS.get(key)))

    if not should_sweep:
        values[key] = current
        return

    gtype = gene["type"]
    if gtype == "bool":
        values[key] = trial.suggest_categorical(key, [False, True])
    elif gtype == "categorical":
        values[key] = trial.suggest_categorical(key, gene["choices"])
    elif gtype == "int":
        lo, hi = gene.get("range") or PARAM_RANGES.get(key, (1, 100))
        step = int(gene.get("step", 1))
        values[key] = trial.suggest_int(key, int(lo), int(hi), step=step)
    elif gtype == "float":
        lo, hi = gene.get("range") or PARAM_RANGES.get(key, (0.1, 10.0))
        step = gene.get("step")
        values[key] = trial.suggest_float(key, float(lo), float(hi), step=step)
    else:
        values[key] = current


def _build_suggest_fn(base_params: dict[str, Any], swept_params: set[str], locked_params: dict[str, Any]):
    def _suggest(trial: optuna.Trial) -> dict[str, Any]:
        values = dict(base_params)
        for gene in SIM_SWEEP_GENES:
            _resolve_gene_value(trial, gene, swept_params, values, locked_params)
        return values

    return _suggest


def _split_sample(df, split_ratio: float = 0.7):
    split = int(len(df) * split_ratio)
    split = max(10, min(split, len(df) - 5))
    return df.iloc[:split], df.iloc[split:]


def _serialize_symbol_result(best_params: dict[str, Any], opt_result: dict[str, Any], in_sample: dict[str, Any], out_sample: dict[str, Any], wf_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_value": _safe_round(opt_result.get("best_value")),
        "best_params": best_params,
        "in_sample": {
            "signals": in_sample.get("total_signals", 0),
            "win_rate": in_sample.get("win_rate", 0.0),
            "tp2_rate": in_sample.get("tp2_rate", 0.0),
            "sl_rate": in_sample.get("sl_rate", 0.0),
        },
        "out_of_sample": {
            "signals": out_sample.get("total_signals", 0),
            "win_rate": out_sample.get("win_rate", 0.0),
            "tp2_rate": out_sample.get("tp2_rate", 0.0),
            "sl_rate": out_sample.get("sl_rate", 0.0),
        },
        "walk_forward": {
            "score": wf_result.get("walk_forward_score", 0.0),
            "consistency": wf_result.get("consistency", 0.0),
            "avg_oos_win_rate": wf_result.get("avg_oos_win_rate", 0.0),
        },
        "top_trials": opt_result.get("top_trials", []),
    }


def run_simulation(sim_id: int) -> None:
    from app.database import get_session_factory
    from app.models import SimulationRun

    factory = get_session_factory()
    session = factory()
    run = session.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
    if run is None:
        session.close()
        return

    symbols = [s.strip().upper() for s in _json_load(run.symbols, []) if str(s).strip()]
    swept_params = set(_json_load(run.swept_params, []))
    locked_params = _json_load(run.locked_params, {})
    run_timeframe = run.timeframe
    run_start = run.start_date
    run_end = run.end_date
    run_objective = run.objective
    run_trials = int(run.n_trials)
    min_trades = int(locked_params.get("min_trades", 20))
    locked_params = {k: v for k, v in locked_params.items() if k != "min_trades"}

    run.status = "running"
    run.progress_current = 0
    run.progress_total = max(1, run_trials * max(1, len(symbols)))
    run.best_value = None
    run.results = json.dumps({})
    session.commit()
    session.close()

    all_results: dict[str, Any] = {}
    best_seen: float | None = None

    def _cancel_requested() -> bool:
        local = factory()
        try:
            current = local.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
            return bool(current and current.cancel_requested)
        finally:
            local.close()

    def _on_trial_complete(_trial: optuna.trial.FrozenTrial):
        local = factory()
        try:
            current = local.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
            if current is None:
                return
            current.progress_current = min((current.progress_current or 0) + 1, current.progress_total or 0)
            local.commit()
        finally:
            local.close()

    def _bump_progress(amount: int):
        if amount <= 0:
            return
        local = factory()
        try:
            current = local.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
            if current is None:
                return
            current.progress_current = min((current.progress_current or 0) + amount, current.progress_total or 0)
            local.commit()
        finally:
            local.close()

    try:
        base_params = _build_base_params(locked_params)
        suggest_fn = _build_suggest_fn(base_params, swept_params, locked_params)
        cancelled_early = False

        for symbol in symbols:
            if _cancel_requested():
                cancelled_early = True
                break

            update_session = factory()
            try:
                current = update_session.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
                if current:
                    current.current_symbol = symbol
                    update_session.commit()
            finally:
                update_session.close()

            ran_optimization = False
            try:
                df = fetch_ohlcv_range(symbol, run_timeframe, run_start, run_end)
                if df is None or df.empty or len(df) < 30:
                    all_results[symbol] = {"error": "No historical OHLCV available for selected range"}
                    _bump_progress(run_trials)
                    continue

                ins_df, oos_df = _split_sample(df)
                ins_open = ins_df["open"].to_numpy(dtype=float)
                ins_high = ins_df["high"].to_numpy(dtype=float)
                ins_low = ins_df["low"].to_numpy(dtype=float)
                ins_close = ins_df["close"].to_numpy(dtype=float)
                ins_vol = ins_df["volume"].to_numpy(dtype=float)

                opt_result = run_optimization(
                    ins_high,
                    ins_low,
                    ins_close,
                    ins_vol,
                    open_=ins_open,
                    timestamps=ins_df.index,
                    timeframe=run_timeframe,
                    n_trials=run_trials,
                    objective=run_objective,
                    min_trades=min_trades,
                    suggest_params_fn=suggest_fn,
                    base_params=base_params,
                    trial_complete_callback=_on_trial_complete,
                    stop_requested=_cancel_requested,
                )
                ran_optimization = True
                if int(opt_result.get("n_trials_completed", 0)) == 0:
                    if _cancel_requested():
                        cancelled_early = True
                        break
                    all_results[symbol] = {"error": f"No completed trials (all pruned; min_trades={min_trades})"}
                    continue
                best_params = opt_result["best_params"]
                in_sample = opt_result["best_backtest"]

                oos_open = oos_df["open"].to_numpy(dtype=float)
                oos_high = oos_df["high"].to_numpy(dtype=float)
                oos_low = oos_df["low"].to_numpy(dtype=float)
                oos_close = oos_df["close"].to_numpy(dtype=float)
                oos_vol = oos_df["volume"].to_numpy(dtype=float)
                out_sample = run_backtest(
                    oos_high,
                    oos_low,
                    oos_close,
                    oos_vol,
                    open_=oos_open,
                    timestamps=oos_df.index,
                    timeframe=run_timeframe,
                    **best_params,
                )

                full_open = df["open"].to_numpy(dtype=float)
                full_high = df["high"].to_numpy(dtype=float)
                full_low = df["low"].to_numpy(dtype=float)
                full_close = df["close"].to_numpy(dtype=float)
                full_vol = df["volume"].to_numpy(dtype=float)
                wf_result = run_walk_forward(
                    full_high,
                    full_low,
                    full_close,
                    full_vol,
                    best_params,
                    open_=full_open,
                    timestamps=df.index,
                    timeframe=run_timeframe,
                )

                all_results[symbol] = _serialize_symbol_result(best_params, opt_result, in_sample, out_sample, wf_result)
                value = opt_result.get("best_value")
                if value is not None and (best_seen is None or float(value) > best_seen):
                    best_seen = float(value)
            except Exception as symbol_exc:
                logger.error("Simulation symbol failed: %s %s", symbol, symbol_exc)
                all_results[symbol] = {"error": str(symbol_exc)}
                if not ran_optimization:
                    _bump_progress(run_trials)

            update_session = factory()
            try:
                current = update_session.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
                if current:
                    current.results = json.dumps(all_results)
                    current.best_value = _safe_round(best_seen)
                    update_session.commit()
            finally:
                update_session.close()

        final_session = factory()
        try:
            current = final_session.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
            if current:
                current.results = json.dumps(all_results)
                current.best_value = _safe_round(best_seen)
                current.current_symbol = None
                if cancelled_early:
                    current.status = "cancelled"
                else:
                    current.status = "completed"
                    current.progress_current = current.progress_total
                final_session.commit()
        finally:
            final_session.close()

    except Exception as exc:
        logger.error("Simulation failed: %s", exc)
        err_session = factory()
        try:
            current = err_session.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
            if current:
                current.status = "error"
                current.error = str(exc)
                current.current_symbol = None
                err_session.commit()
        finally:
            err_session.close()


def apply_simulation_results(sim_id: int) -> dict[str, Any]:
    from app.database import get_session_factory
    from app.models import OptimizationResult, SimulationRun

    factory = get_session_factory()
    session = factory()
    try:
        run = session.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
        if run is None:
            return {"applied": []}

        results = _json_load(run.results, {})
        applied: list[dict[str, Any]] = []

        for symbol, payload in results.items():
            if not isinstance(payload, dict) or payload.get("error"):
                continue

            best_params = payload.get("best_params") or {}
            in_sample = payload.get("in_sample") or {}
            wf = payload.get("walk_forward") or {}
            record_kwargs = {k: best_params.get(k) for k in OptimizationResult.__table__.columns.keys() if k in best_params}

            current_row = (
                session.query(OptimizationResult)
                .filter(
                    OptimizationResult.symbol == symbol,
                    OptimizationResult.timeframe == run.timeframe,
                    OptimizationResult.is_current == True,
                )
                .first()
            )
            if current_row is None:
                existing_row = (
                    session.query(OptimizationResult)
                    .filter(
                        OptimizationResult.symbol == symbol,
                        OptimizationResult.timeframe == run.timeframe,
                    )
                    .order_by(OptimizationResult.id.desc())
                    .first()
                )
                if existing_row is None:
                    current_row = OptimizationResult(symbol=symbol, timeframe=run.timeframe)
                    session.add(current_row)
                    session.flush()
                else:
                    current_row = existing_row
                session.execute(
                    update(OptimizationResult)
                    .where(
                        OptimizationResult.symbol == symbol,
                        OptimizationResult.timeframe == run.timeframe,
                        OptimizationResult.id != current_row.id,
                    )
                    .values(is_current=False)
                )
            else:
                session.execute(
                    update(OptimizationResult)
                    .where(
                        OptimizationResult.symbol == symbol,
                        OptimizationResult.timeframe == run.timeframe,
                        OptimizationResult.id != current_row.id,
                    )
                    .values(is_current=False)
                )

            current_row.win_rate = float(in_sample.get("win_rate", 0.0))
            current_row.tp2_rate = float(in_sample.get("tp2_rate", 0.0))
            current_row.sl_rate = float(in_sample.get("sl_rate", 0.0))
            current_row.total_signals = int(in_sample.get("signals", 0))
            current_row.walk_forward_score = float(wf.get("score", 0.0))
            current_row.consistency_score = float(wf.get("consistency", 0.0))
            current_row.is_current = True
            for key, value in record_kwargs.items():
                setattr(current_row, key, value)
            applied.append({"symbol": symbol, "timeframe": run.timeframe})

        session.commit()
        return {"applied": applied}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
