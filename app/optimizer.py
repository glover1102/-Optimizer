"""Optuna optimizer for the KLS+MoM strategy."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import optuna

from app.backtester import run_backtest
from app.config import DEFAULT_SIGNAL_PARAMS, DEFAULT_TRIALS, PARAM_RANGES

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _trade_r_multiple(trade: dict[str, Any], params: dict[str, Any]) -> float:
    highest_tp = int(trade.get("highest_tp_hit", 0) or 0)
    if highest_tp >= 1:
        rr = params.get(f"tp{highest_tp}_rr")
        return float(rr) if rr is not None else 0.0
    outcome = str(trade.get("outcome") or "")
    if outcome == "sl_hit":
        return -1.0
    if outcome == "breakeven":
        return 0.0
    return 0.0


def compute_expectancy_r(result: dict[str, Any], params: dict[str, Any]) -> float:
    trades = list(result.get("trades") or [])
    if not trades:
        return 0.0
    values = [_trade_r_multiple(t, params) for t in trades]
    return float(np.mean(values)) if values else 0.0


def compute_profit_factor(result: dict[str, Any], params: dict[str, Any]) -> float:
    trades = list(result.get("trades") or [])
    if not trades:
        return 0.0
    values = [_trade_r_multiple(t, params) for t in trades]
    gross_profit = sum(v for v in values if v > 0)
    gross_loss = -sum(v for v in values if v < 0)
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    p = dict(DEFAULT_SIGNAL_PARAMS)
    p.update(
        {
            "atr_length": trial.suggest_int("atr_length", *PARAM_RANGES["atr_length"]),
            "atr_smoothing": trial.suggest_categorical("atr_smoothing", PARAM_RANGES["categorical"]["atr_smoothing"]),
            "sl_mult": trial.suggest_float("sl_mult", *PARAM_RANGES["sl_mult"]),
            "tp1_rr": trial.suggest_float("tp1_rr", *PARAM_RANGES["tp1_rr"]),
            "tp2_rr": trial.suggest_float("tp2_rr", *PARAM_RANGES["tp2_rr"]),
            "tp3_rr": trial.suggest_float("tp3_rr", *PARAM_RANGES["tp3_rr"]),
            "tp4_rr": trial.suggest_float("tp4_rr", *PARAM_RANGES["tp4_rr"]),
            "be_after_tp": trial.suggest_categorical("be_after_tp", PARAM_RANGES["categorical"]["be_after_tp"]),
            "runner_tgt": trial.suggest_categorical("runner_tgt", PARAM_RANGES["categorical"]["runner_tgt"]),
            "resolve_mode": trial.suggest_categorical("resolve_mode", PARAM_RANGES["categorical"]["resolve_mode"]),
            "filt_mode": trial.suggest_categorical("filt_mode", PARAM_RANGES["categorical"]["filt_mode"]),
            "or2_minutes": trial.suggest_int("or2_minutes", *PARAM_RANGES["or2_minutes"]),
            "or3_minutes": trial.suggest_int("or3_minutes", *PARAM_RANGES["or3_minutes"]),
            "or4_minutes": trial.suggest_int("or4_minutes", *PARAM_RANGES["or4_minutes"]),
            "fvol_mult": trial.suggest_float("fvol_mult", *PARAM_RANGES["fvol_mult"]),
            "fvol_ma_type": trial.suggest_categorical("fvol_ma_type", PARAM_RANGES["categorical"]["fvol_ma_type"]),
            "rsi_neutral": trial.suggest_float("rsi_neutral", *PARAM_RANGES["rsi_neutral"]),
            "or2_enabled": trial.suggest_categorical("or2_enabled", [False, True]),
            "or3_enabled": trial.suggest_categorical("or3_enabled", [False, True]),
            "or4_enabled": trial.suggest_categorical("or4_enabled", [True]),
            "use_fvol": trial.suggest_categorical("use_fvol", [True, False]),
            "use_frsi": trial.suggest_categorical("use_frsi", [True, False]),
            "use_fmacd": trial.suggest_categorical("use_fmacd", [True, False]),
            "use_fvwap": trial.suggest_categorical("use_fvwap", [True, False]),
            "entry_at_level": trial.suggest_categorical("entry_at_level", [False, True]),
        }
    )
    return p


def _objective_score(result: dict[str, Any], objective: str, params: dict[str, Any]) -> float:
    if objective == "win_rate":
        return float(result["win_rate"])
    if objective == "tp2_rate":
        return float(result["tp2_rate"])
    if objective == "profit_factor":
        val = compute_profit_factor(result, params)
        return 999.0 if np.isinf(val) else float(val)
    return compute_expectancy_r(result, params)


def _objective_fn(
    trial: optuna.Trial,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None,
    timestamps,
    timeframe: str | None,
    objective: str,
    min_trades: int,
    suggest_params_fn: Callable[[optuna.Trial], dict[str, Any]] | None = None,
    base_params: dict[str, Any] | None = None,
) -> float:
    if suggest_params_fn is None and base_params is None:
        params = _suggest_params(trial)
    else:
        params = dict(base_params or DEFAULT_SIGNAL_PARAMS)
        suggested = _suggest_params(trial) if suggest_params_fn is None else suggest_params_fn(trial)
        params.update(suggested)
    result = run_backtest(
        high,
        low,
        close,
        volume,
        timestamps=timestamps,
        timeframe=timeframe,
        **params,
    )
    if result["total_signals"] < min_trades:
        raise optuna.exceptions.TrialPruned()
    return _objective_score(result, objective, params)


def run_optimization(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None = None,
    *,
    timestamps=None,
    timeframe: str | None = None,
    open_: np.ndarray | None = None,
    n_trials: int = DEFAULT_TRIALS,
    objective: str = "risk_adjusted",
    storage: str | None = None,
    study_name: str | None = None,
    min_trades: int = 3,
    suggest_params_fn: Callable[[optuna.Trial], dict[str, Any]] | None = None,
    base_params: dict[str, Any] | None = None,
    trial_complete_callback: Callable[[optuna.trial.FrozenTrial], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run Bayesian optimisation over KLS+MoM parameters."""
    resolved_base = dict(DEFAULT_SIGNAL_PARAMS)
    if base_params:
        resolved_base.update(base_params)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    create_kwargs: dict[str, Any] = {"direction": "maximize", "sampler": sampler, "pruner": pruner}
    if storage:
        create_kwargs.update({"storage": storage, "study_name": study_name or "kls_mom_optimization", "load_if_exists": True})
    study = optuna.create_study(**create_kwargs)
    def _cb(study_obj: optuna.Study, frozen_trial: optuna.trial.FrozenTrial):
        if trial_complete_callback is not None:
            trial_complete_callback(frozen_trial)
        if stop_requested is not None and stop_requested():
            study_obj.stop()

    callbacks = [_cb] if trial_complete_callback is not None or stop_requested is not None else None
    study.optimize(
        lambda trial: _objective_fn(
            trial,
            high,
            low,
            close,
            volume,
            timestamps,
            timeframe,
            objective,
            min_trades,
            suggest_params_fn,
            resolved_base if (suggest_params_fn is not None or base_params is not None) else None,
        ),
        n_trials=n_trials,
        catch=(Exception,),
        show_progress_bar=False,
        callbacks=callbacks,
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("No completed Optuna trials (all pruned). Returning default KLS params.")
        best_backtest = run_backtest(high, low, close, volume, open_=open_, timestamps=timestamps, timeframe=timeframe, **resolved_base)
        return {
            "best_params": dict(resolved_base),
            "best_value": 0.0,
            "best_backtest": best_backtest,
            "top_trials": [],
            "n_trials_completed": 0,
        }
    best_trial = study.best_trial
    best_params = dict(resolved_base)
    best_params.update(best_trial.params)
    best_backtest = run_backtest(high, low, close, volume, open_=open_, timestamps=timestamps, timeframe=timeframe, **best_params)
    completed.sort(key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    top_trials = [{"params": {**resolved_base, **t.params}, "value": round(float(t.value), 4)} for t in completed[:10] if t.value is not None]
    return {
        "best_params": best_params,
        "best_value": round(float(best_trial.value), 4),
        "best_backtest": best_backtest,
        "top_trials": top_trials,
        "n_trials_completed": len(completed),
    }
