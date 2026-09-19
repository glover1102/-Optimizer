"""Optuna optimizer for the KLS+MoM strategy."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna

from app.backtester import run_backtest
from app.config import DEFAULT_SIGNAL_PARAMS, DEFAULT_TRIALS, PARAM_RANGES

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def _objective_fn(
    trial: optuna.Trial,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None,
    timestamps,
    timeframe: str | None,
    objective: str,
) -> float:
    params = _suggest_params(trial)
    result = run_backtest(
        high,
        low,
        close,
        volume,
        timestamps=timestamps,
        timeframe=timeframe,
        **params,
    )
    if result["total_signals"] < 3:
        raise optuna.exceptions.TrialPruned()
    if objective == "win_rate":
        return result["win_rate"]
    if objective == "tp2_rate":
        return result["tp2_rate"]
    return result["win_rate"] - result["sl_rate"]


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
) -> dict[str, Any]:
    """Run Bayesian optimisation over KLS+MoM parameters."""
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    create_kwargs: dict[str, Any] = {"direction": "maximize", "sampler": sampler, "pruner": pruner}
    if storage:
        create_kwargs.update({"storage": storage, "study_name": study_name or "kls_mom_optimization", "load_if_exists": True})
    study = optuna.create_study(**create_kwargs)
    study.optimize(
        lambda trial: _objective_fn(trial, high, low, close, volume, timestamps, timeframe, objective),
        n_trials=n_trials,
        catch=(Exception,),
        show_progress_bar=False,
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("No completed Optuna trials (all pruned). Returning default KLS params.")
        best_backtest = run_backtest(high, low, close, volume, open_=open_, timestamps=timestamps, timeframe=timeframe, **DEFAULT_SIGNAL_PARAMS)
        return {
            "best_params": dict(DEFAULT_SIGNAL_PARAMS),
            "best_value": 0.0,
            "best_backtest": best_backtest,
            "top_trials": [],
            "n_trials_completed": 0,
        }
    best_trial = study.best_trial
    best_params = dict(DEFAULT_SIGNAL_PARAMS)
    best_params.update(best_trial.params)
    best_backtest = run_backtest(high, low, close, volume, open_=open_, timestamps=timestamps, timeframe=timeframe, **best_params)
    completed.sort(key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    top_trials = [{"params": {**DEFAULT_SIGNAL_PARAMS, **t.params}, "value": round(float(t.value), 4)} for t in completed[:10] if t.value is not None]
    return {
        "best_params": best_params,
        "best_value": round(float(best_trial.value), 4),
        "best_backtest": best_backtest,
        "top_trials": top_trials,
        "n_trials_completed": len(completed),
    }
