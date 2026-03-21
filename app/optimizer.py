"""
Optuna Bayesian optimization engine.

Uses the TPE sampler with MedianPruner early stopping to efficiently search
the QTAlgo parameter space.

Objectives:
  - "win_rate"       : maximise raw win rate
  - "tp2_rate"       : maximise TP2 hit rate
  - "risk_adjusted"  : maximise win_rate − sl_rate
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna

from app.backtester import run_backtest
from app.config import DEFAULT_TRIALS, PARAM_RANGES

logger = logging.getLogger(__name__)

# Suppress verbose Optuna output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective_fn(
    trial: optuna.Trial,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    objective: str,
) -> float:
    left_bars = trial.suggest_int("left_bars", *PARAM_RANGES["left_bars"])
    right_bars = trial.suggest_int("right_bars", *PARAM_RANGES["right_bars"])
    offset = trial.suggest_float("offset", *PARAM_RANGES["offset"])
    atr_multiplier = trial.suggest_float("atr_multiplier", *PARAM_RANGES["atr_multiplier"])
    atr_period = trial.suggest_int("atr_period", *PARAM_RANGES["atr_period"])

    result = run_backtest(
        high, low, close,
        left_bars=left_bars,
        right_bars=right_bars,
        offset=offset,
        atr_multiplier=atr_multiplier,
        atr_period=atr_period,
    )

    if result["total_signals"] < 10:
        raise optuna.exceptions.TrialPruned()

    if objective == "win_rate":
        return result["win_rate"]
    if objective == "tp2_rate":
        return result["tp2_rate"]
    # risk_adjusted
    return result["win_rate"] - result["sl_rate"]


def run_optimization(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n_trials: int = DEFAULT_TRIALS,
    objective: str = "risk_adjusted",
    storage: str | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """
    Run Bayesian optimisation over QTAlgo parameters.

    Parameters
    ----------
    high, low, close : price arrays
    n_trials         : number of Optuna trials (default 500)
    objective        : "win_rate", "tp2_rate", or "risk_adjusted"
    storage          : Optuna RDB storage URL (PostgreSQL), or None for in-memory
    study_name       : name for the Optuna study (used with storage)

    Returns
    -------
    {
        "best_params": dict,
        "best_value": float,
        "best_backtest": dict,
        "top_trials": list[dict],   # top 10 trials
        "n_trials_completed": int,
    }
    """
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)

    create_kwargs: dict[str, Any] = {
        "direction": "maximize",
        "sampler": sampler,
        "pruner": pruner,
    }
    if storage:
        create_kwargs["storage"] = storage
        create_kwargs["study_name"] = study_name or "qtalgo_optimization"
        create_kwargs["load_if_exists"] = True

    study = optuna.create_study(**create_kwargs)

    study.optimize(
        lambda trial: _objective_fn(trial, high, low, close, objective),
        n_trials=n_trials,
        catch=(Exception,),
        show_progress_bar=False,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed:
        # Fallback: return default parameters when no trials completed
        logger.warning("No completed Optuna trials (all pruned). Returning default params.")
        default_params = {
            "left_bars": 10,
            "right_bars": 10,
            "offset": 2.0,
            "atr_multiplier": 1.0,
            "atr_period": 14,
        }
        best_backtest = run_backtest(high, low, close, **default_params)
        return {
            "best_params": default_params,
            "best_value": 0.0,
            "best_backtest": best_backtest,
            "top_trials": [],
            "n_trials_completed": 0,
        }

    best_trial = study.best_trial
    best_params = best_trial.params

    # Run full backtest with best params to get all metrics
    best_backtest = run_backtest(high, low, close, **best_params)

    # Top 10 completed trials sorted by value
    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = [
        {"params": t.params, "value": round(t.value, 4)}
        for t in completed[:10]
    ]

    return {
        "best_params": best_params,
        "best_value": round(best_trial.value, 4),
        "best_backtest": best_backtest,
        "top_trials": top_trials,
        "n_trials_completed": len(completed),
    }
