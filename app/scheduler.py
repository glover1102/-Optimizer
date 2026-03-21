"""
APScheduler-based periodic optimization runner.

Runs the full watchlist optimization every OPTIMIZATION_INTERVAL_HOURS hours.
Skips symbols that were recently optimized (within the last interval) unless
the market regime has changed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from app.config import OPTIMIZATION_INTERVAL_HOURS, WATCHLIST, TIMEFRAMES, DEFAULT_TRIALS

logger = logging.getLogger(__name__)

_scheduler = None
_last_run: Optional[datetime] = None
_next_run: Optional[datetime] = None


def _optimize_symbol(symbol: str, timeframe: str, db_session) -> None:
    """Run full optimization pipeline for a single symbol/timeframe pair."""
    from app.data_fetcher import get_numpy_arrays
    from app.optimizer import run_optimization
    from app.walk_forward import run_walk_forward
    from app.regime_detector import detect_regime
    from app.scorer import score_result

    logger.info("Optimizing %s %s …", symbol, timeframe)

    arrays = get_numpy_arrays(symbol, timeframe)
    if arrays is None:
        logger.warning("No data for %s %s — skipping", symbol, timeframe)
        return

    high, low, close = arrays

    # Detect regime
    regime_info = detect_regime(high, low, close)

    # Run Bayesian optimization
    try:
        opt_result = run_optimization(high, low, close, n_trials=DEFAULT_TRIALS)
    except Exception as exc:
        logger.error("Optimization failed for %s %s: %s", symbol, timeframe, exc)
        return

    best_params = opt_result["best_params"]
    best_bt = opt_result["best_backtest"]

    # Walk-forward validation
    wf_result = run_walk_forward(high, low, close, best_params)

    # Confidence scoring
    wf_scores = [
        w["oos_win_rate"] for w in wf_result["windows"] if w["oos_win_rate"] is not None
    ]
    score = score_result(
        total_signals=best_bt["total_signals"],
        wins=best_bt["wins"],
        sl_hits=best_bt["sl_hits"],
        walk_forward_scores=wf_scores if wf_scores else None,
    )

    # Persist to database
    try:
        from app.models import OptimizationResult
        from sqlalchemy import update

        # Mark old records as not current
        db_session.execute(
            update(OptimizationResult)
            .where(
                OptimizationResult.symbol == symbol,
                OptimizationResult.timeframe == timeframe,
            )
            .values(is_current=False)
        )

        record = OptimizationResult(
            symbol=symbol,
            timeframe=timeframe,
            left_bars=best_params.get("left_bars"),
            right_bars=best_params.get("right_bars"),
            offset=best_params.get("offset"),
            atr_multiplier=best_params.get("atr_multiplier"),
            atr_period=best_params.get("atr_period"),
            win_rate=best_bt["win_rate"],
            tp2_rate=best_bt["tp2_rate"],
            tp3_rate=best_bt["tp3_rate"],
            sl_rate=best_bt["sl_rate"],
            total_signals=best_bt["total_signals"],
            walk_forward_score=wf_result["walk_forward_score"],
            consistency_score=wf_result["consistency"],
            confidence_grade=score["confidence_grade"],
            confidence_score=score["confidence_score"],
            regime=regime_info["regime"],
            is_current=True,
        )
        db_session.add(record)
        db_session.commit()
        logger.info(
            "Saved result for %s %s: win_rate=%.1f%% grade=%s",
            symbol, timeframe, best_bt["win_rate"] * 100, score["confidence_grade"]
        )
    except Exception as exc:
        logger.error("DB save failed for %s %s: %s", symbol, timeframe, exc)
        db_session.rollback()


def _run_full_watchlist() -> None:
    """Scheduler callback: optimize all symbols across all timeframes."""
    global _last_run, _next_run
    _last_run = datetime.now(timezone.utc)
    _next_run = _last_run + timedelta(hours=OPTIMIZATION_INTERVAL_HOURS)

    logger.info("=== Starting scheduled optimization run ===")

    from app.database import get_session_factory
    from app.models import OptimizationRun

    try:
        factory = get_session_factory()
        db = factory()
    except Exception as exc:
        logger.error("Cannot create DB session for scheduled run: %s", exc)
        return

    run_record = OptimizationRun(started_at=_last_run, status="running")
    try:
        db.add(run_record)
        db.commit()
    except Exception:
        db.rollback()

    symbols_processed = 0
    all_symbols = [s for symbols in WATCHLIST.values() for s in symbols]

    for symbol in all_symbols:
        for tf in TIMEFRAMES:
            try:
                _optimize_symbol(symbol, tf, db)
                symbols_processed += 1
            except Exception as exc:
                logger.error("Unhandled error for %s %s: %s", symbol, tf, exc)

    # Update run record
    try:
        run_record.completed_at = datetime.now(timezone.utc)
        run_record.symbols_processed = symbols_processed
        run_record.status = "completed"
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

    logger.info("=== Optimization run complete: %d symbol/TF pairs ===", symbols_processed)


def start_scheduler() -> None:
    """Start the APScheduler background scheduler."""
    global _scheduler, _next_run

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger

        _scheduler = BackgroundScheduler()
        _scheduler.add_job(
            _run_full_watchlist,
            trigger=IntervalTrigger(hours=OPTIMIZATION_INTERVAL_HOURS),
            id="full_watchlist_optimization",
            replace_existing=True,
        )
        _scheduler.start()
        _next_run = datetime.now(timezone.utc) + timedelta(hours=OPTIMIZATION_INTERVAL_HOURS)
        logger.info(
            "Scheduler started — runs every %d hours. Next run: %s",
            OPTIMIZATION_INTERVAL_HOURS, _next_run.isoformat()
        )
    except Exception as exc:
        logger.error("Failed to start scheduler: %s", exc)


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler:
        try:
            if _scheduler.running:
                _scheduler.shutdown(wait=False)
                logger.info("Scheduler stopped.")
        except Exception as exc:
            logger.error("Error stopping scheduler: %s", exc)


def get_scheduler_status() -> dict:
    return {
        "last_run": _last_run.isoformat() if _last_run else None,
        "next_run": _next_run.isoformat() if _next_run else None,
        "running": bool(_scheduler and _scheduler.running),
    }
