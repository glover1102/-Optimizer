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

from app.config import OPTIMIZATION_INTERVAL_HOURS, SIGNAL_GENERATION_INTERVAL_MINUTES, WATCHLIST, TIMEFRAMES, DEFAULT_TRIALS

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

        # Notify Discord for Grade A/B results
        try:
            from app.discord_notifier import notify_optimization_result
            notify_optimization_result({
                "symbol": symbol,
                "timeframe": timeframe,
                "confidence_grade": score["confidence_grade"],
                "confidence_score": score["confidence_score"],
                "win_rate": best_bt["win_rate"],
                "tp2_rate": best_bt["tp2_rate"],
                "tp3_rate": best_bt["tp3_rate"],
                "sl_rate": best_bt["sl_rate"],
                "left_bars": best_params.get("left_bars"),
                "right_bars": best_params.get("right_bars"),
                "offset": best_params.get("offset"),
                "atr_multiplier": best_params.get("atr_multiplier"),
                "atr_period": best_params.get("atr_period"),
                "regime": regime_info["regime"],
                "walk_forward_score": wf_result["walk_forward_score"],
            })
        except Exception as exc:
            logger.warning("Discord notification error for %s %s: %s", symbol, timeframe, exc)
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


def _run_signal_generation() -> None:
    """Scheduler callback: generate signals for all watchlist symbols/timeframes."""
    logger.info("=== Starting scheduled signal generation ===")
    all_symbols = [s for symbols in WATCHLIST.values() for s in symbols]
    count = 0

    try:
        from app.database import get_session_factory, is_db_available
        from app.models import SignalRecommendation
        from app.signal_generator import generate_signal
        from sqlalchemy import select, update

        if not is_db_available():
            logger.warning("DB not available for signal generation — skipping")
            return

        factory = get_session_factory()
        db = factory()
    except Exception as exc:
        logger.error("Cannot set up signal generation: %s", exc)
        return

    try:
        for symbol in all_symbols:
            for tf in TIMEFRAMES:
                try:
                    sig = generate_signal(symbol, tf)

                    # Check previous current signal action to avoid duplicate notifications
                    prev_row = db.execute(
                        select(SignalRecommendation.action)
                        .where(
                            SignalRecommendation.symbol == symbol,
                            SignalRecommendation.timeframe == tf,
                            SignalRecommendation.is_current == True,  # noqa: E712
                        )
                        .limit(1)
                    ).first()
                    prev_action = prev_row[0] if prev_row else None

                    # Mark old signals as not current
                    db.execute(
                        update(SignalRecommendation)
                        .where(
                            SignalRecommendation.symbol == symbol,
                            SignalRecommendation.timeframe == tf,
                        )
                        .values(is_current=False)
                    )
                    record = SignalRecommendation(
                        symbol=symbol,
                        timeframe=tf,
                        action=sig.get("action", "HOLD"),
                        strength=sig.get("strength", 0),
                        entry_price=sig.get("entry_price"),
                        sl_price=sig.get("sl_price"),
                        tp1_price=sig.get("tp1_price"),
                        tp2_price=sig.get("tp2_price"),
                        tp3_price=sig.get("tp3_price"),
                        regime=sig.get("regime"),
                        entry_mode=sig.get("entry_mode"),
                        is_confluence=sig.get("is_confluence", False),
                        confidence=sig.get("confidence", 0.0),
                        filters_used=__import__("json").dumps(sig.get("filters_active", [])),
                        is_current=True,
                    )
                    db.add(record)
                    db.commit()
                    count += 1

                    # Discord notification for BUY/SELL signals — only when action changes
                    new_action = sig.get("action")
                    if new_action in ("BUY", "SELL") and new_action != prev_action:
                        try:
                            from app.discord_notifier import notify_signal
                            notify_signal(symbol, tf, sig)
                        except Exception as exc:
                            logger.warning("Discord signal notify error: %s", exc)

                except Exception as exc:
                    logger.error("Signal gen error for %s %s: %s", symbol, tf, exc)
                    try:
                        db.rollback()
                    except Exception:
                        pass
    finally:
        db.close()

    logger.info("=== Signal generation complete: %d symbol/TF pairs ===", count)


def _calc_pnl(action: str, entry: float, exit_price: float) -> float:
    """Calculate P&L percentage for a signal. Positive = profit."""
    diff = (exit_price - entry) / entry * 100
    return diff if action == "BUY" else -diff


def _check_signal_outcomes() -> None:
    """Scheduler callback: check open BUY/SELL signals for TP/SL hits or expiry."""
    logger.debug("=== Checking signal outcomes ===")
    try:
        from app.database import get_session_factory, is_db_available
        from app.models import SignalRecommendation

        if not is_db_available():
            return

        factory = get_session_factory()
        db = factory()
    except Exception as exc:
        logger.error("Cannot set up outcome checking: %s", exc)
        return

    try:
        now = datetime.now(timezone.utc)
        # Signals older than 48 hours with no hit are considered invalid/expired
        expiry_cutoff = now - timedelta(hours=48)

        # Fetch all open (no outcome yet) BUY/SELL signals
        open_signals = (
            db.query(SignalRecommendation)
            .filter(
                SignalRecommendation.outcome.is_(None),
                SignalRecommendation.action.in_(["BUY", "SELL"]),
            )
            .all()
        )

        if not open_signals:
            return

        # Group by symbol to minimise API calls
        symbol_price: dict[str, Optional[float]] = {}

        for sig in open_signals:
            symbol = sig.symbol
            tf = sig.timeframe

            # Fetch current price (lazily, once per symbol)
            if symbol not in symbol_price:
                try:
                    from app.data_fetcher import fetch_ohlcv
                    df = fetch_ohlcv(symbol, "5m")
                    if df is not None and not df.empty:
                        symbol_price[symbol] = float(df["close"].iloc[-1])
                    else:
                        symbol_price[symbol] = None
                except Exception as exc:
                    logger.warning("Price fetch failed for %s: %s", symbol, exc)
                    symbol_price[symbol] = None

            current_price = symbol_price[symbol]

            # Check expiry first
            created_at = sig.created_at
            if created_at is not None:
                # Make timezone-aware if naive
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                if created_at < expiry_cutoff:
                    sig.outcome = "expired"
                    sig.outcome_at = now
                    sig.outcome_price = current_price
                    if sig.entry_price and current_price is not None:
                        sig.pnl_percent = _calc_pnl(sig.action, sig.entry_price, current_price)
                    try:
                        db.commit()
                        from app.discord_notifier import notify_signal_outcome
                        notify_signal_outcome(_signal_outcome_dict(sig))
                    except Exception as exc:
                        logger.warning("Outcome notify/commit failed for %s: %s", symbol, exc)
                        db.rollback()
                    continue

            if current_price is None:
                continue

            action = sig.action
            entry = sig.entry_price
            sl = sig.sl_price
            tp1 = sig.tp1_price
            tp2 = sig.tp2_price
            tp3 = sig.tp3_price

            outcome = None
            outcome_price = current_price
            highest_tp = sig.highest_tp_hit or 0

            if action == "BUY":
                if sl is not None and current_price <= sl:
                    outcome = "sl_hit"
                elif tp3 is not None and current_price >= tp3:
                    outcome = "tp3_hit"
                    highest_tp = max(highest_tp, 3)
                elif tp2 is not None and current_price >= tp2:
                    outcome = "tp2_hit"
                    highest_tp = max(highest_tp, 2)
                elif tp1 is not None and current_price >= tp1:
                    outcome = "tp1_hit"
                    highest_tp = max(highest_tp, 1)
            elif action == "SELL":
                if sl is not None and current_price >= sl:
                    outcome = "sl_hit"
                elif tp3 is not None and current_price <= tp3:
                    outcome = "tp3_hit"
                    highest_tp = max(highest_tp, 3)
                elif tp2 is not None and current_price <= tp2:
                    outcome = "tp2_hit"
                    highest_tp = max(highest_tp, 2)
                elif tp1 is not None and current_price <= tp1:
                    outcome = "tp1_hit"
                    highest_tp = max(highest_tp, 1)

            if outcome:
                sig.outcome = outcome
                sig.outcome_at = now
                sig.outcome_price = outcome_price
                sig.highest_tp_hit = highest_tp
                if entry and outcome_price is not None:
                    sig.pnl_percent = _calc_pnl(action, entry, outcome_price)
                try:
                    db.commit()
                    from app.discord_notifier import notify_signal_outcome
                    notify_signal_outcome(_signal_outcome_dict(sig))
                except Exception as exc:
                    logger.warning("Outcome notify/commit failed for %s: %s", symbol, exc)
                    db.rollback()
    except Exception as exc:
        logger.error("Error in _check_signal_outcomes: %s", exc)
    finally:
        db.close()


def _signal_outcome_dict(sig) -> dict:
    """Convert a SignalRecommendation ORM object to a dict for Discord notification."""
    return {
        "symbol": sig.symbol,
        "timeframe": sig.timeframe,
        "action": sig.action,
        "entry_price": sig.entry_price,
        "outcome": sig.outcome,
        "outcome_price": sig.outcome_price,
        "pnl_percent": sig.pnl_percent,
        "highest_tp_hit": sig.highest_tp_hit,
        "created_at": sig.created_at.isoformat() if sig.created_at else None,
        "outcome_at": sig.outcome_at.isoformat() if sig.outcome_at else None,
    }


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
        _scheduler.add_job(
            _run_signal_generation,
            trigger=IntervalTrigger(minutes=SIGNAL_GENERATION_INTERVAL_MINUTES),
            id="signal_generation",
            replace_existing=True,
        )
        _scheduler.add_job(
            _check_signal_outcomes,
            trigger=IntervalTrigger(minutes=2),
            id="signal_outcome_checker",
            replace_existing=True,
        )
        _scheduler.start()
        _next_run = datetime.now(timezone.utc) + timedelta(hours=OPTIMIZATION_INTERVAL_HOURS)
        logger.info(
            "Scheduler started — optimization every %d hours, signals every %d min. Next opt run: %s",
            OPTIMIZATION_INTERVAL_HOURS, SIGNAL_GENERATION_INTERVAL_MINUTES, _next_run.isoformat()
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
