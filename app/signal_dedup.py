"""
Signal deduplication helper.

Before persisting a new SignalRecommendation row and firing Discord/Pushover
notifications, callers should check whether the incoming signal is materially
different from the most-recent existing signal for the same symbol+timeframe.

Deduplication logic
-------------------
1. No existing signal → not duplicate (False).
2. Different action → not duplicate (False).
3. Both HOLD → duplicate (True).
4. Same BUY/SELL action **and** existing signal is still open (``outcome`` is
   ``None``) → duplicate (True).  Prices are refreshed on the existing record
   but no new row or notification is created.
5. Same BUY/SELL action **and** existing signal is resolved (``outcome`` is not
   ``None``, e.g. tp1_hit / sl_hit / expired) **but** the resolution happened
   within a cooldown window (``SIGNAL_GENERATION_INTERVAL_MINUTES * 2``) →
   duplicate (True).  Suppresses the immediate re-alert that would otherwise
   fire on the very next generation cycle after outcome resolution.
6. Same BUY/SELL action **and** existing signal is resolved **and** the
   cooldown has expired → not duplicate (False), allow a fresh notification.

Usage
-----
    from app.signal_dedup import is_duplicate_signal

    dup = is_duplicate_signal(db, symbol, timeframe, sig)
    if dup:
        # Nothing to do — is_current was already refreshed by this function.
        logger.info("Skipping duplicate signal for %s %s", symbol, timeframe)
    else:
        # Proceed with insert + notifications.
        ...
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Relative tolerance for price comparison (0.01 %)
_PRICE_TOL = 0.0001

# Fields copied from the incoming signal dict onto the existing DB record when
# an open same-direction signal is refreshed in-place (no new row / no alert).
_REFRESH_FIELDS = (
    "entry_price",
    "sl_price",
    "tp1_price",
    "tp2_price",
    "tp3_price",
    "confidence",
    "strength",
)


def _prices_match(a: Optional[float], b: Optional[float]) -> bool:
    """Return True if two nullable prices are equal within _PRICE_TOL."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom < _PRICE_TOL


def is_duplicate_signal(db, symbol: str, timeframe: str, sig: dict) -> bool:
    """Check whether *sig* duplicates the most-recent signal for symbol+timeframe.

    Side-effects when a duplicate is detected
    ------------------------------------------
    The existing ``SignalRecommendation`` row's ``is_current`` attribute is set
    to ``True`` in-memory so that callers do **not** need to do a separate
    UPDATE.  Callers are responsible for committing this change.

    Returns
    -------
    True  — signal is a duplicate; caller should skip insert and notifications.
    False — signal is new/changed; caller should proceed normally.
    """
    try:
        from app.models import SignalRecommendation
        from sqlalchemy import desc

        # Fetch the most-recent signal for this symbol/timeframe
        existing: Optional[SignalRecommendation] = (
            db.query(SignalRecommendation)
            .filter(
                SignalRecommendation.symbol == symbol,
                SignalRecommendation.timeframe == timeframe,
            )
            .order_by(desc(SignalRecommendation.created_at))
            .first()
        )

        if existing is None:
            return False

        new_action = sig.get("action", "HOLD")

        # Different action → not a duplicate
        if existing.action != new_action:
            return False

        # HOLD vs HOLD → always a duplicate (nothing actionable to re-notify)
        if new_action == "HOLD":
            # Re-mark as current (in-memory; caller is responsible for commit)
            existing.is_current = True
            logger.debug(
                "Duplicate HOLD signal skipped for %s %s", symbol, timeframe
            )
            return True

        # BUY/SELL — check whether the existing signal is still open
        if existing.outcome is None:
            # The existing signal has not resolved yet (no TP/SL/expiry hit).
            # Treat any same-direction signal as a duplicate regardless of price
            # differences — live OHLCV recalculations always produce slightly
            # different prices, so price comparison is not a reliable gate here.
            existing.is_current = True
            # Refresh prices and metadata on the existing record so the DB stays
            # current without triggering a new notification.
            for field in _REFRESH_FIELDS:
                new_val = sig.get(field)
                if new_val is not None:
                    setattr(existing, field, new_val)
            logger.info(
                "Duplicate %s signal skipped for %s %s (existing signal still open)",
                new_action,
                symbol,
                timeframe,
            )
            return True

        # The existing signal has resolved (tp1_hit, tp2_hit, tp3_hit, sl_hit,
        # or expired).  Before allowing a fresh signal + notification, apply a
        # cooldown guard: if the signal was resolved very recently (within
        # SIGNAL_GENERATION_INTERVAL_MINUTES * 2), treat the incoming signal as
        # a duplicate to prevent the "resolve → immediate re-alert" race.
        if existing.outcome_at is not None:
            from app.config import SIGNAL_GENERATION_INTERVAL_MINUTES

            outcome_at = existing.outcome_at
            # outcome_at is stored as UTC in the DB; if the column is naive
            # (no tzinfo), assume UTC so arithmetic is timezone-aware.
            if outcome_at.tzinfo is None:
                outcome_at = outcome_at.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - outcome_at
            cooldown = timedelta(minutes=SIGNAL_GENERATION_INTERVAL_MINUTES * 2)
            if age < cooldown:
                # Recently resolved — suppress re-notification
                existing.is_current = True
                logger.info(
                    "Suppressing re-alert for %s %s: outcome resolved %s ago (cooldown %s)",
                    symbol,
                    timeframe,
                    age,
                    cooldown,
                )
                return True

        # Cooldown expired (or outcome_at not set) — allow fresh signal + notification.
        return False

    except Exception as exc:  # noqa: BLE001
        # If the dedup check itself fails, allow the signal through to avoid
        # silently dropping legitimate signals.
        logger.warning("signal_dedup check failed for %s %s: %s", symbol, timeframe, exc)
        return False
