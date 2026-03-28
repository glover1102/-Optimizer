"""
FastAPI application — QTAlgo Optimizer.

Endpoints:
  GET  /                              → Dashboard HTML
  GET  /results/{symbol}/{timeframe}  → Detail HTML
  GET  /api/results                   → All results JSON
  GET  /api/results/{symbol}/{tf}     → Specific result JSON
  POST /api/optimize                  → Trigger manual optimization (single symbol)
  POST /api/optimize/all              → Trigger full watchlist optimization
  POST /api/signals/generate          → Trigger signal generation (single symbol)
  POST /api/signals/generate/all      → Trigger signal generation for all watchlist
  GET  /api/signals                   → Get all current signals
  GET  /api/signals/{symbol}/{tf}     → Get signal for symbol/timeframe
  POST /api/webhook/tv                → Receive TradingView alert webhooks
  GET  /api/health                    → Health check
  POST /api/db/migrate                → Manually trigger schema migration (requires passcode)
  GET  /api/db/status                 → Show table/column status (requires passcode)
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import WATCHLIST

logger = logging.getLogger(__name__)

# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from app.database import init_db
        init_db()
    except Exception as exc:
        logger.error("Database init failed: %s — app will start without DB", exc)
    try:
        from app.scheduler import start_scheduler
        start_scheduler()
    except Exception as exc:
        logger.error("Scheduler start failed: %s", exc)
    # Verify Discord webhook connectivity on startup
    import os as _os
    _webhook_url = _os.environ.get("DISCORD_WEBHOOK_OPTIMIZER", "") or _os.environ.get("DISCORD_WEBHOOK_URL", "")
    if _webhook_url:
        logger.info("Discord webhook configured — notifications enabled")
        try:
            from app.discord_notifier import send_startup_message
            send_startup_message()
        except Exception as exc:
            logger.warning("Discord startup test failed: %s", exc)
    else:
        logger.warning("No Discord webhook URL set (DISCORD_WEBHOOK_OPTIMIZER or DISCORD_WEBHOOK_URL) — Discord notifications DISABLED")
    logger.info("QTAlgo Optimizer started.")
    yield
    try:
        from app.scheduler import stop_scheduler
        stop_scheduler()
    except Exception as exc:
        logger.error("Scheduler stop failed: %s", exc)
    logger.info("QTAlgo Optimizer stopped.")


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="QTAlgo Optimizer",
    description="Bayesian parameter optimization server for the QTAlgo TradingView indicator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")


# ── Helpers ───────────────────────────────────────────────────────────────────

_ASSET_CLASS_MAP: dict[str, str] = {}
for cls, syms in WATCHLIST.items():
    for s in syms:
        _ASSET_CLASS_MAP[s] = cls


def _asset_class(symbol: str) -> str:
    return _ASSET_CLASS_MAP.get(symbol, "other")


def _result_to_dict(r) -> dict[str, Any]:
    return {
        "id": r.id,
        "symbol": r.symbol,
        "timeframe": r.timeframe,
        "asset_class": _asset_class(r.symbol),
        "left_bars": r.left_bars,
        "right_bars": r.right_bars,
        "offset": r.offset,
        "atr_multiplier": r.atr_multiplier,
        "atr_period": r.atr_period,
        "win_rate": r.win_rate,
        "tp2_rate": r.tp2_rate,
        "tp3_rate": r.tp3_rate,
        "sl_rate": r.sl_rate,
        "total_signals": r.total_signals,
        "walk_forward_score": r.walk_forward_score,
        "consistency_score": r.consistency_score,
        "confidence_grade": r.confidence_grade,
        "confidence_score": r.confidence_score,
        "regime": r.regime,
        "optimized_at": r.optimized_at.isoformat() if r.optimized_at else None,
        "is_current": r.is_current,
    }


def _signal_to_dict(s) -> dict[str, Any]:
    return {
        "id": s.id,
        "symbol": s.symbol,
        "timeframe": s.timeframe,
        "asset_class": _asset_class(s.symbol),
        "action": s.action,
        "strength": s.strength,
        "entry_price": s.entry_price,
        "sl_price": s.sl_price,
        "tp1_price": s.tp1_price,
        "tp2_price": s.tp2_price,
        "tp3_price": s.tp3_price,
        "regime": s.regime,
        "entry_mode": s.entry_mode,
        "is_confluence": s.is_confluence,
        "confidence": s.confidence,
        "filters_used": s.filters_used,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "is_current": s.is_current,
        "outcome": s.outcome,
        "outcome_at": s.outcome_at.isoformat() if s.outcome_at else None,
        "outcome_price": s.outcome_price,
        "highest_tp_hit": s.highest_tp_hit,
        "pnl_percent": s.pnl_percent,
    }


# ── HTML routes ───────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    asset_class: Optional[str] = None,
):
    try:
        from app.database import get_db, is_db_available
        from app.models import OptimizationResult

        if not is_db_available():
            return templates.TemplateResponse(
                request=request,
                name="dashboard.html",
                context={
                    "request": request,
                    "results": [],
                    "signals": [],
                    "selected_class": asset_class,
                    "last_run": None,
                    "next_run": None,
                    "signal_stats": {},
                    "signal_top_symbol": "",
                    "top_picks_symbol": "",
                },
            )

        db_gen = get_db()
        db = next(db_gen)
        try:
            from app.models import SignalRecommendation
            query = db.query(OptimizationResult).filter(OptimizationResult.is_current == True)
            if asset_class:
                symbols = WATCHLIST.get(asset_class, [])
                query = query.filter(OptimizationResult.symbol.in_(symbols))

            results = query.order_by(OptimizationResult.confidence_score.desc()).all()
            enriched = []
            for r in results:
                obj = type("R", (), _result_to_dict(r))()
                obj.optimized_at = r.optimized_at
                obj.asset_class = _asset_class(r.symbol)
                enriched.append(obj)

            sig_query = db.query(SignalRecommendation).filter(SignalRecommendation.is_current == True)
            if asset_class:
                symbols = WATCHLIST.get(asset_class, [])
                sig_query = sig_query.filter(SignalRecommendation.symbol.in_(symbols))
            raw_signals = sig_query.order_by(SignalRecommendation.created_at.desc()).all()
            signals = [type("S", (), _signal_to_dict(s))() for s in raw_signals]
        finally:
            db_gen.close()

        try:
            from app.scheduler import get_scheduler_status
            status = get_scheduler_status()
        except Exception:
            status = {"last_run": None, "next_run": None, "running": False}

        # Compute enhanced signal stats for dashboard stat cards
        all_signals_list = [_signal_to_dict(s) for s in raw_signals]
        resolved = [s for s in all_signals_list if s.get("outcome")]
        tp1_hits = sum(1 for s in resolved if s["outcome"] in ("tp1_hit", "tp2_hit", "tp3_hit"))
        tp2_hits = sum(1 for s in resolved if s["outcome"] in ("tp2_hit", "tp3_hit"))
        tp3_hits = sum(1 for s in resolved if s["outcome"] == "tp3_hit")
        sl_hits = sum(1 for s in resolved if s["outcome"] == "sl_hit")
        open_count = sum(1 for s in all_signals_list if not s.get("outcome"))
        buy_count = sum(1 for s in all_signals_list if s["action"] == "BUY")
        sell_count = sum(1 for s in all_signals_list if s["action"] == "SELL")
        confluence_count = sum(1 for s in all_signals_list if s.get("is_confluence"))
        active_list = [s for s in all_signals_list if s["action"] != "HOLD"]
        avg_confidence = (
            sum(s.get("confidence") or 0 for s in active_list) / len(active_list)
            if active_list else 0.0
        )

        # Top symbol for active signals (highest combined strength + confidence score)
        signal_top_symbol = ""
        if active_list:
            symbol_scores: dict[str, float] = {}
            for s in active_list:
                sym = s["symbol"]
                # Weight confidence 4x more than strength (confidence is 0–1, strength is 0–4)
                score = (s.get("strength") or 0) + (s.get("confidence") or 0) * 4
                symbol_scores[sym] = symbol_scores.get(sym, 0) + score
            if symbol_scores:
                signal_top_symbol = max(symbol_scores, key=symbol_scores.get)

        # Top symbol for Top Picks (Grade A & B only, highest win rate)
        top_picks_symbol = ""
        top_picks = [r for r in enriched if getattr(r, "confidence_grade", "") in ("A", "B")]
        if top_picks:
            best_pick = max(top_picks, key=lambda r: (r.win_rate or 0))
            top_picks_symbol = best_pick.symbol

        signal_stats = {
            "resolved_count": len(resolved),
            "tp1_hits": tp1_hits,
            "tp2_hits": tp2_hits,
            "tp3_hits": tp3_hits,
            "sl_hits": sl_hits,
            "open_count": open_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "confluence_count": confluence_count,
            "avg_confidence": avg_confidence,
        }

        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "request": request,
                "results": enriched,
                "signals": signals,
                "selected_class": asset_class,
                "last_run": status["last_run"],
                "next_run": status["next_run"],
                "signal_stats": signal_stats,
                "signal_top_symbol": signal_top_symbol,
                "top_picks_symbol": top_picks_symbol,
            },
        )
    except Exception as exc:
        logger.error("Dashboard error: %s", exc)
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "request": request,
                "results": [],
                "signals": [],
                "selected_class": asset_class,
                "last_run": None,
                "next_run": None,
                "signal_stats": {},
                "signal_top_symbol": "",
                "top_picks_symbol": "",
            },
        )


@app.get("/results/{symbol}/{timeframe}", response_class=HTMLResponse)
async def detail(
    request: Request,
    symbol: str,
    timeframe: str,
):
    try:
        from app.database import get_db
        from app.models import OptimizationResult

        db_gen = get_db()
        db = next(db_gen)
        try:
            result = (
                db.query(OptimizationResult)
                .filter(
                    OptimizationResult.symbol == symbol,
                    OptimizationResult.timeframe == timeframe,
                    OptimizationResult.is_current == True,
                )
                .first()
            )
            if result:
                result.asset_class = _asset_class(result.symbol)
        finally:
            db_gen.close()
    except Exception:
        result = None

    return templates.TemplateResponse(
        request=request,
        name="detail.html",
        context={
            "request": request,
            "result": result,
            "symbol": symbol,
            "timeframe": timeframe,
            "wf_windows": None,
        },
    )


# ── JSON API routes ───────────────────────────────────────────────────────────


@app.get("/api/results")
async def api_results():
    try:
        from app.database import get_db
        from app.models import OptimizationResult

        db_gen = get_db()
        db = next(db_gen)
        try:
            results = (
                db.query(OptimizationResult)
                .filter(OptimizationResult.is_current == True)
                .order_by(OptimizationResult.confidence_score.desc())
                .all()
            )
            return [_result_to_dict(r) for r in results]
        finally:
            db_gen.close()
    except Exception as exc:
        logger.error("API results error: %s", exc)
        return []


@app.get("/api/results/{symbol}/{timeframe}")
async def api_result_detail(symbol: str, timeframe: str):
    try:
        from app.database import get_db
        from app.models import OptimizationResult

        db_gen = get_db()
        db = next(db_gen)
        try:
            result = (
                db.query(OptimizationResult)
                .filter(
                    OptimizationResult.symbol == symbol,
                    OptimizationResult.timeframe == timeframe,
                    OptimizationResult.is_current == True,
                )
                .first()
            )
            if not result:
                raise HTTPException(status_code=404, detail=f"No result for {symbol} {timeframe}")
            return _result_to_dict(result)
        finally:
            db_gen.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class OptimizeRequest(BaseModel):
    symbol: str
    timeframe: str
    trials: int = 100
    objective: str = "risk_adjusted"
    code: str = ""


@app.post("/api/optimize")
async def api_optimize(req: OptimizeRequest):
    """Trigger a manual optimization for a single symbol/timeframe."""
    from app.config import OPTIMIZE_PASSCODE
    if req.code != OPTIMIZE_PASSCODE:
        raise HTTPException(status_code=403, detail="Invalid optimization code")

    def _run():
        from app.scheduler import _optimize_symbol
        from app.database import get_session_factory
        try:
            factory = get_session_factory()
            session = factory()
            try:
                _optimize_symbol(req.symbol, req.timeframe, session)
            finally:
                session.close()
        except Exception as exc:
            logger.error("Manual optimization failed: %s", exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "status": "started",
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "trials": req.trials,
        "message": "Optimization running in background. Check /api/results for updates.",
    }


class OptimizeAllRequest(BaseModel):
    code: str


@app.post("/api/optimize/all")
async def api_optimize_all(req: OptimizeAllRequest):
    """Trigger optimization for all symbols/timeframes in the watchlist."""
    from app.config import OPTIMIZE_PASSCODE
    if req.code != OPTIMIZE_PASSCODE:
        raise HTTPException(status_code=403, detail="Invalid optimization code")

    def _run():
        from app.scheduler import _run_full_watchlist  # deferred — consistent with app import pattern
        try:
            _run_full_watchlist()
        except Exception as exc:
            logger.error("Full watchlist optimization failed: %s", exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": "Full watchlist optimization running in background...",
    }


class TVWebhookPayload(BaseModel):
    symbol: str
    timeframe: Optional[str] = "1h"
    action: Optional[str] = None
    price: Optional[float] = None
    message: Optional[str] = None


def _persist_signal(sig: dict, symbol: str, timeframe: str) -> None:
    """Store a signal recommendation in the database (best-effort)."""
    try:
        from app.database import get_db, is_db_available
        from app.models import SignalRecommendation
        from sqlalchemy import update

        if not is_db_available():
            return

        db_gen = get_db()
        db = next(db_gen)
        try:
            db.execute(
                update(SignalRecommendation)
                .where(
                    SignalRecommendation.symbol == symbol,
                    SignalRecommendation.timeframe == timeframe,
                )
                .values(is_current=False)
            )
            record = SignalRecommendation(
                symbol=symbol,
                timeframe=timeframe,
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
                filters_used=json.dumps(sig.get("filters_active", [])),
                is_current=True,
            )
            db.add(record)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("Signal DB persist failed: %s", exc)
        finally:
            db_gen.close()
    except Exception as exc:
        logger.warning("Signal persist error: %s", exc)


@app.post("/api/webhook/tv")
async def webhook_tradingview(payload: TVWebhookPayload):
    """Receive TradingView alert webhooks, store signal, optionally notify Discord."""
    logger.info(
        "TV webhook: symbol=%s tf=%s action=%s price=%s",
        payload.symbol, payload.timeframe, payload.action, payload.price
    )

    action = (payload.action or "").upper()
    if action in ("BUY", "SELL") and payload.price is not None:
        sig = {
            "action": action,
            "strength": 2,
            "entry_price": payload.price,
            "sl_price": None,
            "tp1_price": None,
            "tp2_price": None,
            "tp3_price": None,
            "regime": "unknown",
            "filters_active": ["tradingview_webhook"],
            "entry_mode": "Pivot",
            "is_confluence": False,
            "confidence": 0.5,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        def _bg():
            _persist_signal(sig, payload.symbol, payload.timeframe or "1h")
            try:
                from app.discord_notifier import notify_signal
                notify_signal(payload.symbol, payload.timeframe or "1h", sig)
            except Exception:
                pass

        threading.Thread(target=_bg, daemon=True).start()

    return {"status": "received", "symbol": payload.symbol, "timeframe": payload.timeframe}


# ── Signal API routes ──────────────────────────────────────────────────────────


@app.get("/api/signals")
async def api_signals_all():
    """Get all current signal recommendations."""
    try:
        from app.database import get_db, is_db_available
        from app.models import SignalRecommendation

        if not is_db_available():
            return []

        db_gen = get_db()
        db = next(db_gen)
        try:
            signals = (
                db.query(SignalRecommendation)
                .filter(SignalRecommendation.is_current == True)
                .order_by(SignalRecommendation.created_at.desc())
                .all()
            )
            return [_signal_to_dict(s) for s in signals]
        finally:
            db_gen.close()
    except Exception as exc:
        logger.error("API signals error: %s", exc)
        return []


@app.get("/api/signals/{symbol}/{timeframe}")
async def api_signal_detail(symbol: str, timeframe: str):
    """Get current signal for a specific symbol/timeframe."""
    try:
        from app.database import get_db, is_db_available
        from app.models import SignalRecommendation

        if not is_db_available():
            raise HTTPException(status_code=503, detail="Database unavailable")

        db_gen = get_db()
        db = next(db_gen)
        try:
            sig = (
                db.query(SignalRecommendation)
                .filter(
                    SignalRecommendation.symbol == symbol,
                    SignalRecommendation.timeframe == timeframe,
                    SignalRecommendation.is_current == True,
                )
                .first()
            )
            if not sig:
                raise HTTPException(status_code=404, detail=f"No signal for {symbol} {timeframe}")
            return _signal_to_dict(sig)
        finally:
            db_gen.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class SignalGenerateRequest(BaseModel):
    symbol: str
    timeframe: str
    code: str = ""
    params: Optional[dict] = None


@app.post("/api/signals/generate")
async def api_signals_generate(req: SignalGenerateRequest):
    """Trigger signal generation for a single symbol/timeframe (requires passcode)."""
    from app.config import OPTIMIZE_PASSCODE
    if req.code != OPTIMIZE_PASSCODE:
        raise HTTPException(status_code=403, detail="Invalid code")

    def _run():
        try:
            from app.signal_generator import generate_signal
            sig = generate_signal(req.symbol, req.timeframe, req.params)
            _persist_signal(sig, req.symbol, req.timeframe)
            if sig.get("action") in ("BUY", "SELL"):
                try:
                    from app.discord_notifier import notify_signal
                    notify_signal(req.symbol, req.timeframe, sig)
                except Exception as exc:
                    logger.warning("Discord notification failed: %s", exc)
        except Exception as exc:
            logger.error("Manual signal generation failed: %s", exc)

    threading.Thread(target=_run, daemon=True).start()
    return {
        "status": "started",
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "message": "Signal generation running in background. Check /api/signals for updates.",
    }


class SignalGenerateAllRequest(BaseModel):
    code: str


@app.post("/api/signals/generate/all")
async def api_signals_generate_all(req: SignalGenerateAllRequest):
    """Trigger signal generation for all watchlist symbols."""
    from app.config import OPTIMIZE_PASSCODE
    if req.code != OPTIMIZE_PASSCODE:
        raise HTTPException(status_code=403, detail="Invalid code")

    def _run():
        try:
            from app.scheduler import _run_signal_generation
            _run_signal_generation()
        except Exception as exc:
            logger.error("Bulk signal generation failed: %s", exc)

    threading.Thread(target=_run, daemon=True).start()
    return {
        "status": "started",
        "message": "Signal generation for all watchlist symbols running in background...",
    }


@app.get("/api/health")
async def health():
    """Health check — must ALWAYS return 200, no matter what."""
    try:
        from app.scheduler import get_scheduler_status
        status = get_scheduler_status()
    except Exception:
        status = {"last_run": None, "next_run": None, "running": False}
    return {
        "status": "ok",
        "last_run": status.get("last_run"),
        "next_run": status.get("next_run"),
        "scheduler_running": status.get("running", False),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/discord/test")
async def test_discord():
    """Send a test message to Discord webhook."""
    from app.discord_notifier import _get_webhook_url
    webhook_url = _get_webhook_url()
    if not webhook_url:
        raise HTTPException(status_code=400, detail="No Discord webhook URL configured (set DISCORD_WEBHOOK_OPTIMIZER or DISCORD_WEBHOOK_URL)")
    try:
        from app.discord_notifier import send_startup_message
        send_startup_message()
        return {"status": "sent", "message": "Test message sent to Discord"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Discord test failed: {exc}")


class DbMigrateRequest(BaseModel):
    code: str


@app.post("/api/db/migrate")
async def api_db_migrate(req: DbMigrateRequest):
    """Manually trigger schema migration to add missing columns (requires passcode)."""
    from app.config import OPTIMIZE_PASSCODE
    if req.code != OPTIMIZE_PASSCODE:
        raise HTTPException(status_code=403, detail="Invalid code")

    try:
        from app.database import get_engine, run_migrations
        result = run_migrations(get_engine())
        return {
            "status": "ok",
            "added": result["added"],
            "skipped": result["skipped"],
            "errors": result["errors"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Migration failed: {exc}")


@app.get("/api/db/status")
async def api_db_status(code: str = ""):
    """Show per-table column status (requires passcode query param ?code=...)."""
    from app.config import OPTIMIZE_PASSCODE
    if code != OPTIMIZE_PASSCODE:
        raise HTTPException(status_code=403, detail="Invalid code")

    try:
        from sqlalchemy import inspect as sa_inspect
        from app.database import get_engine, is_db_available
        from app.models import Base

        if not is_db_available():
            return {"db_available": False}

        engine = get_engine()
        inspector = sa_inspect(engine)
        existing_tables = set(inspector.get_table_names())

        tables_status = {}
        for table_name, table in Base.metadata.tables.items():
            if table_name not in existing_tables:
                tables_status[table_name] = {"exists": False}
                continue

            existing_cols = {row["name"] for row in inspector.get_columns(table_name)}
            model_cols = {col.name for col in table.columns}
            missing = sorted(model_cols - existing_cols)
            tables_status[table_name] = {
                "exists": True,
                "existing_columns": sorted(existing_cols),
                "model_columns": sorted(model_cols),
                "missing_columns": missing,
            }

        return {"db_available": True, "dialect": engine.dialect.name, "tables": tables_status}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB status check failed: {exc}")


@app.get("/api/signals/history")
async def api_signals_history():
    """Get the last 50 resolved signals (where outcome is not None), ordered by outcome_at desc."""
    try:
        from app.database import get_db, is_db_available
        from app.models import SignalRecommendation

        if not is_db_available():
            return []

        db_gen = get_db()
        db = next(db_gen)
        try:
            signals = (
                db.query(SignalRecommendation)
                .filter(SignalRecommendation.outcome.isnot(None))
                .order_by(SignalRecommendation.outcome_at.desc())
                .limit(50)
                .all()
            )
            return [_signal_to_dict(s) for s in signals]
        finally:
            db_gen.close()
    except Exception as exc:
        logger.error("API signals history error: %s", exc)
        return []
