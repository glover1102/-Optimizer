"""
FastAPI application — QTAlgo Optimizer.

Endpoints:
  GET  /                              → Dashboard HTML
  GET  /results/{symbol}/{timeframe}  → Detail HTML
  GET  /api/results                   → All results JSON
  GET  /api/results/{symbol}/{tf}     → Specific result JSON
  POST /api/optimize                  → Trigger manual optimization
  POST /api/webhook/tv                → Receive TradingView alert webhooks
  GET  /api/health                    → Health check
"""

from __future__ import annotations

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
                "dashboard.html",
                {
                    "request": request,
                    "results": [],
                    "selected_class": asset_class,
                    "last_run": None,
                    "next_run": None,
                },
            )

        db_gen = get_db()
        db = next(db_gen)
        try:
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
        finally:
            db_gen.close()

        try:
            from app.scheduler import get_scheduler_status
            status = get_scheduler_status()
        except Exception:
            status = {"last_run": None, "next_run": None, "running": False}
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "results": enriched,
                "selected_class": asset_class,
                "last_run": status["last_run"],
                "next_run": status["next_run"],
            },
        )
    except Exception as exc:
        logger.error("Dashboard error: %s", exc)
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "results": [],
                "selected_class": asset_class,
                "last_run": None,
                "next_run": None,
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
        "detail.html",
        {
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


@app.post("/api/optimize")
async def api_optimize(req: OptimizeRequest):
    """Trigger a manual optimization for a single symbol/timeframe."""

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


class TVWebhookPayload(BaseModel):
    symbol: str
    timeframe: Optional[str] = "1h"
    action: Optional[str] = None
    price: Optional[float] = None
    message: Optional[str] = None


@app.post("/api/webhook/tv")
async def webhook_tradingview(payload: TVWebhookPayload):
    """Receive TradingView alert webhooks and log them."""
    logger.info(
        "TV webhook: symbol=%s tf=%s action=%s price=%s",
        payload.symbol, payload.timeframe, payload.action, payload.price
    )
    return {"status": "received", "symbol": payload.symbol, "timeframe": payload.timeframe}


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
