import logging
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import DATABASE_URL
from app.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None
_db_available = False


def get_engine():
    """Lazily create and return the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
        )
    return _engine


def get_session_factory():
    """Lazily create and return the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


# Backward compat alias used by scheduler.py — set properly after init_db()
SessionLocal: Optional[sessionmaker] = None


def init_db() -> None:
    """Create all tables if they don't exist."""
    global SessionLocal, _db_available
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        SessionLocal = get_session_factory()
        _db_available = True
        logger.info("Database tables created / verified.")
    except Exception as exc:
        logger.error("Failed to initialise database: %s", exc)
        logger.warning("App starting without database — will retry on next request.")
        # Still set SessionLocal so callers don't crash on None
        try:
            SessionLocal = get_session_factory()
        except Exception as inner_exc:
            logger.error("Failed to create session factory after DB init failure: %s", inner_exc)


def is_db_available() -> bool:
    return _db_available


def get_db():
    """FastAPI dependency that yields a database session."""
    factory = get_session_factory()
    db: Session = factory()
    try:
        yield db
    finally:
        db.close()
