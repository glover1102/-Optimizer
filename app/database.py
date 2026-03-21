import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None
_db_available = False

# Backward compat alias — set to a real sessionmaker after a successful init_db()
SessionLocal: Optional["sessionmaker"] = None


def _get_database_url() -> str:
    from app.config import DATABASE_URL
    return DATABASE_URL


def get_engine():
    """Lazily create and return the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        db_url = _get_database_url()
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
        )
    return _engine


def get_session_factory():
    """Lazily create and return the session factory."""
    global _session_factory
    if _session_factory is None:
        from sqlalchemy.orm import sessionmaker
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _session_factory


def init_db() -> None:
    """Create all tables if they don't exist. Non-fatal — logs errors and continues."""
    global SessionLocal, _db_available
    try:
        from app.models import Base
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        SessionLocal = get_session_factory()
        _db_available = True
        logger.info("Database tables created / verified.")
    except Exception as exc:
        logger.error("Failed to initialise database: %s", exc)
        logger.warning("App starting without database — will retry on next request.")
        # Attempt to set up SessionLocal anyway so callers don't crash on None
        try:
            SessionLocal = get_session_factory()
        except Exception as inner_exc:
            logger.error("Failed to create session factory after DB init failure: %s", inner_exc)


def is_db_available() -> bool:
    return _db_available


def get_db():
    """FastAPI dependency that yields a database session."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()