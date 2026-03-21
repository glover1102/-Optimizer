import logging
from typing import Optional

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None
_db_available = False

def _get_database_url():
    from app.config import DATABASE_URL
    return DATABASE_URL

def get_engine():
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
    global _session_factory
    if _session_factory is None:
        from sqlalchemy.orm import sessionmaker
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _session_factory

# Backward compat — used by scheduler.py
class _SessionLocalProxy:
    def __call__(self):
        return get_session_factory()()

SessionLocal = _SessionLocalProxy()

def init_db() -> None:
    global _db_available
    try:
        from app.models import Base
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        _db_available = True
        logger.info("Database tables created / verified.")
    except Exception as exc:
        logger.error("Failed to initialise database: %s", exc)
        logger.warning("App starting without database — will retry on next request.")

def is_db_available() -> bool:
    return _db_available

def get_db():
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()