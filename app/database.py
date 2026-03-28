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


def _col_type_sql(col, dialect_name: str) -> str:
    """Return a SQL type string for a SQLAlchemy column, adapted for the target dialect."""
    from sqlalchemy import Integer, String, Float, Boolean, DateTime, Text

    col_type = type(col.type)

    if col_type is Integer or issubclass(col_type, Integer):
        return "INTEGER"
    if col_type is Float or issubclass(col_type, Float):
        return "DOUBLE PRECISION" if dialect_name == "postgresql" else "REAL"
    if col_type is Boolean or issubclass(col_type, Boolean):
        return "BOOLEAN" if dialect_name == "postgresql" else "INTEGER"
    if col_type is DateTime or issubclass(col_type, DateTime):
        return "TIMESTAMP" if dialect_name == "postgresql" else "DATETIME"
    if col_type is Text or issubclass(col_type, Text):
        return "TEXT"
    # String — may carry a length attribute
    if col_type is String or issubclass(col_type, String):
        length = getattr(col.type, "length", None)
        if length and dialect_name != "sqlite":
            return f"VARCHAR({length})"
        return "TEXT" if dialect_name == "sqlite" else "VARCHAR(255)"
    # Fallback: let SQLAlchemy compile the type
    return str(col.type.compile())


def _col_default_sql(col, dialect_name: str) -> str:
    """Return a DEFAULT clause fragment (e.g. ' DEFAULT 0') or empty string."""
    from sqlalchemy import Integer, Boolean

    server_default = col.server_default
    col_default = col.default

    # Honour explicit server_default first
    if server_default is not None:
        clause = getattr(server_default, "clauses", None) or getattr(server_default, "arg", None)
        if clause is not None:
            return f" DEFAULT {clause}"

    if col_default is not None and col_default.is_scalar:
        val = col_default.arg
        col_type = type(col.type)
        if col_type is Boolean or issubclass(col_type, Boolean):
            if dialect_name == "sqlite":
                return " DEFAULT 1" if val else " DEFAULT 0"
            return " DEFAULT true" if val else " DEFAULT false"
        if col_type is Integer or issubclass(col_type, Integer):
            return f" DEFAULT {int(val)}"
        # Other scalar defaults
        return f" DEFAULT {val!r}"

    return ""


def run_migrations(engine) -> dict:
    """
    Inspect every table defined in the SQLAlchemy metadata and add any columns
    that exist in the model but are missing from the live database.

    Safe to run multiple times (idempotent).  Returns a summary dict.
    """
    from sqlalchemy import inspect, text
    from app.models import Base

    inspector = inspect(engine)
    dialect_name = engine.dialect.name
    existing_tables = set(inspector.get_table_names())

    added: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            # Table doesn't exist yet — create_all will handle it; skip
            skipped.append(f"{table_name} (table not yet created)")
            continue

        existing_cols = {row["name"] for row in inspector.get_columns(table_name)}

        for col in table.columns:
            if col.name in existing_cols:
                continue  # already present

            type_sql = _col_type_sql(col, dialect_name)
            default_sql = _col_default_sql(col, dialect_name)
            nullable_sql = "" if col.nullable else " NOT NULL"

            alter_sql = (
                f"ALTER TABLE {table_name} "
                f"ADD COLUMN {col.name} {type_sql}{nullable_sql}{default_sql}"
            )

            try:
                with engine.begin() as conn:
                    conn.execute(text(alter_sql))
                logger.info("Migration: added column %s.%s (%s)", table_name, col.name, type_sql)
                added.append(f"{table_name}.{col.name}")
            except Exception as exc:
                msg = f"{table_name}.{col.name}: {exc}"
                logger.error("Migration error — %s", msg)
                errors.append(msg)

    return {"added": added, "skipped": skipped, "errors": errors}


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
        return

    try:
        engine = get_engine()
        result = run_migrations(engine)
        if result["added"]:
            logger.info("Schema migration complete — added columns: %s", result["added"])
        if result["errors"]:
            logger.warning("Schema migration had errors: %s", result["errors"])
    except Exception as exc:
        logger.error("Schema migration failed (non-fatal): %s", exc)


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