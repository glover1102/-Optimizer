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


def _col_type_ddl(col, dialect_name: str) -> str:
    """Return a SQL type string suitable for ALTER TABLE ADD COLUMN."""
    from sqlalchemy import Integer, Float, Boolean, String, DateTime, Text

    col_type = col.type
    if isinstance(col_type, Integer):
        return "INTEGER"
    if isinstance(col_type, Float):
        return "DOUBLE PRECISION" if dialect_name == "postgresql" else "REAL"
    if isinstance(col_type, Boolean):
        return "BOOLEAN" if dialect_name == "postgresql" else "INTEGER"
    if isinstance(col_type, DateTime):
        return "TIMESTAMP" if dialect_name == "postgresql" else "DATETIME"
    if isinstance(col_type, (String, Text)):
        length = getattr(col_type, "length", None)
        if dialect_name == "postgresql":
            return f"VARCHAR({length})" if length else "TEXT"
        return f"VARCHAR({length})" if length else "TEXT"
    # Fallback — let the driver figure it out
    logger.warning("_col_type_ddl: unrecognised SQLAlchemy type %r — using str() fallback", col_type)
    return str(col_type.compile(dialect=None))


def run_migrations(engine=None) -> list[str]:
    """Inspect the live schema and ADD any columns that exist in the model but not in the DB.

    Returns a list of DDL statements that were executed.
    Only ADDs columns — never drops or alters existing ones.  Idempotent.
    """
    from sqlalchemy import inspect as sa_inspect, text
    from app.models import Base

    if engine is None:
        engine = get_engine()

    dialect_name = engine.dialect.name
    inspector = sa_inspect(engine)
    executed: list[str] = []

    with engine.connect() as conn:
        for table_name, table in Base.metadata.tables.items():
            # Skip tables that don't exist yet (create_all will handle them)
            if not inspector.has_table(table_name):
                continue

            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}

            for col in table.columns:
                if col.name in existing_cols:
                    continue

                col_ddl = _col_type_ddl(col, dialect_name)

                # Build DEFAULT clause
                default_clause = ""
                if col.default is not None and col.default.is_scalar:
                    val = col.default.arg
                    if isinstance(val, bool):
                        default_clause = " DEFAULT TRUE" if val else " DEFAULT FALSE"
                    elif isinstance(val, (int, float)):
                        default_clause = f" DEFAULT {val}"
                    elif isinstance(val, str):
                        safe_val = val.replace("'", "''")
                        default_clause = f" DEFAULT '{safe_val}'"
                elif not col.nullable:
                    # Non-nullable column with no explicit default — use a type-appropriate safe
                    # default so the ALTER TABLE succeeds against tables that already have rows.
                    from sqlalchemy import Integer, Float, Boolean, String, Text, DateTime
                    if isinstance(col.type, (Integer, Float, Boolean)):
                        default_clause = " DEFAULT 0"
                    elif isinstance(col.type, (String, Text)):
                        default_clause = " DEFAULT ''"
                    elif isinstance(col.type, DateTime):
                        default_clause = (
                            " DEFAULT CURRENT_TIMESTAMP"
                            if dialect_name == "postgresql"
                            else " DEFAULT CURRENT_TIMESTAMP"
                        )
                    else:
                        default_clause = " DEFAULT 0"

                null_clause = "" if col.nullable else " NOT NULL"

                ddl = (
                    f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_ddl}"
                    f"{default_clause}{null_clause}"
                )
                try:
                    conn.execute(text(ddl))
                    # Commit per-column intentionally: if a later column fails, the earlier ones
                    # are still persisted so the migration is as complete as possible.
                    conn.commit()
                    executed.append(ddl)
                    logger.info("Schema migration: added column — %s.%s (%s)", table_name, col.name, col_ddl)
                except Exception as exc:
                    logger.warning("Schema migration failed for %s.%s: %s", table_name, col.name, exc)
                    try:
                        conn.rollback()
                    except Exception as rb_exc:
                        logger.debug("Rollback failed after migration error: %s", rb_exc)

    return executed


def get_schema_status(engine=None) -> dict:
    """Return a dict describing the current schema vs model definitions.

    Includes, per table: existing column count, model column count, and any missing columns.
    """
    from sqlalchemy import inspect as sa_inspect
    from app.models import Base

    if engine is None:
        engine = get_engine()

    inspector = sa_inspect(engine)
    tables_info: dict = {}

    for table_name, table in Base.metadata.tables.items():
        if not inspector.has_table(table_name):
            tables_info[table_name] = {
                "exists": False,
                "db_columns": 0,
                "model_columns": len(list(table.columns)),
                "missing_columns": [c.name for c in table.columns],
            }
            continue

        existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
        model_cols = [c.name for c in table.columns]
        missing = [c for c in model_cols if c not in existing_cols]

        tables_info[table_name] = {
            "exists": True,
            "db_columns": len(existing_cols),
            "model_columns": len(model_cols),
            "missing_columns": missing,
        }

    return tables_info


def init_db() -> None:
    """Create all tables if they don't exist, then migrate any missing columns. Non-fatal."""
    global SessionLocal, _db_available
    try:
        from app.models import Base
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created / verified.")
    except Exception as exc:
        logger.error("Failed to create tables: %s", exc)

    # Auto-migrate: add any columns present in models but missing from the live schema
    try:
        executed = run_migrations()
        if executed:
            logger.info("Schema migration complete — %d column(s) added.", len(executed))
        else:
            logger.info("Schema up-to-date — no migrations needed.")
    except Exception as exc:
        logger.error("Schema migration failed (non-fatal): %s", exc)

    try:
        SessionLocal = get_session_factory()
        _db_available = True
    except Exception as exc:
        logger.error("Failed to initialise database session factory: %s", exc)
        logger.warning("App starting without database — will retry on next request.")


def reset_db() -> None:
    """Drop all tables and recreate them from scratch.

    WARNING: This destroys all data.  Only call programmatically for recovery.
    """
    from app.models import Base
    engine = get_engine()
    logger.warning("reset_db: dropping all tables — ALL DATA WILL BE LOST")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("reset_db: tables recreated successfully.")


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