from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class OptimizationResult(Base):
    __tablename__ = "optimization_results"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timeframe = Column(String, index=True, nullable=False)
    left_bars = Column(Integer)
    right_bars = Column(Integer)
    offset = Column(Float)
    atr_multiplier = Column(Float)
    atr_period = Column(Integer)
    win_rate = Column(Float)
    tp2_rate = Column(Float)
    tp3_rate = Column(Float)
    sl_rate = Column(Float)
    total_signals = Column(Integer)
    walk_forward_score = Column(Float)
    consistency_score = Column(Float)
    confidence_grade = Column(String(1))
    confidence_score = Column(Float)
    regime = Column(String)
    optimized_at = Column(DateTime, default=func.now())
    is_current = Column(Boolean, default=True)


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    symbols_processed = Column(Integer, default=0)
    status = Column(String, default="running")


class MarketRegime(Base):
    __tablename__ = "market_regimes"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timeframe = Column(String, nullable=False)
    regime = Column(String, nullable=False)
    adx = Column(Float)
    atr_ratio = Column(Float)
    bb_width = Column(Float)
    detected_at = Column(DateTime, default=func.now())


class SignalRecommendation(Base):
    __tablename__ = "signal_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timeframe = Column(String, index=True, nullable=False)
    action = Column(String, nullable=False)       # "BUY", "SELL", "HOLD"
    strength = Column(Integer)                    # 0-4
    entry_price = Column(Float)
    sl_price = Column(Float)
    tp1_price = Column(Float)
    tp2_price = Column(Float)
    tp3_price = Column(Float)
    regime = Column(String)
    entry_mode = Column(String)                   # "Pivot", "Crossover", "Hybrid"
    is_confluence = Column(Boolean, default=False)
    confidence = Column(Float)
    filters_used = Column(String)                 # JSON string of active filters
    created_at = Column(DateTime, default=func.now())
    is_current = Column(Boolean, default=True)
