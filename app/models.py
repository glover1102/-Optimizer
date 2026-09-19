from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class OptimizationResult(Base):
    __tablename__ = "optimization_results"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timeframe = Column(String, index=True, nullable=False)

    enable_pdl = Column(Boolean)
    enable_pdo = Column(Boolean)
    enable_pdc = Column(Boolean)
    enable_pdh = Column(Boolean)
    enable_pwh = Column(Boolean)
    enable_pwl = Column(Boolean)
    enable_open = Column(Boolean)

    or1_enabled = Column(Boolean)
    or1_minutes = Column(Integer)
    or1_ext_enabled = Column(Boolean)
    or2_enabled = Column(Boolean)
    or2_minutes = Column(Integer)
    or2_ext_enabled = Column(Boolean)
    or3_enabled = Column(Boolean)
    or3_minutes = Column(Integer)
    or3_ext_enabled = Column(Boolean)
    or4_enabled = Column(Boolean)
    or4_minutes = Column(Integer)
    or4_ext_enabled = Column(Boolean)

    atr_length = Column(Integer)
    atr_smoothing = Column(String)
    sl_mult = Column(Float)
    tp1_rr = Column(Float)
    tp2_rr = Column(Float)
    tp3_rr = Column(Float)
    tp4_rr = Column(Float)
    be_after_tp = Column(String)
    be_off_ticks = Column(Float)
    runner_tgt = Column(String)
    clear_on_tp = Column(Boolean)
    clear_tp_sel = Column(String)
    resolve_mode = Column(String)
    filt_mode = Column(String)
    entry_at_level = Column(Boolean)
    re_arm = Column(Boolean)

    use_structure_15m = Column(Boolean)
    use_structure_1h = Column(Boolean)
    use_structure_4h = Column(Boolean)
    use_fmom = Column(Boolean)
    use_fvol = Column(Boolean)
    use_frsi = Column(Boolean)
    use_fmacd = Column(Boolean)
    use_fvwap = Column(Boolean)
    use_fatr = Column(Boolean)
    fvol_len = Column(Integer)
    fvol_mult = Column(Float)
    fvol_ma_type = Column(String)
    rsi_length = Column(Integer)
    rsi_neutral = Column(Float)
    fatr_baseline = Column(Integer)
    fatr_allow = Column(String)

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


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    status = Column(String, default="queued")
    symbols = Column(Text, nullable=False)
    timeframe = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    objective = Column(String, default="risk_adjusted")
    auto_mode = Column(Boolean, default=False)
    n_trials = Column(Integer, default=50)
    min_trades = Column(Integer, default=10)
    swept_params = Column(Text, default="[]")
    locked_params = Column(Text, default="{}")
    progress_current = Column(Integer, default=0)
    progress_total = Column(Integer, default=0)
    current_symbol = Column(String, nullable=True)
    best_value = Column(Float, nullable=True)
    results = Column(Text, default="{}")
    error = Column(Text, nullable=True)
    cancel_requested = Column(Boolean, default=False)


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
    action = Column(String, nullable=False)
    strength = Column(Integer)
    entry_price = Column(Float)
    sl_price = Column(Float)
    tp1_price = Column(Float)
    tp2_price = Column(Float)
    tp3_price = Column(Float)
    tp4_price = Column(Float)
    regime = Column(String)
    entry_mode = Column(String)
    is_confluence = Column(Boolean, default=False)
    confidence = Column(Float)
    filters_used = Column(String)
    created_at = Column(DateTime, default=func.now())
    is_current = Column(Boolean, default=True)

    outcome = Column(String, nullable=True)
    outcome_at = Column(DateTime, nullable=True)
    outcome_price = Column(Float, nullable=True)
    highest_tp_hit = Column(Integer, default=0)
    pnl_percent = Column(Float, nullable=True)
