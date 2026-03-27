import os
from dotenv import load_dotenv

load_dotenv()

_raw_db_url = os.getenv("DATABASE_URL", "sqlite:///./qtalgo.db")
# Railway PostgreSQL uses postgres:// but SQLAlchemy needs postgresql://
if _raw_db_url.startswith("postgres://"):
    DATABASE_URL: str = _raw_db_url.replace("postgres://", "postgresql://", 1)
else:
    DATABASE_URL: str = _raw_db_url
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET: str = os.getenv("BINANCE_SECRET", "")
OPTIMIZATION_INTERVAL_HOURS: int = int(os.getenv("OPTIMIZATION_INTERVAL_HOURS", "6"))
DEFAULT_TRIALS: int = int(os.getenv("DEFAULT_TRIALS", "500"))
PORT: int = int(os.getenv("PORT", "8000"))

DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
OPTIMIZE_PASSCODE: str = os.getenv("OPTIMIZE_PASSCODE", "96541230")

# Signal generation config
SIGNAL_GENERATION_INTERVAL_MINUTES: int = int(os.getenv("SIGNAL_INTERVAL", "15"))
DEFAULT_ENTRY_MODE: str = os.getenv("DEFAULT_ENTRY_MODE", "Pivot")

DEFAULT_SIGNAL_PARAMS: dict = {
    "left_bars": 10,
    "right_bars": 10,
    "offset": 2.0,
    "atr_period": 14,
    "regime_length": 20,
    "use_regime_filter": True,
    "trend_threshold": 3.0,
    "volume_threshold": 2.0,
    "require_trend_alignment": True,
    "require_volume_confirmation": False,
    "use_rsi_filter": False,
    "rsi_length": 14,
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0,
    "use_wt_filter": False,
    "wt_channel_len": 10,
    "wt_avg_len": 21,
    "wt_ob_level": 60,
    "wt_os_level": -60,
    "use_ema_trend_filter": False,
    "ema_trend_period": 300,
    "use_golden_line": False,
    "gl_w1": 1.0,
    "gl_w2": 1.0,
    "gl_w3": 1.0,
    "gl_w4": 1.0,
    "gl_ema1_period": 9,
    "gl_ema2_period": 50,
    "use_price_position_filter": False,
    "entry_mode": "Pivot",
    "use_rr_targets": False,
    "rr_tp1": 1.0,
    "rr_tp2": 2.0,
    "rr_tp3": 3.0,
    "atr_multiplier": 0.8,
    "atr_target": 0.0,
    "gl_min_separation_atr": 0.5,
    "gl_cooldown_bars": 3,
    "gl_confirm_window": 2,
    "wt_require_cross": False,
}

WATCHLIST: dict[str, list[str]] = {
    "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"],
    "forex": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "GBPJPY=X",
        "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    ],
    "stocks": ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "SPY", "QQQ"],
    "indices": ["^GSPC", "^NDX", "^DJI", "^RUT"],
    "futures": ["MNQ=F", "MES=F", "MYM=F", "M2K=F", "MGC=F", "MCL=F"],
}

TIMEFRAMES: list[str] = ["5m", "15m", "1h", "4h", "1d"]

# Param search ranges
PARAM_RANGES: dict = {
    "left_bars": (3, 40),
    "right_bars": (3, 40),
    "offset": (0.5, 5.0),
    "atr_multiplier": (0.3, 3.0),
    "atr_period": (5, 50),
}
