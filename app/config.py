import os
from dotenv import load_dotenv

load_dotenv()

_raw_db_url = os.getenv("DATABASE_URL", "sqlite:///./qtalgo.db")
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
DISCORD_WEBHOOK_OPTIMIZER: str = os.getenv("DISCORD_WEBHOOK_OPTIMIZER", "")
OPTIMIZE_PASSCODE: str = os.getenv("OPTIMIZE_PASSCODE", "96541230")

PUSHOVER_USER_KEY: str = os.getenv("PUSHOVER_USER_KEY", "")
PUSHOVER_API_TOKEN: str = os.getenv("PUSHOVER_API_TOKEN", "")
PUSHOVER_ENABLED: bool = os.getenv("PUSHOVER_ENABLED", "true").lower() in ("true", "1", "yes")

SIGNAL_GENERATION_INTERVAL_MINUTES: int = int(os.getenv("SIGNAL_INTERVAL", "15"))
DEFAULT_ENTRY_MODE: str = os.getenv("DEFAULT_ENTRY_MODE", "KeyLevel")

DEFAULT_SIGNAL_PARAMS: dict = {
    "enable_pdl": True,
    "enable_pdo": True,
    "enable_pdc": True,
    "enable_pdh": True,
    "enable_pwh": False,
    "enable_pwl": False,
    "enable_open": False,
    "or1_enabled": False,
    "or1_minutes": 5,
    "or1_ext_enabled": False,
    "or2_enabled": False,
    "or2_minutes": 360,
    "or2_ext_enabled": True,
    "or3_enabled": False,
    "or3_minutes": 480,
    "or3_ext_enabled": True,
    "or4_enabled": True,
    "or4_minutes": 720,
    "or4_ext_enabled": True,
    "or_ext_0_5_enabled": False,
    "or_ext_1_0_enabled": False,
    "or_ext_1_5_enabled": True,
    "or_ext_2_0_enabled": True,
    "or_as_trigger": True,
    "or_ext_as_trigger": True,
    "re_arm": True,
    "entry_at_level": False,
    "atr_length": 14,
    "atr_smoothing": "RMA",
    "sl_mult": 1.5,
    "enable_tp1": True,
    "enable_tp2": True,
    "enable_tp3": True,
    "enable_tp4": True,
    "tp1_rr": 0.75,
    "tp2_rr": 1.5,
    "tp3_rr": 2.25,
    "tp4_rr": 3.0,
    "be_after_tp": "TP1",
    "be_off_ticks": 0.0,
    "runner_tgt": "TP4",
    "clear_on_tp": False,
    "clear_tp_sel": "TP1",
    "resolve_mode": "First touch",
    "filt_mode": "Strict",
    "use_structure_15m": True,
    "use_structure_1h": False,
    "use_structure_4h": False,
    "use_fmom": True,
    "fmom_strict_fan": False,
    "ema_9_enabled": True,
    "ema_21_enabled": True,
    "ema_50_enabled": True,
    "ema_100_enabled": False,
    "ema_200_enabled": False,
    "ema_300_enabled": False,
    "ema_9": 9,
    "ema_21": 21,
    "ema_50": 50,
    "ema_100": 100,
    "ema_200": 200,
    "ema_300": 300,
    "use_fvol": True,
    "fvol_len": 20,
    "fvol_mult": 1.0,
    "fvol_ma_type": "SMA",
    "use_frsi": True,
    "rsi_length": 14,
    "rsi_neutral": 5.0,
    "use_fmacd": True,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "use_fvwap": True,
    "use_fatr": False,
    "fatr_baseline": 20,
    "fatr_allow": "Only Stable",
    "entry_mode": DEFAULT_ENTRY_MODE,
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

PARAM_RANGES: dict = {
    "atr_length": (7, 28),
    "sl_mult": (1.0, 2.5),
    "tp1_rr": (0.5, 1.25),
    "tp2_rr": (1.0, 2.0),
    "tp3_rr": (1.5, 3.0),
    "tp4_rr": (2.0, 4.0),
    "or2_minutes": (120, 480),
    "or3_minutes": (240, 720),
    "or4_minutes": (480, 1440),
    "fvol_mult": (0.8, 1.5),
    "rsi_neutral": (2.0, 10.0),
    "categorical": {
        "atr_smoothing": ["RMA", "SMA", "EMA", "WMA"],
        "be_after_tp": ["TP1", "TP2", "TP3", "TP4"],
        "runner_tgt": ["TP4", "Last enabled TP"],
        "resolve_mode": ["First touch", "Close"],
        "filt_mode": ["Strict", "Not against"],
        "fvol_ma_type": ["SMA", "EMA", "WMA", "RMA"],
        "fatr_allow": ["Only Stable", "Stable or Low", "Stable or High"],
    },
}
