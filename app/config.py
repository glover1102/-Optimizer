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
