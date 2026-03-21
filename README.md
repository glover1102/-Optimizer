# QTAlgo Optimizer

Server-side Bayesian parameter optimization engine for the **QTAlgo** TradingView indicator. Replaces Pine Script's limited 8-profile auto-tune with a full optimization engine that tests thousands of parameter combinations across all asset classes.

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────┐
│   TradingView       │     │   Railway Server           │     │  GitHub Repo │
│                     │     │                            │     │             │
│  QTAlgo Indicator   │◄────│  Python Optimization API   │◄────│  Source Code │
│  (reads best params)│     │                            │     │  Auto-deploy │
│                     │     │  - Fetch OHLCV data        │     │             │
│  Sends alerts ──────│────►│  - Run Bayesian search     │     └─────────────┘
└─────────────────────┘     │  - Walk-forward validation │
                             │  - Store best params       │
                             └──────────────────────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │  PostgreSQL / Redis   │
                             │  - Best params cache  │
                             │  - Historical results │
                             └──────────────────────┘
```

## What It Does

| Capability | Pine Script | This Server |
|---|---|---|
| Parameter combinations | ~8 profiles | 10,000+ per run |
| Walk-forward optimization | Not possible | Full rolling window |
| Regime detection | Basic | ADX + ATR + Bollinger |
| Continuous parameters | Fixed grid | Bayesian (Optuna TPE) |
| Data history | ~20,000 bars | Years of data |
| Asset classes | Single chart | All simultaneously |

## Watchlist

- **Crypto**: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT
- **Forex**: EURUSD=X, GBPUSD=X, USDJPY=X, GBPJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X, NZDUSD=X
- **Stocks**: AAPL, TSLA, NVDA, MSFT, AMZN, META, GOOGL, SPY, QQQ
- **Indices**: ^GSPC, ^NDX, ^DJI, ^RUT
- **Futures Micros**: MNQ=F, MES=F, MYM=F, M2K=F, MGC=F, MCL=F

## Local Development

### Prerequisites
- Docker + Docker Compose, OR Python 3.11+ with PostgreSQL and Redis

### With Docker Compose
```bash
git clone https://github.com/glover1102/-Optimizer.git
cd -Optimizer
cp .env.example .env
docker-compose up -d
# Open http://localhost:8000
```

### Without Docker
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your DATABASE_URL and REDIS_URL
uvicorn app.main:app --reload
```

## Railway Deployment

1. **Create project** at [railway.app](https://railway.app)
2. **Add PostgreSQL** plugin — Railway auto-sets `DATABASE_URL`
3. **Add Redis** plugin — Railway auto-sets `REDIS_URL`
4. **Connect GitHub** repo `glover1102/-Optimizer`
5. **Set environment variables** (optional overrides):
   - `OPTIMIZATION_INTERVAL_HOURS` (default: 6)
   - `DEFAULT_TRIALS` (default: 500)
   - `BINANCE_API_KEY` / `BINANCE_SECRET` (for crypto data, optional)
6. **Deploy** — Railway picks up `railway.toml` automatically

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Dark-themed dashboard |
| `GET` | `/results/{symbol}/{timeframe}` | Per-symbol detail view |
| `GET` | `/api/results` | All current results (JSON) |
| `GET` | `/api/results/{symbol}/{timeframe}` | Single result (JSON) |
| `POST` | `/api/optimize` | Trigger manual optimization |
| `POST` | `/api/webhook/tv` | Receive TradingView alert |
| `GET` | `/api/health` | Health check + scheduler status |

### Trigger Manual Optimization
```bash
curl -X POST http://localhost:8000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "timeframe": "4h", "trials": 500}'
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | SQLite (dev) | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL (optional caching) |
| `OPTIMIZATION_INTERVAL_HOURS` | `6` | How often the scheduler runs |
| `DEFAULT_TRIALS` | `500` | Optuna trials per symbol/timeframe |
| `BINANCE_API_KEY` | — | Optional, for higher Binance rate limits |

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## How Results Connect to TradingView

After optimization completes:
1. View best parameters on the dashboard at your Railway URL
2. Apply `left_bars`, `right_bars`, `offset`, `atr_multiplier`, `atr_period` to your QTAlgo chart
3. The regime label tells you whether these params were optimized for trending, ranging, or volatile conditions
4. Grade A/B results are statistically significant and walk-forward validated
