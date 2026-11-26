"""Small helper package for the BTC return-tails notebook."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BTC_TAILS_DATA", ROOT / "data"))
RAW_BINANCE = DATA_DIR / "raw" / "binance"
RAW_KALSHI = DATA_DIR / "raw" / "kalshi"
DB_PATH = DATA_DIR / "btc.sqlite"
SQL_DIR = Path(__file__).resolve().parent / "sql"

VENUE = "binance"
PAIR = "BTCUSDT"
# Fixed research window: six complete calendar months (UTC).
MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
