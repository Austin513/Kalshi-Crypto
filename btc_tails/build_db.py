"""Load the cached Binance archives into SQLite and normalize them.

    python -m btc_tails.build_db

Binance kline CSVs have no header. Columns (per the binance-public-data README):
open_time, open, high, low, close, volume, close_time, quote_volume, n_trades,
taker_buy_base, taker_buy_quote, ignore.
Spot timestamps from 2025-01-01 onward are in MICROseconds; earlier files use milliseconds.
`open_time` is the candle OPEN. We store ts_open in UTC epoch seconds.
"""
import re
import sqlite3
import zipfile

import pandas as pd

from . import DB_PATH, PAIR, RAW_BINANCE, SQL_DIR, VENUE

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def read_sql_blocks(name):
    """Return {block_name: sql} for files using `-- name: x` headers."""
    text = (SQL_DIR / name).read_text()
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def to_epoch_seconds(open_time):
    """Binance open_time -> epoch seconds. Microseconds have 16 digits, milliseconds 13."""
    s = pd.Series(open_time, dtype="int64")
    us = s >= 10**15
    return (s.where(~us, s // 1_000_000).where(us, s // 1_000)).astype("int64")


def read_archive(path):
    with zipfile.ZipFile(path) as z:
        (member,) = z.namelist()
        with z.open(member) as f:
            df = pd.read_csv(f, header=None, names=COLS)
    out = pd.DataFrame({
        "venue": VENUE, "pair": PAIR,
        "ts_open": to_epoch_seconds(df["open_time"]),
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "volume": df["volume"], "n_trades": df["n_trades"].astype("int64"),
        "source_file": path.name,
    })
    return out


def build(db_path=DB_PATH, raw_dir=RAW_BINANCE):
    files = sorted(raw_dir.glob(f"{PAIR}-1m-*.zip"))
    if not files:
        raise SystemExit(f"No archives in {raw_dir}; run `python -m btc_tails.download_binance` first.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript((SQL_DIR / "schema.sql").read_text())
    for p in files:
        df = read_archive(p)
        df.to_sql("raw_candles", con, if_exists="append", index=False)
        print(f"loaded {p.name}: {len(df):,} rows")
    con.executescript((SQL_DIR / "normalize.sql").read_text())
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    print(f"candles table: {n:,} rows -> {db_path}")
    con.close()


if __name__ == "__main__":
    build()
