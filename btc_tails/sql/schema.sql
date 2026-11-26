-- Research database schema (SQLite).
-- Time convention: ts_open = UTC epoch SECONDS of the candle's OPEN.
-- A 1-minute candle with ts_open = t covers [t, t+60) and its close is known at t+60.

-- Staging table: every raw row exactly as read from the archives (no key, so duplicates survive for the audit).
DROP TABLE IF EXISTS raw_candles;
CREATE TABLE raw_candles (
    venue      TEXT    NOT NULL,
    pair       TEXT    NOT NULL,
    ts_open    INTEGER NOT NULL,
    open       REAL    NOT NULL,
    high       REAL    NOT NULL,
    low        REAL    NOT NULL,
    close      REAL    NOT NULL,
    volume     REAL    NOT NULL,
    n_trades   INTEGER NOT NULL,
    source_file TEXT   NOT NULL
);

-- Normalized candles: one row per (venue, pair, minute).
DROP TABLE IF EXISTS candles;
CREATE TABLE candles (
    venue    TEXT    NOT NULL,
    pair     TEXT    NOT NULL,
    ts_open  INTEGER NOT NULL,
    open     REAL    NOT NULL,
    high     REAL    NOT NULL,
    low      REAL    NOT NULL,
    close    REAL    NOT NULL,
    volume   REAL    NOT NULL,
    n_trades INTEGER NOT NULL,
    PRIMARY KEY (venue, pair, ts_open)
);
