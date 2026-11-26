-- Move staging rows into the keyed table.
-- * Exact duplicates (same key, identical OHLCV) collapse to one row.
-- * Conflicting duplicates (same key, different values) are NOT averaged: every row for that key is dropped,
--   so the minute becomes a gap and any 15-minute window touching it is excluded downstream.
INSERT INTO candles (venue, pair, ts_open, open, high, low, close, volume, n_trades)
SELECT venue, pair, ts_open, open, high, low, close, volume, n_trades
FROM (
    SELECT DISTINCT venue, pair, ts_open, open, high, low, close, volume, n_trades
    FROM raw_candles
)
WHERE (venue, pair, ts_open) IN (
    SELECT venue, pair, ts_open
    FROM (SELECT DISTINCT venue, pair, ts_open, open, high, low, close, volume, n_trades FROM raw_candles)
    GROUP BY venue, pair, ts_open
    HAVING COUNT(*) = 1
);
