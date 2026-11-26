-- name: row_counts
-- Raw rows read vs. normalized rows kept.
SELECT
    (SELECT COUNT(*) FROM raw_candles)                                           AS raw_rows,
    (SELECT COUNT(*) FROM candles)                                               AS kept_rows,
    (SELECT MIN(ts_open) FROM candles)                                           AS first_ts_open,
    (SELECT MAX(ts_open) FROM candles)                                           AS last_ts_open;

-- name: duplicate_keys
-- Keys that appear more than once in the raw data, split into exact copies vs. conflicting values.
SELECT
    COUNT(*)                                        AS duplicated_keys,
    SUM(CASE WHEN n_distinct = 1 THEN 1 ELSE 0 END) AS exact_duplicate_keys,
    SUM(CASE WHEN n_distinct > 1 THEN 1 ELSE 0 END) AS conflicting_keys
FROM (
    SELECT venue, pair, ts_open,
           COUNT(*) AS n_rows,
           COUNT(DISTINCT open || '|' || high || '|' || low || '|' || close || '|' || volume) AS n_distinct
    FROM raw_candles
    GROUP BY venue, pair, ts_open
    HAVING COUNT(*) > 1
);

-- name: misaligned_rows
-- Rows whose open time is not on a whole UTC minute.
SELECT COUNT(*) AS misaligned_rows FROM candles WHERE ts_open % 60 <> 0;

-- name: gaps
-- Every place where the next candle is not exactly 60 seconds after the previous one.
SELECT
    ts_open                       AS gap_after_ts_open,
    next_ts_open,
    (next_ts_open - ts_open) / 60 - 1 AS missing_minutes
FROM (
    SELECT ts_open, LEAD(ts_open) OVER (PARTITION BY venue, pair ORDER BY ts_open) AS next_ts_open
    FROM candles
    WHERE venue = :venue AND pair = :pair
)
WHERE next_ts_open IS NOT NULL AND next_ts_open - ts_open <> 60
ORDER BY ts_open;
