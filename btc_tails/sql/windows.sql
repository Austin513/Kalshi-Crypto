-- name: windows
-- One row per 15-minute UTC window [T, T+900), T on :00/:15/:30/:45.
--   decision time  = T
--   start price S  = close of the candle with ts_open = T-60   (that candle completes exactly at T)
--   end price      = close of the candle with ts_open = T+840  (completes at T+900)
-- A window is valid only if all 16 candles T-60, T, T+60, ..., T+840 exist. No filling.
WITH RECURSIVE grid(T) AS (           -- full calendar of windows, so windows with zero candles still appear
    SELECT :t_min
    UNION ALL
    SELECT T + 900 FROM grid WHERE T + 900 < :t_max
),
agg AS (
    SELECT g.T,
           COUNT(c.ts_open)                                          AS n_candles,
           MAX(CASE WHEN c.ts_open = g.T - 60  THEN c.close END)     AS s_start,
           MAX(CASE WHEN c.ts_open = g.T + 840 THEN c.close END)     AS s_end
    FROM grid g
    LEFT JOIN candles c
      ON c.venue = :venue AND c.pair = :pair
     AND c.ts_open BETWEEN g.T - 60 AND g.T + 840
    GROUP BY g.T
)
SELECT T AS t_decision,
       T + 900 AS t_label,
       n_candles,
       s_start,
       s_end,
       CASE WHEN n_candles = 16 THEN 1 ELSE 0 END AS is_valid
FROM agg
ORDER BY T;

-- name: minute_series
-- Ordered close series for feature construction (trailing windows are built in pandas, without filling gaps).
SELECT ts_open, open, high, low, close
FROM candles
WHERE venue = :venue AND pair = :pair
ORDER BY ts_open;
