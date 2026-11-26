# Do Bitcoin Return Tails Improve Short-Horizon Event Pricing?

Does using the actual distribution of 15-minute BTC log returns give better out-of-sample probabilities that BTC finishes above a level $K$ than a fitted normal? Does conditioning on recent volatility help more? And how do these probabilities compare with Kalshi's 15-minute BTC market prices on the same windows?

**Main deliverable:** [`notebooks/btc_tails.ipynb`](notebooks/btc_tails.ipynb), committed with its executed outputs.

## Data

| | |
|---|---|
| Spot venue / pair | Binance spot, `BTCUSDT`, 1-minute klines |
| Source | Monthly bulk archives at `data.binance.vision` ([docs](https://github.com/binance/binance-public-data)), SHA-256 checked against Binance's `.CHECKSUM` files |
| Window | 2026-05-01 00:00 to 2026-10-31 23:59 UTC (six complete months; November was not yet archived on 2026-11-26) |
| Timestamps | `open_time` = candle **open**, in microseconds (spot files from 2025 on). Stored as UTC epoch seconds in `ts_open`. A candle closes at `ts_open + 60s`. |
| Prediction market | Kalshi series `KXBTC15M` via the public REST API (no key). Test month October 2026: metadata plus 1-minute YES bid/ask/trade candles for each market |

Raw files and the SQLite database are cached under `data/` (ignored by Git). Set `BTC_TAILS_DATA` to use another directory.

## Reproduce

Local (Python 3.14, project venv):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # on Linux: install torch from the CPU index first (see Dockerfile)
.venv/bin/python -m pytest -q tests                # fixture checks, no data needed
.venv/bin/python -m btc_tails.download_binance     # 6 zip files, ~12 MB total, checksum-verified
.venv/bin/python -m btc_tails.build_db             # data/btc.sqlite
.venv/bin/python -m btc_tails.kalshi 2026-10-01 2026-11-01   # 2,976 markets; rate-limited; took about 1.5 hours here
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/btc_tails.ipynb
```

Docker (CPU only; `data/` mounted so downloads are cached):

```bash
docker build -t btc-tails .
docker run --rm -v "$PWD/data:/work/data" -v "$PWD/notebooks:/work/notebooks" btc-tails
```

All downloads skip files that are already cached, so re-runs are fast.

## Repository layout

```
notebooks/btc_tails.ipynb   the research notebook (one thread: data -> distribution -> forecasts -> Kalshi)
btc_tails/
  download_binance.py       bulk archive download + checksum
  build_db.py               load CSVs into SQLite (staging table -> keyed table)
  sql/schema.sql            tables; PRIMARY KEY (venue, pair, ts_open)
  sql/normalize.sql         collapse exact duplicates, drop conflicting ones (never average)
  sql/audit.sql             duplicate keys, misaligned rows, timestamp gaps
  sql/windows.sql           15-minute windows: start/end closes and a 16-candle completeness check
  features.py               trailing features, strike grid, purged chronological split
  models.py                 normal, empirical, logistic (sklearn Pipeline), PyTorch conditional Gaussian
  evaluate.py               Brier, log loss, reliability bins, monotonicity, day-block bootstrap
  kalshi.py                 public Kalshi API downloader (live + historical endpoints)
tests/test_checks.py        hand-built fixture: known return, missing minute, strict threshold, purged split boundary
```

What each tool does here: **SQLite/SQL** stores the candles and does the audit and window extraction. **pandas** handles features and tables. **NumPy/SciPy** compute moments and the normal model. **Matplotlib** draws the figures. **scikit-learn** fits the logistic model. **PyTorch** fits the conditional Gaussian. **pytest** runs the fixture checks. **Docker** gives a clean rerun. **Git** keeps the history.

## Implementation plan (as carried out)

1. Verify sources: Binance archive docs, coverage, schema and timestamp units; Kalshi `KXBTC15M` contract terms, historical endpoints and fee information.
2. Download six months of 1m candles. Load them into SQLite through a staging table and audit duplicates, alignment and gaps in SQL.
3. Build non-overlapping 15-minute windows aligned to UTC (decision time = window start). Drop any window that is missing a candle.
4. Describe the distribution: moments, histogram, Q–Q, tail exceedances, monthly subperiods and extreme windows.
5. Run the forecasting experiment: strikes at −50/−25/0/+25/+50 log-bps. Train May–Aug, validate Sep, test Oct. Fit normal and empirical baselines first, then logistic and PyTorch.
6. Score on the same test rows: Brier, log loss, per strike, reliability, monotonicity, and day-block bootstrap intervals.
7. Kalshi: align at open+1 min, compare the market mid and the models against actual outcomes, and record what data is missing for a P&L test.
8. Verify: fixture tests, a fresh-kernel execution, and a check that README numbers match the notebook.

## Results (from the executed notebook)

All numbers below are copied from the executed notebook. The section it comes from is in brackets.

**Data audit [§1].** 264,960 one-minute candles kept out of 264,960 read (= 184 days × 1,440). 0 duplicate keys, 0 gaps. 17,664 calendar windows; 1 dropped for a missing candle (the first window's start candle is on 2026-04-30) and 16 for feature warm-up, leaving 17,647.

**Distribution [§2].** 15-minute log returns: sd 19.88 bps, skewness 0.160, excess kurtosis 16.95. Monthly skew ranges from −0.41 (Oct) to +1.54 (Aug); without the 5 largest |r| windows, skew is 0.086 (labeled sensitivity check only). Moves beyond ±25 bps happen *less* often than the fitted normal implies (1,150 up vs 1,841 expected). Moves beyond ±50 bps happen *more* often (241 vs 105).

**Splits [§3].** Train 2026-05-01 04:15 to 2026-08-31 23:45 UTC (11,791 windows); validation September (2,880); test October (2,976 windows × 5 strikes = 14,880 rows). No rows needed purging.

**Held-out spot experiment [§4–5]**, test month, same 14,880 rows for every model:

| model | Brier | log loss | log loss vs normal (95% day-block CI) |
|---|---|---|---|
| A. normal | 0.07407 | 0.24823 | – |
| B. empirical | 0.07255 | 0.23992 | −0.0083 [−0.0103, −0.0063] |
| C. logistic (interaction spec, chosen on validation) | 0.07182 | 0.23266 | −0.0156 |
| D. PyTorch conditional Gaussian | 0.07190 | 0.23351 | −0.0147 [−0.0200, −0.0098] |

The PyTorch settings were revised once after a first run's test scores had been seen. The draft hit its epoch cap; the disclosure is in §5 and the test log loss moved from 0.2339 to 0.2335. The empirical model's gain is concentrated at ±25 bps. At ±50 bps it is slightly worse than the normal and the CI crosses zero. Logistic beat empirical by −0.0073 [−0.0095, −0.0054] in log loss. No model produced a monotonicity violation.

**Kalshi KXBTC15M, October 2026 [§6].** 2,936 of 2,976 windows had a market. The 40 missing windows all fall in Kalshi's documented Thursday 3–5 AM ET maintenance. All 2,936 had a two-sided quote at open+1 min. The Binance proxy reproduces the actual outcome in 96.5% of markets. On the same 2,936 markets, log loss was: market mid 0.6572, PyTorch (horizon-rescaled) 0.6626, normal 0.6671, empirical 0.6690. Every model was worse than the market mid, and each model − market CI excludes zero. **No P&L was computed** (see below).

## What was not verified

- **Docker:** not installed on the build machine (`docker: command not found`), so the image was neither built nor smoke-tested. The Dockerfile has not been run.
- **Kalshi fee schedule:** `https://kalshi.com/docs/kalshi-fee-schedule.pdf` returned HTTP 429 to automated requests. The API reports the current `fee_type=quadratic`, `fee_multiplier=1` and an empty fee-change history, which does not prove what applied in October 2026.
- **Executable prices:** Kalshi's historical 1-minute candles give the best YES bid and ask but no size, so fills at the ask cannot be shown. Combined with the unverified fees, the repo makes no profitability claim and does not test market making.
- **Linux torch wheel:** the pinned versions were installed and run on macOS arm64 only.

## How this was built

The code, notebook text and this README were drafted with an AI coding assistant (Claude Code), working from my written specification. All numbers come from code that runs against the downloaded data. The interpretation sections in the notebook are left blank for me to write.
