# Pricing 15-minute Bitcoin moves: fat tails, GARCH, and Kalshi

Kalshi runs a market every 15 minutes on one question: will BTC be higher at the end of the window than at the start? That's basically a 15-minute digital option. Black-Scholes would price it with a normal distribution and a constant volatility, and anyone who's watched BTC knows that's wrong. It sits still for a while and then jumps.

I wanted to know how wrong that assumption is at the 15-minute horizon, which volatility models actually fix it, and whether any of that is good enough to beat the people trading on Kalshi.

The full analysis is in [`notebooks/btc_tails.ipynb`](notebooks/btc_tails.ipynb). It's saved with outputs, so it can be read on GitHub without running anything.

## Data

- Every 1-minute BTCUSDT candle from Binance's public archive, May through October 2026 (264,960 minutes, checksums verified, no gaps or duplicates). Loaded into SQLite and cut into 15-minute windows aligned to :00/:15/:30/:45, the same as Kalshi's. That gives 17,647 usable windows.
- Kalshi's public KXBTC15M data for September and October: market terms, results, and 1-minute bid/ask snapshots.

Models are fit on May–August, every modeling choice is made on September, and October is used once at the end.

## What I found

### 15-minute returns are far from normal

A typical 15-minute move is about 20 bps (0.2%), and excess kurtosis is about 17. Compared with a normal distribution with the same variance:

- Moves past ±25 bps happened 2,295 times. A normal predicts about 3,700.
- Moves past ±50 bps happened 466 times, more than twice the 210 predicted.
- Moves past ±75 bps happened 154 times. A normal predicts about 3.

So the normal puts too much weight on medium moves and almost none on big ones. There's no reliable skew. It was positive in August, negative in October, and mostly disappears without the five biggest windows.

### Volatility models

Volatility clusters, so the obvious fix is to forecast σ for each window instead of using one number. I fit two standard models by maximum likelihood:

- EWMA (RiskMetrics): σ²ₜ = λσ²ₜ₋₁ + (1−λ)r²ₜ₋₁
- GARCH(1,1): σ²ₜ = ω + αr²ₜ₋₁ + βσ²ₜ₋₁

Each one comes with either normal or Student-t shocks. There's also an optional intraday seasonal adjustment, because BTC volatility is about twice its daily average right after the US stock market opens and about 0.6× in the quietest hours. Without the adjustment, GARCH reads that daily cycle as volatility shocks.

The forecast σ becomes a price through the Black-Scholes digital formula P(S_T > K) = N(d₂), or the Student-t CDF in place of N for the t models.

Some things that came out of the fits:

- Student-t shocks beat normal shocks for every model, and by more than anything else. The fitted degrees of freedom is ν ≈ 4.5, which is extremely fat-tailed. Even after dividing each October return by its GARCH forecast σ, the result still has excess kurtosis of about 16. Volatility clustering explains part of the fat tails, but the shocks themselves aren't normal.
- GARCH persistence α+β goes to the stationarity boundary. I checked by fixing persistence at values from 0.95 to 0.99999 and refitting, and the likelihood keeps rising toward 1. So volatility shocks barely decay over these few months, which is why EWMA (the special case α+β = 1, ω = 0) does almost as well.
- Removing the time-of-day pattern helped every model.

### October results

Log loss on 14,880 window-strike pairs (2,976 October windows × 5 strikes), lower is better:

| model | log loss | vs. normal |
|---|---|---|
| Normal, constant σ (Black-Scholes) | 0.2482 | |
| Empirical distribution | 0.2399 | −3.3% |
| GARCH(1,1), normal shocks | 0.2359 | −5.0% |
| Small PyTorch net predicting μ and σ | 0.2335 | −5.9% |
| Logistic regression on recent realized vol | 0.2327 | −6.3% |
| GARCH(1,1), Student-t, seasonal adjustment | 0.2302 | −7.3% |

The GARCH-t model was picked on September before I looked at October. It beat the neural net on log loss, with a 95% confidence interval (resampling whole days) that excludes zero, and it tied the logistic regression. So a GARCH model with 4 parameters (plus the time-of-day scales) does as well as the machine learning ones, and every parameter means something.

### The market was still better

On October's Kalshi markets, one minute after the open (the first moment a quote exists), the market's own price had lower log loss than every model, by more than chance would explain. The best model there was 0.6626 and GARCH-t was 0.6665, against the market's 0.6572. Kalshi traders already price BTC's volatility at least as well as GARCH does.

### Trying to trade it

I tested four strategies, picking every rule on September and testing once on October, with Kalshi's fee formula and one contract per trade:

| strategy | Sep P&L | Oct trades | Oct P&L per contract |
|---|---|---|---|
| S1: trade when the GARCH-t price disagrees with the market at the open | +$8.42 | 220 | −5.0¢ |
| S2: live fair value each minute from Binance price, time left, and recent vol | +$11.64 | 2,095 | −2.5¢ |
| S3: price ranges that looked mispriced in September | +$30.60 | 719 | −1.2¢ |
| S4: logistic regression on top of the market price | −$9.26 (wouldn't trade) | 432 | −4.2¢ |

All four lost money in October. The September profits were overfitting. S2 lost even before fees, and the market's price was more accurate than my live fair value at every minute of the window. These results also assume every order fills at the quoted price, so real results would be worse.

## Takeaways

- The normal distribution is wrong for BTC at this horizon in a specific way: too few small moves, far too few large ones.
- Fixing it takes three things: time-varying volatility (GARCH/EWMA), the intraday volatility cycle, and fat-tailed shocks (Student-t). Together that's about a 7% improvement in log loss. That's a real improvement, but a small one.
- None of that beats the market. Finding an edge would probably take data this project doesn't have: the order book, confirmed historical fees, and prices faster than one snapshot per minute.

Things I'd try next: a GJR-GARCH or realized-GARCH model that uses the 1-minute returns inside each window, and comparing GARCH's σ to implied volatility from Deribit BTC options.

## Code

```
btc_tails/
  download_binance.py   Binance 1-minute archives, SHA-256 verified
  build_db.py, sql/     SQLite schema, data checks, 15-minute windows in SQL
  features.py           realized vol and lagged returns, train/val/test split
  models.py             normal, empirical, logistic regression, PyTorch net
  vol.py                EWMA and GARCH(1,1) by maximum likelihood, Black-Scholes digital pricing
  evaluate.py           Brier score, log loss, calibration, day-block bootstrap
  kalshi.py             Kalshi public market data downloader
  strategies.py         fee model, fair value, backtest execution
tests/                  data-pipeline checks and volatility model tests
```

## Running it

Python 3.14:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest -q tests                                  # no data needed
.venv/bin/python -m btc_tails.download_binance                       # ~12 MB
.venv/bin/python -m btc_tails.build_db                               # builds data/btc.sqlite
.venv/bin/python -m btc_tails.kalshi 2026-10-01 2026-11-01           # ~1.5 hours, Kalshi rate-limits
.venv/bin/python -m btc_tails.kalshi 2026-09-01 2026-11-01 --full    # full-window quotes, ~15 minutes
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/btc_tails.ipynb
```

Downloads are cached in `data/` (ignored by git), so reruns are quick. In VS Code or Jupyter, pick `.venv/bin/python` as the kernel.
