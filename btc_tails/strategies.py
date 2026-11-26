"""Trading-rule backtests on Kalshi KXBTC15M using the full-window quote candles (data/raw/kalshi_full/).

Conventions used everywhere here
- Minute j of a window opening at T is the Kalshi candle with end_period_ts = T + 60*j. Its close is the best YES
  bid/ask at the instant T + 60*j. The Binance candle with ts_open = T + 60*(j-1) closes at that same instant, so a
  decision "at minute j" only uses Binance prices up to T + 60*j and the Kalshi quote at T + 60*j.
- One contract per trade, at most one trade per market. Buying YES at the ask a pays y - a - fee(a); buying NO at
  1 - bid pays (1 - y) - (1 - b) - fee(1 - b) = b - y - fee(b).
- fee(P) = 0.07 * P * (1 - P) per contract: Kalshi's published general taker formula with the multiplier the API reports
  for this series (fee_type=quadratic, fee_multiplier=1). The schedule that applied in October 2026 is NOT verified,
  and per-order rounding up to the next cent is ignored (it matters for small orders).
- Fills assume the full contract trades at the quoted top-of-book price. Quote sizes are not in the public data.
"""
import json

import numpy as np
import pandas as pd

FEE_RATE = 0.07
MAX_J = 14          # last minute with a tradable quote before the settlement minute ends


def fee(p, rate=FEE_RATE):
    return rate * p * (1 - p)


def load_full_quotes(raw_dir):
    """Long table: one row per (market, minute j) with a two-sided quote."""
    rows = []
    for path in sorted(raw_dir.glob("KXBTC15M-*.json")):
        x = json.loads(path.read_text())
        m = x["market"]
        if not m or m.get("result") not in ("yes", "no"):
            continue
        T = x["window_open_utc"]
        y = 1.0 if m["result"] == "yes" else 0.0
        for c in x["candlesticks"]:
            j = (c["end_period_ts"] - T) // 60
            if not 1 <= j <= MAX_J:
                continue
            g = lambda d, k: float(d.get(k + "_dollars", d.get(k)) or np.nan)
            bid, ask = g(c["yes_bid"], "close"), g(c["yes_ask"], "close")
            if not (0 < bid < ask < 1):
                continue
            rows.append({"ticker": m["ticker"], "T": T, "j": int(j), "bid": bid, "ask": ask, "y": y,
                         "floor_strike": m.get("floor_strike"),
                         "volume": float(c.get("volume_fp", c.get("volume")) or 0)})
    q = pd.DataFrame(rows)
    q["mid"] = (q.bid + q.ask) / 2
    q["spread"] = q.ask - q.bid
    return q


def binance_state(minutes, T, j):
    """Binance quantities known at T + 60*j, plus the settlement proxies.

    S_j     close of the candle opening at T + 60*(j-1)
    K_proxy mean OHLC of the minute before the open (proxy for the 60-second BRTI opening average)
    end     mean OHLC of the last minute of the window (proxy for the settlement average)
    vol1m   root-mean-square of the last 60 one-minute log returns ending at S_j (NaN if any minute is missing)
    ret1m   last one-minute log return ending at S_j
    """
    g = minutes.set_index("ts_open")
    full = np.arange(g.index.min(), g.index.max() + 60, 60)
    g = g.reindex(full)
    typ = g[["open", "high", "low", "close"]].mean(axis=1)
    lr = np.log(g["close"]).diff()
    vol1m = np.sqrt((lr ** 2).rolling(60, min_periods=60).mean())
    T, j = np.asarray(T), np.asarray(j)
    last = T + 60 * (j - 1)
    return pd.DataFrame({
        "S_j": g["close"].reindex(last).to_numpy(),
        "K_proxy": typ.reindex(T - 60).to_numpy(),
        "end": typ.reindex(T + 840).to_numpy(),
        "vol1m": vol1m.reindex(last).to_numpy(),
        "ret1m": lr.reindex(last).to_numpy(),
    })


class FairValue:
    """P(settlement average >= opening average) given the Binance state at minute j.

    For each j the remaining move z = log(end / S_j) / (vol1m * sqrt(h_j)) is collected from the training windows,
    where h_j = 14.5 - j minutes is the time from T + 60*j to the middle of the settlement minute. The forecast is the
    empirical share of training z at or above the required standardized move. No bell curve is assumed, and the scale
    tracks the last hour's volatility.
    """

    def __init__(self, minutes, train_T):
        self.z = {}
        for j in range(1, MAX_J + 1):
            b = binance_state(minutes, train_T, np.full(len(train_T), j))
            z = np.log(b.end / b.S_j) / (b.vol1m * np.sqrt(14.5 - j))
            self.z[j] = np.sort(z.dropna().to_numpy())

    def prob(self, state, j):
        req = np.log(state.K_proxy / state.S_j) / (state.vol1m * np.sqrt(14.5 - np.asarray(j)))
        out = np.full(len(state), np.nan)
        for jj in np.unique(j):
            mask = (np.asarray(j) == jj) & req.notna().to_numpy()
            zs = self.z[jj]
            out[mask] = 1 - np.searchsorted(zs, req[mask].to_numpy(), side="left") / len(zs)
        return out


def edge_signals(df, p_col, theta):
    """+1 = buy YES, -1 = buy NO, 0 = no trade. Requires the edge after fees to exceed theta (in dollars)."""
    p = df[p_col].to_numpy()
    buy_yes = p - df.ask.to_numpy() - fee(df.ask.to_numpy()) > theta
    buy_no = df.bid.to_numpy() - p - fee(df.bid.to_numpy()) > theta
    return np.where(buy_yes, 1, np.where(buy_no, -1, 0))


def execute(df, signal, delay=0):
    """First signalled minute per market. delay=1 fills at the next minute's quote instead (a latency stress test)."""
    d = df.assign(signal=signal)
    first = d[d.signal != 0].sort_values(["ticker", "j"]).groupby("ticker", as_index=False).first()
    if delay:
        nxt = df[["ticker", "j", "bid", "ask"]].assign(j=lambda x: x.j - delay)
        first = first.drop(columns=["bid", "ask"]).merge(nxt, on=["ticker", "j"], how="inner")
    yes = first.signal == 1
    first["price"] = np.where(yes, first.ask, 1 - first.bid)
    first["fee"] = fee(first.price)
    first["pnl_gross"] = np.where(yes, first.y - first.ask, first.bid - first.y)
    first["pnl"] = first.pnl_gross - first.fee
    return first


def summarize(trades, n_boot=2000, seed=0):
    """Per-contract P&L in cents, with a 95% CI that resamples whole calendar days."""
    if len(trades) == 0:
        return {"trades": 0, "win rate": np.nan, "avg before fees (¢)": np.nan, "avg P&L (¢)": np.nan, "total ($)": 0.0,
                "95% CI low (¢)": np.nan, "95% CI high (¢)": np.nan}
    day = pd.to_datetime(trades["T"], unit="s", utc=True).dt.date
    per = trades.groupby(day)["pnl"].agg(["sum", "size"])
    s, n = per["sum"].to_numpy(), per["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return {"trades": len(trades), "win rate": float((trades.pnl > 0).mean()),
            "avg before fees (¢)": 100 * float(trades.pnl_gross.mean()), "avg P&L (¢)": 100 * float(trades.pnl.mean()), "total ($)": float(trades.pnl.sum()),
            "95% CI low (¢)": 100 * float(np.percentile(boots, 2.5)), "95% CI high (¢)": 100 * float(np.percentile(boots, 97.5))}
