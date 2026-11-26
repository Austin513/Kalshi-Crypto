"""Windows, trailing features, strikes, and chronological splits.

Conventions
-----------
* All times are UTC epoch seconds. A candle with ts_open = t closes (is known) at t + 60.
* Decision time T is the start of a 15-minute window. Start price S = close of candle T-60.
* Label: r = log(S_end / S) with S_end = close of candle T+840 (known at T+900).
* Features at T may only use candles with ts_open <= T-60 (i.e. completed by T).
* Strike offsets are in LOG-return basis points: K = S * exp(k / 10_000). Event: S_end > K  <=>  r > k / 10_000.
"""
import sqlite3

import numpy as np
import pandas as pd

from . import DB_PATH, PAIR, VENUE
from .build_db import read_sql_blocks

STRIKES_BPS = [-50, -25, 0, 25, 50]
BPS = 1e4


def load_windows(t_min, t_max, con=None, venue=VENUE, pair=PAIR):
    con = con or sqlite3.connect(DB_PATH)
    q = read_sql_blocks("windows.sql")["windows"]
    w = pd.read_sql_query(q, con, params={"venue": venue, "pair": pair, "t_min": t_min, "t_max": t_max})
    w["r"] = np.where(w["is_valid"] == 1, np.log(w["s_end"] / w["s_start"]), np.nan)
    return w


def load_minutes(con=None, venue=VENUE, pair=PAIR):
    con = con or sqlite3.connect(DB_PATH)
    q = read_sql_blocks("windows.sql")["minute_series"]
    return pd.read_sql_query(q, con, params={"venue": venue, "pair": pair})


def minute_grid(minutes):
    """Reindex to a complete 1-minute grid. Missing minutes stay NaN (never filled)."""
    m = minutes.set_index("ts_open").sort_index()
    full = np.arange(m.index.min(), m.index.max() + 60, 60)
    return m.reindex(full)


def trailing_features(minutes, decision_times):
    """Features known at each decision time T, computed only from candles with ts_open <= T-60.

    rv_60  : realized vol over the last 60 one-minute log returns, scaled to a 15-minute horizon
    rv_240 : same over the last 240 minutes
    r_lag1 : previous 15-minute log return  log(close[T-60] / close[T-960])
    r_lag2 : the one before that            log(close[T-960] / close[T-1860])
    A feature is NaN if any minute it needs is missing (rolling windows use min_periods = full length).
    """
    g = minute_grid(minutes)
    lr = np.log(g["close"]).diff()                      # NaN across any gap
    sq = lr ** 2
    rv60 = np.sqrt(sq.rolling(60, min_periods=60).sum() * 15 / 60)       # trailing (right-aligned) windows only
    rv240 = np.sqrt(sq.rolling(240, min_periods=240).sum() * 15 / 240)
    logc = np.log(g["close"])
    last = np.asarray(decision_times) - 60              # last candle completed by T
    out = pd.DataFrame({"t_decision": decision_times})
    out["rv_60"] = rv60.reindex(last).to_numpy()
    out["rv_240"] = rv240.reindex(last).to_numpy()
    out["r_lag1"] = logc.reindex(last).to_numpy() - logc.reindex(last - 900).to_numpy()
    out["r_lag2"] = logc.reindex(last - 900).to_numpy() - logc.reindex(last - 1800).to_numpy()
    out["feature_last_candle_open"] = last
    out["feature_known_at"] = last + 60
    return out


def stack_strikes(df, strikes_bps=STRIKES_BPS):
    """One row per (decision time, strike). y = 1{r > k} with a STRICT inequality."""
    rows = []
    for k in strikes_bps:
        d = df.copy()
        d["k_bps"] = k
        d["K"] = d["s_start"] * np.exp(k / BPS)
        d["y"] = (d["r"] > k / BPS).astype(int)
        rows.append(d)
    return pd.concat(rows, ignore_index=True).sort_values(["t_decision", "k_bps"], ignore_index=True)


def event_threshold(S, K):
    """Required log move for S_end > K."""
    return np.log(K / S)


def chronological_split(df, val_start, test_start, test_end=None, horizon=900):
    """Assign train/val/test by decision time and PURGE rows whose label resolves after the next block starts.

    Returns (df_with_split_column, n_purged). A training row is kept only if t_decision + horizon <= val_start;
    a validation row only if t_decision + horizon <= test_start.
    """
    d = df.copy()
    t = d["t_decision"]
    d["split"] = np.select([t < val_start, t < test_start], ["train", "val"], "test")
    if test_end is not None:
        d.loc[t >= test_end, "split"] = "out"
    label_end = t + horizon
    purge = ((d["split"] == "train") & (label_end > val_start)) | ((d["split"] == "val") & (label_end > test_start))
    n_purged = int(purge.sum())
    d = d.loc[~purge & (d["split"] != "out")].copy()
    return d, n_purged
