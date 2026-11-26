"""Checks for the silent errors that would quietly corrupt the results.

Uses a tiny hand-built set of candles in a temporary SQLite file; it never touches data/.
Fixture layout (epoch seconds, 1-minute candles):
  window A: T = 900   needs candles 840 .. 1740  -> all present, start close 100.0, end close 110.0
  window B: T = 1800  needs candles 1740 .. 2640 -> candle 2100 deliberately missing
The conflicting-duplicate test adds a second, different row for minute 1200 on top of this fixture.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from btc_tails import SQL_DIR
from btc_tails.build_db import read_sql_blocks
from btc_tails.features import chronological_split, event_threshold, stack_strikes, trailing_features


def make_db(tmp_path, rows):
    con = sqlite3.connect(tmp_path / "fixture.sqlite")
    con.executescript((SQL_DIR / "schema.sql").read_text())
    con.executemany(
        "INSERT INTO raw_candles VALUES ('fx','TEST',?,?,?,?,?,1.0,1,'fixture')",
        [(t, c, c, c, c) for t, c in rows])
    con.executescript((SQL_DIR / "normalize.sql").read_text())
    return con


def fixture_rows():
    rows = [(t, 105.0) for t in range(840, 2700, 60)]   # flat 105 everywhere ...
    d = dict(rows)
    d[840] = 100.0     # ... except the start of window A
    d[1740] = 110.0    # ... and the end of window A (also the start of window B)
    del d[2100]        # one missing minute inside window B
    return sorted(d.items())


def windows(con, t_min=900, t_max=2700):
    w = pd.read_sql_query(read_sql_blocks("windows.sql")["windows"], con,
                          params={"venue": "fx", "pair": "TEST", "t_min": t_min, "t_max": t_max})
    w["r"] = np.where(w.is_valid == 1, np.log(w.s_end / w.s_start), np.nan)
    return w.set_index("t_decision")


def test_known_return(tmp_path):
    w = windows(make_db(tmp_path, fixture_rows()))
    assert w.loc[900, "is_valid"] == 1
    assert w.loc[900, "s_start"] == 100.0 and w.loc[900, "s_end"] == 110.0
    assert w.loc[900, "r"] == pytest.approx(np.log(1.10))


def test_missing_minute_window_is_dropped(tmp_path):
    w = windows(make_db(tmp_path, fixture_rows()))
    assert w.loc[1800, "n_candles"] == 15          # 16 needed, one missing
    assert w.loc[1800, "is_valid"] == 0
    assert np.isnan(w.loc[1800, "r"])


def test_conflicting_duplicate_is_not_averaged(tmp_path):
    rows = fixture_rows() + [(1200, 999.0)]         # second, different row for minute 1200
    con = make_db(tmp_path, rows)
    assert con.execute("SELECT COUNT(*) FROM candles WHERE ts_open = 1200").fetchone()[0] == 0
    assert windows(con).loc[900, "is_valid"] == 0  # window A now has a hole -> excluded


def test_event_threshold_and_strict_inequality():
    S = 100.0
    K = S * np.exp(25 / 1e4)                         # +25 log-bps strike
    assert event_threshold(S, K) == pytest.approx(0.0025)
    df = pd.DataFrame({"t_decision": [0, 900], "s_start": [S, S], "r": [0.0025, 0.0025 + 1e-9]})
    y = stack_strikes(df, [25]).set_index("t_decision")["y"]
    assert y.loc[0] == 0                              # exactly at the strike -> NOT above
    assert y.loc[900] == 1


def test_split_purges_label_that_overlaps_next_block():
    df = pd.DataFrame({"t_decision": [0, 900, 1800, 2700]})
    # validation starts at 1500: the training row at 900 resolves at 1800 > 1500 and must be purged
    out, n_purged = chronological_split(df, val_start=1500, test_start=2700)
    assert n_purged == 1
    assert 900 not in out["t_decision"].values
    assert (out.loc[out.split == "train", "t_decision"] + 900 <= 1500).all()
    assert (out.loc[out.split == "val", "t_decision"] + 900 <= 2700).all()


def test_features_use_only_completed_candles():
    # A price spike at the decision minute itself must not leak into features at that decision time.
    t = np.arange(0, 600 * 60, 60)
    close = np.full(len(t), 100.0)
    T = 300 * 60
    close[t >= T] = 200.0                             # candle T (still open at T) and later jump
    f = trailing_features(pd.DataFrame({"ts_open": t, "open": close, "high": close, "low": close, "close": close}), [T])
    assert f.loc[0, "rv_60"] == 0.0 and f.loc[0, "r_lag1"] == 0.0
    assert f.loc[0, "feature_known_at"] == T
