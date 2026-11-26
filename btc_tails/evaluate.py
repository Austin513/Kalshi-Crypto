"""Scoring rules, reliability bins, monotonicity checks, and a day-block bootstrap."""
import numpy as np
import pandas as pd

# Scoring convention (applied to EVERY model): probabilities are clipped to [EPS, 1-EPS] before log loss.
# Brier score uses the unclipped probability.
EPS = 1e-4


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score_table(df, models, by=None):
    """Brier / log loss / n for each model, overall or grouped by `by`."""
    groups = [("all", df)] if by is None else list(df.groupby(by))
    rows = []
    for g, d in groups:
        for m in models:
            rows.append({"group": g, "model": m, "n": len(d),
                         "brier": brier(d[m], d["y"]), "log_loss": log_loss(d[m], d["y"])})
    return pd.DataFrame(rows)


def reliability(p, y, bins=np.linspace(0, 1, 11)):
    d = pd.DataFrame({"p": p, "y": y})
    d["bin"] = pd.cut(d["p"], bins, include_lowest=True)
    return d.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), obs_freq=("y", "mean"), n=("y", "size")).reset_index()


def monotonicity_violations(df, model, tol=1e-12):
    """Count decision times where P(clear strike) INCREASES as the strike rises."""
    d = df.sort_values(["t_decision", "k_bps"])
    diffs = d.groupby("t_decision")[model].diff()
    bad = d.loc[diffs > tol, "t_decision"].unique()
    return len(bad)


def block_bootstrap_diff(df, model, baseline, metric, block_col="day", n_boot=2000, seed=0):
    """Mean metric(model) - metric(baseline) with a percentile 95% CI.

    Resamples whole blocks (calendar days) of decision times, so every strike from a decision time and
    neighbouring windows stay together. Negative = model better (both metrics are losses).
    """
    f = {"brier": lambda p, y: (p - y) ** 2,
         "log_loss": lambda p, y: -(y * np.log(np.clip(p, EPS, 1 - EPS)) + (1 - y) * np.log(1 - np.clip(p, EPS, 1 - EPS)))}[metric]
    d = df.assign(_diff=f(df[model].to_numpy(), df["y"].to_numpy()) - f(df[baseline].to_numpy(), df["y"].to_numpy()))
    per_block = d.groupby(block_col)["_diff"].agg(["sum", "size"])
    s, n = per_block["sum"].to_numpy(), per_block["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return float(d["_diff"].mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
