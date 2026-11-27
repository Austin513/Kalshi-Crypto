"""Checks on the volatility models: the fast filters match the textbook recursions, the digital price matches
Black-Scholes, and the GARCH fit recovers known parameters from simulated data."""
import numpy as np
from scipy import stats

from btc_tails.vol import VolModel, bs_digital, digital_prob, ewma_var, garch_var


def test_filters_match_loops():
    r = np.random.default_rng(0).normal(size=200)
    g, e = [2.0], [2.0]
    for t in range(1, len(r)):
        g.append(0.1 + 0.1 * r[t - 1] ** 2 + 0.8 * g[-1])
        e.append(0.9 * e[-1] + 0.1 * r[t - 1] ** 2)
    assert np.allclose(garch_var(r, 0.1, 0.1, 0.8, 2.0), g)
    assert np.allclose(ewma_var(r, 0.9, 2.0), e)


def test_forecast_uses_only_the_past():
    r = np.random.default_rng(1).normal(size=50)
    r2 = r.copy()
    r2[30] = 100.0                      # a huge shock at t = 30 can't change the forecast for t <= 30
    a, b = garch_var(r, 0.1, 0.1, 0.8, 1.0), garch_var(r2, 0.1, 0.1, 0.8, 1.0)
    assert np.allclose(a[:31], b[:31]) and b[31] > a[31]


def test_digital_matches_black_scholes():
    S, K, sigma = 100.0, 100.3, 0.002
    k_bps = np.log(K / S) * 1e4
    mu_bps = -0.5 * sigma ** 2 * 1e4    # the BS risk-neutral drift of the log price
    assert np.isclose(digital_prob(k_bps, mu_bps, sigma * 1e4), bs_digital(S, K, sigma))


def test_t_digital_has_fatter_tails():
    # same variance, but a t shock puts more probability on a 3-sigma move
    assert digital_prob(3.0, 0.0, 1.0, nu=4.5) > stats.norm.sf(3.0)
    assert np.isclose(digital_prob(0.0, 0.0, 1.0, nu=4.5), 0.5)


def test_garch_recovers_simulated_parameters():
    rng = np.random.default_rng(2)
    omega, alpha, beta, n = 2.0, 0.08, 0.90, 20_000
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    m = VolModel("garch", "normal").fit(r, np.arange(n) * 900)
    assert abs(m.params["alpha"] - alpha) < 0.02
    assert abs(m.params["beta"] - beta) < 0.03
