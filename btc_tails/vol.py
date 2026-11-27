"""Volatility forecasts for the next 15-minute return, and digital-option pricing from them.

Models (all fit by maximum likelihood on the training months only):
  EWMA        sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r[t-1]^2                 (RiskMetrics)
  GARCH(1,1)  sigma2[t] = omega + alpha * r[t-1]^2 + beta * sigma2[t-1]
              with either normal or Student-t shocks.

Bitcoin is much more volatile at some hours than others (US equity open, funding times), so there is an option to
divide each return by a time-of-day scale first and run the model on what's left. The scale for each of the 96
quarter-hours is the RMS return in that slot over the training months.

Returns are in basis points throughout so the parameters are readable.
"""
import numpy as np
from scipy import optimize, stats
from scipy.signal import lfilter

SLOTS = 96   # quarter-hours per day


# ---- intraday seasonality ---------------------------------------------------------------------------

def slot_of(t_decision):
    return (np.asarray(t_decision) % 86400) // 900


def seasonal_scale(r, t_decision):
    """RMS return for each quarter-hour of the day, normalised so the average squared scale is 1."""
    r, s = np.asarray(r), slot_of(t_decision)
    ms = np.array([np.mean(r[s == k] ** 2) for k in range(SLOTS)])
    return np.sqrt(ms / ms.mean())


# ---- variance recursions ----------------------------------------------------------------------------
# Both recursions are linear filters in r^2, so lfilter runs them in C instead of a Python loop.
# sigma2[t] only uses returns up to t-1, i.e. it is a genuine one-step-ahead forecast.

def ewma_var(r, lam, sigma2_0):
    x = (1 - lam) * np.concatenate([[0.0], np.asarray(r)[:-1] ** 2])
    x[0] = sigma2_0                       # out[0] = sigma2_0, then out[t] = x[t] + lam * out[t-1]
    return lfilter([1.0], [1.0, -lam], x)


def garch_var(r, omega, alpha, beta, sigma2_0):
    x = omega + alpha * np.concatenate([[0.0], np.asarray(r)[:-1] ** 2])
    x[0] = sigma2_0
    return lfilter([1.0], [1.0, -beta], x)


# ---- log likelihoods --------------------------------------------------------------------------------

def _nll(r, sigma2, nu=None):
    """Negative mean log likelihood of r given its forecast variance. nu=None means normal shocks.

    For Student-t the shock is scaled to have variance 1, so sigma2 is still the variance of r.
    """
    if nu is None:
        return 0.5 * np.mean(np.log(2 * np.pi * sigma2) + r ** 2 / sigma2)
    scale = np.sqrt(sigma2 * (nu - 2) / nu)
    return -np.mean(stats.t.logpdf(r / scale, nu) - np.log(scale))


class VolModel:
    """kind in {"ewma", "garch"}, dist in {"normal", "t"}, seasonal: divide out time-of-day scale first."""

    def __init__(self, kind="garch", dist="normal", seasonal=False):
        self.kind, self.dist, self.seasonal = kind, dist, seasonal

    @property
    def name(self):
        return f"{self.kind}-{self.dist}" + ("-seasonal" if self.seasonal else "")

    def _deseason(self, r, t):
        return r / self.scale[slot_of(t)] if self.seasonal else r

    def _var(self, z, p):
        if self.kind == "ewma":
            return ewma_var(z, p["lam"], self.var0)
        return garch_var(z, p["omega"], p["alpha"], p["beta"], self.var0)

    def fit(self, r_bps, t_decision):
        r_bps = np.asarray(r_bps, float)
        self.mu = r_bps.mean()                       # tiny; kept so the forecast is centred like model A
        self.scale = seasonal_scale(r_bps - self.mu, t_decision) if self.seasonal else None
        z = self._deseason(r_bps - self.mu, t_decision)
        self.var0 = z.var()
        t_dist = self.dist == "t"

        if self.kind == "ewma":
            names, x0, bounds = ["lam"], [0.9], [(0.5, 0.9999)]
        else:
            # start from a typical intraday GARCH: persistence ~0.97
            names, x0 = ["omega", "alpha", "beta"], [0.03 * self.var0, 0.08, 0.89]
            bounds = [(1e-8, None), (1e-6, 0.5), (0.0, 0.9999)]
        if t_dist:
            names, x0, bounds = names + ["nu"], x0 + [5.0], bounds + [(2.05, 100.0)]

        def obj(x):
            p = dict(zip(names, x))
            if self.kind == "garch" and p["alpha"] + p["beta"] >= 0.9999:
                return 1e6                           # keep the process stationary
            return _nll(z, self._var(z, p), p.get("nu"))

        res = optimize.minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        if self.kind == "garch":                     # L-BFGS-B ignores the penalty's kink; polish with Nelder-Mead
            res = optimize.minimize(obj, res.x, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
        self.params = dict(zip(names, res.x))
        self.train_nll = res.fun
        return self

    def sigma(self, r_bps, t_decision):
        """One-step-ahead sigma (bps) for every window, filtered through the whole series with frozen params."""
        z = self._deseason(np.asarray(r_bps, float) - self.mu, t_decision)
        s = np.sqrt(self._var(z, self.params))
        return s * self.scale[slot_of(t_decision)] if self.seasonal else s

    def nll(self, r_bps, sigma):
        return _nll(np.asarray(r_bps) - self.mu, np.asarray(sigma) ** 2, self.params.get("nu"))

    def prob_above(self, k_bps, sigma):
        """P(r > k) for a 15-minute return with forecast sigma, i.e. the price of a cash-or-nothing digital."""
        return digital_prob(k_bps, self.mu, sigma, self.params.get("nu"))

    def half_life(self):
        """Windows until a volatility shock has decayed by half."""
        rho = self.params["lam"] if self.kind == "ewma" else self.params["alpha"] + self.params["beta"]
        return np.log(0.5) / np.log(rho)


# ---- digital pricing --------------------------------------------------------------------------------

def bs_digital(S, K, sigma, tau=1.0):
    """Black-Scholes cash-or-nothing call with zero rates: N(d2), d2 = (ln(S/K) - sigma^2 tau / 2) / (sigma sqrt(tau)).

    sigma is per unit of tau in log-return terms (here: per 15-minute window, as a fraction, not bps).
    """
    d2 = (np.log(S / K) - 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    return stats.norm.cdf(d2)


def digital_prob(k_bps, mu_bps, sigma_bps, nu=None):
    """P(log return > k) when the return is mu + sigma * shock.

    With normal shocks and mu = -sigma^2/2 this is exactly bs_digital; over 15 minutes sigma^2/2 is about
    0.02 bps, so the drift term doesn't matter in practice. With Student-t shocks the standard normal CDF is
    replaced by a unit-variance t CDF, which puts more weight on large moves.
    """
    z = (np.asarray(k_bps) - mu_bps) / np.asarray(sigma_bps)
    if nu is None:
        return stats.norm.sf(z)
    return stats.t.sf(z * np.sqrt(nu / (nu - 2)), nu)
