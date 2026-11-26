"""The four forecasters. Each returns P(r > threshold) for every (decision time, strike) row.

A  NormalModel       unconditional N(mu, sigma) fitted on training returns
B  EmpiricalModel    fraction of training returns strictly above the threshold
C  logistic_pipeline scikit-learn logistic regression on threshold + trailing features
D  GaussianNet       small PyTorch net -> conditional N(mu_t, sigma_t), trained with Gaussian NLL
"""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import BPS


class NormalModel:
    def fit(self, r):
        r = np.asarray(r)
        self.mu, self.sigma, self.n = r.mean(), r.std(ddof=1), len(r)
        return self

    def prob_above(self, thresh):
        return norm.sf((np.asarray(thresh) - self.mu) / self.sigma)


class EmpiricalModel:
    def fit(self, r):
        self.r = np.sort(np.asarray(r))
        self.n = len(self.r)
        return self

    def count_above(self, thresh):
        return self.n - np.searchsorted(self.r, np.asarray(thresh), side="right")   # strict: r > thresh

    def prob_above(self, thresh):
        return self.count_above(thresh) / self.n


# ---- C: logistic regression -------------------------------------------------------------------------

def logistic_design(df, spec):
    """Feature matrix for a named spec. All columns are known at decision time."""
    k = df["k_bps"].to_numpy(float)
    rv = df["rv_60"].to_numpy(float) * BPS          # bps per 15 minutes
    rv240 = df["rv_240"].to_numpy(float) * BPS
    lag1 = df["r_lag1"].to_numpy(float) * BPS
    lag2 = df["r_lag2"].to_numpy(float) * BPS
    cols = {
        "k": k, "rv_60": rv, "rv_240": rv240, "r_lag1": lag1, "r_lag2": lag2,
        "k_x_rv60": k * rv, "k_over_rv240": k / rv240,
    }
    names = {
        "base":        ["k", "rv_60", "rv_240", "r_lag1", "r_lag2"],
        "interaction": ["k", "rv_60", "rv_240", "r_lag1", "r_lag2", "k_x_rv60"],
        "vol_scaled":  ["k", "k_over_rv240", "r_lag1", "r_lag2"],
    }[spec]
    return np.column_stack([cols[c] for c in names]), names


def logistic_pipeline(C=1.0):
    # The scaler sits inside the pipeline, so it is fitted on training rows only.
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=C, max_iter=2000))])


# ---- D: PyTorch conditional Gaussian ----------------------------------------------------------------

NET_FEATURES = ["log_rv_60", "log_rv_240", "r_lag1", "r_lag2"]


def net_inputs(df):
    return np.column_stack([
        np.log(df["rv_60"].to_numpy(float)), np.log(df["rv_240"].to_numpy(float)),
        df["r_lag1"].to_numpy(float) * BPS, df["r_lag2"].to_numpy(float) * BPS,
    ])


def fit_gaussian_net(X_tr, r_tr, X_va, r_va, hidden=16, lr=3e-3, max_epochs=1000, patience=30, seed=0):
    """Fit mean and scale of the next 15-minute return (in bps). Early stopping on validation NLL.

    Returns (predict_fn, history). predict_fn(X) -> (mu_bps, sigma_bps).
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, std = X_tr.mean(0), X_tr.std(0)               # training-only standardization
    tx = lambda X: torch.tensor((X - mean) / std, dtype=torch.float32)
    ty = lambda r: torch.tensor(np.asarray(r) * BPS, dtype=torch.float32)
    Xt, yt, Xv, yv = tx(X_tr), ty(r_tr), tx(X_va), ty(r_va)

    net = nn.Sequential(nn.Linear(X_tr.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, 2))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.GaussianNLLLoss(full=True)
    softplus = nn.Softplus()

    def forward(X):
        out = net(X)
        sigma = softplus(out[:, 1]) + 1e-3
        return out[:, 0], sigma

    def nll(X, y):
        mu, sigma = forward(X)
        return loss_fn(mu, y, sigma ** 2)

    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, hist = np.inf, None, 0, []
    for epoch in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(Xt), generator=gen).split(256):
            opt.zero_grad()
            loss = nll(Xt[idx], yt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tr, va = nll(Xt, yt).item(), nll(Xv, yv).item()
        hist.append((epoch, tr, va))
        if va < best - 1e-5:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            mu, sigma = forward(tx(X))
        return mu.numpy().astype(float), sigma.numpy().astype(float)

    return predict, hist
