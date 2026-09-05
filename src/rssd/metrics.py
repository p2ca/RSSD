"""Forecast-skill metrics: correlation helpers and per-reservoir NSE/R2."""

from __future__ import annotations

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import numpy as np

__all__ = ["corr", "_pearsonr_np", "_spearmanr_simple", "_rankdata_average_ties", "_per_reservoir_r2", "_per_reservoir_r2_daily", "_per_reservoir_r2_log1p"]

def corr(x, y):
    """Return (pearson, spearman) with NaN-safe masking."""
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return (np.nan, np.nan)
    return (pearsonr(x[m], y[m])[0], spearmanr(x[m], y[m])[0])


def _pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum())
    if denom == 0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _spearmanr_simple(x: np.ndarray, y: np.ndarray) -> float:
    """No-tie robust enough for our use; if ties are heavy, switch to scipy."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_rank = x.argsort().argsort().astype(np.float64)
    y_rank = y.argsort().argsort().astype(np.float64)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denom = (np.sqrt((x_rank ** 2).mean()) * np.sqrt((y_rank ** 2).mean()) + 1e-12)
    return float((x_rank * y_rank).mean() / denom)


def _rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    """Numpy-only rankdata with average ranks for ties, 1-based ranks."""
    a = np.asarray(a)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    v = a[order]
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while (j + 1) < n and v[j + 1] == v[i]:
            j += 1
        # average rank for tie group [i, j], 1-based
        r = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = r
        i = j + 1
    return ranks


def _per_reservoir_r2(preds, targets, idx_to_reservoir):
    """
    preds/targets: (n_samples, n_nodes, n_days) in the SAME space (scaled or orig)
    idx_to_reservoir: {node_idx -> reservoir_name}

    Returns:
      r2_dict: {reservoir_name: r2}
      worst: list of (reservoir_name, r2) sorted ascending by r2
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    assert preds.shape == targets.shape, f"preds {preds.shape} != targets {targets.shape}"
    assert preds.ndim == 3, f"Expect (N, nodes, days), got {preds.shape}"

    n_samples, n_nodes, n_days = preds.shape

    r2_dict = {}
    for j in range(n_nodes):
        name = idx_to_reservoir.get(j, f"Node_{j}")
        y_true = targets[:, j, :].reshape(-1)
        y_pred = preds[:, j, :].reshape(-1)

        # guard: if y_true constant -> r2 undefined; set nan
        if np.allclose(np.nanstd(y_true), 0.0):
            r2 = np.nan
        else:
            r2 = r2_score(y_true, y_pred)

        r2_dict[name] = float(r2) if np.isfinite(r2) else np.nan

    worst = sorted(r2_dict.items(), key=lambda kv: (np.inf if np.isnan(kv[1]) else kv[1]))
    return r2_dict, worst


def _per_reservoir_r2_daily(preds, targets, idx_to_reservoir):
    """
    preds/targets: (n_samples, n_nodes, n_days) in the SAME space (scaled or orig)
    Return:
      r2_daily_dict: {reservoir_name: [r2_d1, ..., r2_dN]}
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    assert preds.shape == targets.shape, f"preds {preds.shape} != targets {targets.shape}"
    assert preds.ndim == 3, f"Expect (N, nodes, days), got {preds.shape}"

    n_samples, n_nodes, n_days = preds.shape

    r2_daily_dict = {}
    for j in range(n_nodes):
        name = idx_to_reservoir.get(j, f"Node_{j}")
        r2s = []
        for d in range(n_days):
            y_true = targets[:, j, d].reshape(-1)
            y_pred = preds[:, j, d].reshape(-1)

            # guard: if y_true constant -> r2 undefined; set nan
            if np.allclose(np.nanstd(y_true), 0.0):
                r2 = np.nan
            else:
                r2 = r2_score(y_true, y_pred)

            r2s.append(float(r2) if np.isfinite(r2) else np.nan)
        r2_daily_dict[name] = r2s

    return r2_daily_dict


def _per_reservoir_r2_log1p(preds, targets, idx_to_reservoir):
    """Per-reservoir R² in log(y+1) space. Clips negatives to 0 before transform."""
    log_preds   = np.log1p(np.maximum(np.asarray(preds),   0.0))
    log_targets = np.log1p(np.maximum(np.asarray(targets), 0.0))
    return _per_reservoir_r2(log_preds, log_targets, idx_to_reservoir)
