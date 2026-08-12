"""Statistical comparison of correlated ROC curves — DeLong's test.

Two EWS scores evaluated on the *same* patients give correlated AUCs, so a naive
two-sample test overstates significance. DeLong's method gives the variance of each
AUC and their covariance from structural (placement) components, yielding a z-test
on the AUC difference. This is what makes "Δ AUROC = 0.0015" reportable as
not-significant rather than implied to be a win (issue #7).

Reference: Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm…", IEEE SPL.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of x (ties share the average rank). O(n log n).

    Fully vectorised: the tie blocks are found with a boundary mask rather than a
    Python scan, which matters because the row-level pools here run to millions of
    observations and NEWS-2's integer scores make the tie blocks huge.
    """
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts_block = np.empty(n, dtype=bool)
    starts_block[0] = True
    np.not_equal(xs[1:], xs[:-1], out=starts_block[1:])
    block_start = np.flatnonzero(starts_block)
    block_end = np.r_[block_start[1:], n]                  # exclusive
    # 1-based average rank across each tie block
    avg = 0.5 * (block_start + block_end - 1) + 1
    tr = avg[np.cumsum(starts_block) - 1]
    out = np.empty(n, dtype=np.float64)
    out[order] = tr
    return out


def _fast_delong(predictions_sorted: np.ndarray, m: int):
    """Fast DeLong structural components.

    predictions_sorted: (k_predictors, n) with the m positive samples first.
    Returns (aucs[k], covariance[k,k])."""
    n = predictions_sorted.shape[1]
    n_neg = n - m
    k = predictions_sorted.shape[0]
    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n_neg], dtype=np.float64)
    tz = np.empty([k, n], dtype=np.float64)
    for r in range(k):
        tx[r] = _compute_midrank(predictions_sorted[r, :m])
        ty[r] = _compute_midrank(predictions_sorted[r, m:])
        tz[r] = _compute_midrank(predictions_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n_neg - (m + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :m] - tx[:, :]) / n_neg
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n_neg
    return aucs, np.atleast_2d(cov)


def delong_auc_ci(y_true: np.ndarray, score: np.ndarray, alpha: float = 0.05):
    """Analytic 95% CI for a single AUC from DeLong's variance.

    Same structural components as ``delong_roc_test``, just with one predictor, so the
    point estimate it returns reproduces ``roc_auc_score`` exactly rather than being a
    resampled approximation. Returns (auc, lo, hi), clipped to [0, 1].
    """
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=np.float64)
    keep = np.isfinite(s)
    y, s = y[keep].astype(int), s[keep]
    pos = y == 1
    m = int(pos.sum())
    if m == 0 or m == len(y):
        return float("nan"), float("nan"), float("nan")
    order = np.r_[np.where(pos)[0], np.where(~pos)[0]]
    aucs, cov = _fast_delong(s[order][None, :], m)
    se = float(np.sqrt(max(float(cov[0, 0]), 0.0)))
    half = stats.norm.ppf(1.0 - alpha / 2.0) * se
    auc = float(aucs[0])
    return auc, float(np.clip(auc - half, 0.0, 1.0)), float(np.clip(auc + half, 0.0, 1.0))


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray):
    """Compare two correlated ROC AUCs on the same binary labels.

    Returns dict: auc_a, auc_b, delta (a−b), se (of the difference), z, p (two-sided),
    and ci95 (of the difference). Higher score ⇒ positive class is assumed.
    NaNs in either score (or labels) are dropped pairwise.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(y_true.astype(np.float64))
    y, a, b = y_true[keep].astype(int), a[keep], b[keep]
    pos = y == 1
    m = int(pos.sum())
    if m == 0 or m == len(y):
        raise ValueError("DeLong needs both classes present.")
    order = np.r_[np.where(pos)[0], np.where(~pos)[0]]   # positives first
    preds = np.vstack([a[order], b[order]])
    aucs, cov = _fast_delong(preds, m)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var_diff, 0.0)))
    delta = float(aucs[0] - aucs[1])
    z = delta / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    half = 1.959963984540054 * se
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": delta,
            "se": se, "z": float(z), "p": p,
            "ci95": (delta - half, delta + half)}
