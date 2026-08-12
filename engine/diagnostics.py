"""Diagnostics that make the snapshot↔temporal coupling explicit (issues #1, #7).

AUROC depends only on the *rank* of patients. If snapshot-peak and temporal-peak rank
patients near-identically, their AUROCs must match — regardless of how different the raw
scores look. These helpers quantify that coupling so the near-null result is reported as
a measured fact, not buried under an 8-leaf experiment matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def coupling_stats(snap_pat: np.ndarray, temp_pat: np.ndarray) -> dict:
    """Rank/level coupling between two patient-level score vectors."""
    snap = np.asarray(snap_pat, np.float64)
    temp = np.asarray(temp_pat, np.float64)
    boost = temp - snap
    return {
        "n_patients": int(len(snap)),
        "spearman": float(spearmanr(snap, temp).correlation),
        "pearson": float(np.corrcoef(snap, temp)[0, 1]),
        "pct_identical": float(100.0 * np.mean(np.abs(boost) < 1e-6)),
        "mean_boost": float(boost.mean()),
        "median_boost": float(np.median(boost)),
        "p90_boost": float(np.quantile(boost, 0.90)),
    }


def same_peak_row_fraction(snap_row: np.ndarray, temp_row: np.ndarray,
                           gs: np.ndarray, ge: np.ndarray) -> float:
    """% of patients whose snapshot peak and temporal peak fall on the SAME observation.
    A high value means temporal is just scoring the snapshot's worst moment + a bump."""
    same = 0
    for g in range(len(gs)):
        s, e = gs[g], ge[g]
        if np.argmax(snap_row[s:e]) == np.argmax(temp_row[s:e]):
            same += 1
    return float(100.0 * same / len(gs))


def coupling_table(pairs: dict, snap_row=None, temp_rows=None,
                   gs=None, ge=None) -> pd.DataFrame:
    """Build a coupling table over named (snap_pat, temp_pat) pairs.

    pairs: {label: (snap_pat, temp_pat)}.
    Optional snap_row + temp_rows[label] (+ gs, ge) add a same_peak_row_pct column.
    """
    rows = []
    for label, (snap_pat, temp_pat) in pairs.items():
        rec = {"comparison": label, **coupling_stats(snap_pat, temp_pat)}
        if snap_row is not None and temp_rows is not None and label in temp_rows:
            rec["same_peak_row_pct"] = same_peak_row_fraction(
                snap_row, temp_rows[label], gs, ge)
        rows.append(rec)
    return pd.DataFrame(rows)


def stratified_auroc(y_pat: np.ndarray, scores: dict, stay_len: np.ndarray,
                     edges=(1, 2, 3, 5, 10, 20, np.inf)) -> pd.DataFrame:
    """AUROC per system within stay-length bins (issue #6 visibility).

    y_pat: binary patient labels. scores: {system: patient_score}. stay_len: obs/patient.
    Returns long-form DataFrame (bin, n, n_pos, system, auroc).
    """
    edges = list(edges)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (stay_len >= lo) & (stay_len < hi)
        n = int(mask.sum()); npos = int(y_pat[mask].sum())
        label = f"[{lo},{hi})" if np.isfinite(hi) else f">={lo}"
        for sysname, sc in scores.items():
            if npos == 0 or npos == n:
                au = float("nan")
            else:
                s = np.asarray(sc)[mask]; ok = np.isfinite(s)
                au = (float(roc_auc_score(y_pat[mask][ok], s[ok]))
                      if 0 < y_pat[mask][ok].sum() < ok.sum() else float("nan"))
            rows.append({"stay_bin": label, "n": n, "n_pos": npos,
                         "system": sysname, "auroc": round(au, 5) if au == au else au})
    return pd.DataFrame(rows)
