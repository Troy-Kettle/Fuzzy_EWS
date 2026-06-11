#!/usr/bin/env python3
"""
Evaluate AUROC using one row per patient (the last observation).

Rules requested:
  1) NEWS-2 uses the patient's last observation score.
  2) Snapshot Fuzzy EWS (System 1) uses the patient's last observation score.
  3) Temporal system score is computed normally over each patient's full history,
     then evaluated at that same last observation.
"""

import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from grid_search_auroc import (
    VITALS,
    _ewma_all,
    _ols_slopes_for_group,
    load_data,
    compute_fuzzy_scores,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(over="ignore", invalid="ignore")

# Optimal temporal parameters from your prior search.
ALPHA = 0.70
BETA = 0.5
GAMMA = 0.80
WINDOW_HOURS = 24.0

SCRIPT_DIR = Path(__file__).parent


def precompute_slopes(df, pv_scores):
    """Compute OLS trend slope per row for each vital."""
    window_min = WINDOW_HOURS * 60.0
    patient_ids = df["ANON_ADMISSION_ID"].values
    t_minutes = df["t_minutes"].values.astype(np.float64)

    change = np.empty(len(patient_ids), dtype=bool)
    change[0] = True
    change[1:] = patient_ids[1:] != patient_ids[:-1]
    group_starts = np.where(change)[0]
    group_ends = np.append(group_starts[1:], len(patient_ids))

    all_slopes = {}
    for vital in VITALS:
        print(f"  slopes: {vital:25s}", end="", flush=True)
        t0 = time.time()
        raw = pv_scores[vital].astype(np.float64)
        slopes = np.zeros(len(df), dtype=np.float32)
        for g in range(len(group_starts)):
            s, e = group_starts[g], group_ends[g]
            slopes[s:e] = _ols_slopes_for_group(t_minutes[s:e], raw[s:e], window_min)
        all_slopes[vital] = slopes
        print(f"  {time.time() - t0:.0f}s", flush=True)
    return all_slopes, group_starts, group_ends


def compute_temporal_total(pv_scores, all_slopes, group_starts, group_ends):
    """Compute temporal-adjusted total score at every row."""
    adjusted = {}
    for vital in VITALS:
        raw = pv_scores[vital]
        ew = _ewma_all(group_starts, group_ends, raw, ALPHA)
        clamped = np.maximum(ew, raw)

        slope = all_slopes[vital]
        pos = slope > 0
        tf = np.zeros_like(slope)
        if pos.any():
            ex = np.exp(np.clip(-BETA * slope[pos], -700, 700))
            tf[pos] = 2.0 / (1.0 + ex) - 1.0

        adj = clamped + tf * (3.0 - clamped)
        adjusted[vital] = np.clip(adj, 0.0, 3.0).astype(np.float32)

    snapshot = sum(pv_scores[v] for v in VITALS)
    additive = sum(adjusted[v] for v in VITALS)
    if GAMMA == 1.0:
        return np.maximum(additive, snapshot).astype(np.float32)

    stacked = np.column_stack([adjusted[v] for v in VITALS])
    max_vital = stacked.max(axis=1)
    max_based = (18.0 / 3.0) * max_vital
    total = (1.0 - GAMMA) * max_based + GAMMA * additive
    return np.maximum(total, snapshot).astype(np.float32)


def last_index_per_patient(patient_ids):
    """Return the index of the final row for each patient."""
    change_next = np.empty(len(patient_ids), dtype=bool)
    change_next[:-1] = patient_ids[:-1] != patient_ids[1:]
    change_next[-1] = True
    return np.where(change_next)[0]


def main():
    t_total = time.time()
    print("=" * 72)
    print("  Last-Observation AUROC Evaluation (one row per patient)")
    print(f"  Temporal params: α={ALPHA}  β={BETA}  γ={GAMMA}  window={WINDOW_HOURS}h")
    print("=" * 72)

    # 1) Load and sort data (handled by load_data()).
    df = load_data()
    patient_ids = df["ANON_ADMISSION_ID"].values
    labels = df["REVIEW_WITHIN_4HOURS"].values.astype(np.int8)

    # 2) Build snapshot/system-1 per-vital scores and snapshot total.
    pv_scores = compute_fuzzy_scores(df)
    snapshot_total = sum(pv_scores[v] for v in VITALS).astype(np.float32)

    # 3) Build temporal score at each row using full patient history.
    print("\nPrecomputing slopes for temporal system …")
    all_slopes, group_starts, group_ends = precompute_slopes(df, pv_scores)
    print("\nComputing temporal-adjusted totals …")
    temporal_total = compute_temporal_total(pv_scores, all_slopes, group_starts, group_ends)

    # 4) Keep only each patient's final observation.
    idx_last = last_index_per_patient(patient_ids)
    y_last = labels[idx_last]
    news2_last = df["NEWS-2"].values.astype(np.float32)[idx_last]
    snapshot_last = snapshot_total[idx_last]
    temporal_last = temporal_total[idx_last]

    print("\n── Last-observation cohort ─────────────────────────────────")
    print(f"  Patients (rows kept): {len(idx_last):,}")
    print(f"  Positive labels:      {y_last.sum():,} / {len(y_last):,} ({100 * y_last.mean():.2f}%)")

    # 5) AUROC on one row per patient.
    models = {
        "NEWS-2 (last obs)": news2_last,
        "Snapshot Fuzzy EWS (last obs)": snapshot_last,
        "Temporal Builder (history-aware at last obs)": temporal_last,
    }

    print("\n── AUROC (one row per patient) ─────────────────────────────")
    for name, score in models.items():
        valid = np.isfinite(score)
        auc = roc_auc_score(y_last[valid], score[valid])
        print(f"  {name:45s}  AUROC = {auc:.6f}")

    print(f"\nTotal wall time: {time.time() - t_total:.0f}s")
    print(f"Script path: {SCRIPT_DIR / 'last_observation_auroc.py'}")


if __name__ == "__main__":
    main()
