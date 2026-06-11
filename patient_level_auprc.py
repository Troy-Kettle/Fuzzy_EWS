#!/usr/bin/env python3
"""
Patient-level AUPRC evaluation (one patient = one case).

Key requirement:
  Temporal system is computed row-by-row with full history per patient,
  then collapsed to one score per patient for evaluation.

For consistency, NEWS-2 and Snapshot are also collapsed to one score per
patient using the same aggregation rule (max over history).
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

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

# Optimal temporal parameters
ALPHA = 0.70
BETA = 0.5
GAMMA = 0.80
WINDOW_HOURS = 24.0


def precompute_slopes(df, pv_scores):
    """Compute per-row trend slopes for each vital."""
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
    """Temporal-adjusted score per row (history-aware)."""
    adjusted = {}
    for vital in VITALS:
        raw = pv_scores[vital]
        ew = _ewma_all(group_starts, group_ends, raw, ALPHA)
        clamped = np.maximum(ew, raw)
        slope = all_slopes[vital]

        tf = np.zeros_like(slope)
        pos = slope > 0
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


def collapse_patient_level(df, news2, snapshot, temporal):
    """Collapse row-level outputs to one score and one label per patient."""
    temp = pd.DataFrame({
        "pid": df["ANON_ADMISSION_ID"].values,
        "label": df["REVIEW_WITHIN_4HOURS"].values.astype(np.int8),
        "news2": news2,
        "snapshot": snapshot,
        "temporal": temporal,
    })

    # One case per patient:
    # - label: positive if any deterioration window occurs
    # - score: max risk encountered during patient history
    pat = temp.groupby("pid", sort=False).agg({
        "label": "max",
        "news2": "max",
        "snapshot": "max",
        "temporal": "max",
    })
    return pat.reset_index(drop=False)


def pr_at_recall(y_true, y_score, targets=(0.5, 0.7, 0.8, 0.9)):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    rows = []
    for t in targets:
        idx = np.argmin(np.abs(rec - t))
        rows.append((t, float(rec[idx]), float(prec[idx])))
    return rows


def main():
    t_total = time.time()
    print("=" * 74)
    print("  Patient-Level AUPRC (one patient = one case)")
    print(f"  Temporal params: α={ALPHA}  β={BETA}  γ={GAMMA}  window={WINDOW_HOURS}h")
    print("  Aggregation per patient: max score over history")
    print("=" * 74)

    df = load_data()

    print("\nComputing snapshot/system-1 scores …")
    pv_scores = compute_fuzzy_scores(df)
    snapshot_total = sum(pv_scores[v] for v in VITALS).astype(np.float32)
    news2 = df["NEWS-2"].values.astype(np.float32)

    print("\nComputing temporal scores (history-aware) …")
    all_slopes, gs, ge = precompute_slopes(df, pv_scores)
    temporal_total = compute_temporal_total(pv_scores, all_slopes, gs, ge)

    print("\nCollapsing to patient-level cases …")
    pat = collapse_patient_level(df, news2, snapshot_total, temporal_total)
    y = pat["label"].values.astype(np.int8)
    prevalence = y.mean()
    print(f"  Patients: {len(pat):,}")
    print(f"  Positive patients: {y.sum():,} / {len(y):,} ({100*prevalence:.2f}%)")

    models = {
        "NEWS-2": pat["news2"].values.astype(np.float64),
        "Snapshot Fuzzy EWS": pat["snapshot"].values.astype(np.float64),
        "Temporal Builder": pat["temporal"].values.astype(np.float64),
    }

    print("\n── Patient-level AUPRC (Average Precision) ────────────────")
    print(f"  Baseline prevalence (no-skill AP): {prevalence:.6f}")
    for name, score in models.items():
        valid = np.isfinite(score)
        ap = average_precision_score(y[valid], score[valid])
        print(f"  {name:22s} AP = {ap:.6f}")
        for target, actual_rec, prec in pr_at_recall(y[valid], score[valid]):
            print(
                f"    Recall target {target:.0%} -> actual recall {actual_rec:.4f}, precision {prec:.4f}"
            )

    print(f"\nTotal wall time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
