#!/usr/bin/env python3
"""
Positive-case detection evaluation (TP/FN only).

Goal:
  Ignore negative cases and measure how often each system identifies
  truly positive deterioration windows.

Definition used:
  - A "positive case" is a patient with at least one row where
    REVIEW_WITHIN_4HOURS == 1.
  - A case is TP for a system if, on any positive row for that patient,
    the system score is >= threshold.
  - Otherwise that positive case is FN.

This keeps temporal scoring history-aware and uses all available rows
for score computation.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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

# Thresholds (same defaults used in prior threshold evaluation)
NEWS_THRESHOLD = 7.0
SNAPSHOT_THRESHOLD = 7.0
TEMPORAL_THRESHOLD = 7.0
NEWS_SWEEP = np.arange(0.0, 20.01, 1.0)
FUZZY_SWEEP = np.arange(0.0, 18.01, 0.5)

# Temporal parameters from best grid-search result
ALPHA = 0.70
BETA = 0.5
GAMMA = 0.80
WINDOW_HOURS = 24.0


def precompute_slopes(df, pv_scores):
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

    additive = sum(adjusted[v] for v in VITALS)
    if GAMMA == 1.0:
        return additive
    stacked = np.column_stack([adjusted[v] for v in VITALS])
    max_vital = stacked.max(axis=1)
    max_based = (18.0 / 3.0) * max_vital
    return (1.0 - GAMMA) * max_based + GAMMA * additive


def patient_tp_fn(patient_ids, positive_mask, score, threshold):
    """
    Return TP/FN over positive patients only.
    TP if patient has any positive-row score >= threshold.
    """
    pos_patients = np.unique(patient_ids[positive_mask])
    if len(pos_patients) == 0:
        return 0, 0, 0.0

    eval_df = pd.DataFrame({
        "pid": patient_ids[positive_mask],
        "score": score[positive_mask],
    })
    patient_max_positive_window = eval_df.groupby("pid", sort=False)["score"].max()
    detected = (patient_max_positive_window.values >= threshold)
    tp = int(detected.sum())
    fn = int(len(detected) - tp)
    recall = tp / len(detected)
    return tp, fn, recall


def sweep_thresholds(patient_ids, positive_mask, score, thresholds):
    """Return TP/FN/recall across candidate thresholds."""
    rows = []
    for th in thresholds:
        tp, fn, recall = patient_tp_fn(patient_ids, positive_mask, score, float(th))
        rows.append({
            "threshold": float(th),
            "tp": tp,
            "fn": fn,
            "tp_over_tp_fn": recall,
        })
    return pd.DataFrame(rows)


def main():
    t_total = time.time()
    print("=" * 74)
    print("  Positive-Case Detection (TP/FN only)")
    print("  Positive case = patient with REVIEW_WITHIN_4HOURS == 1")
    print("  Detection = any score >= threshold on a positive row")
    print("=" * 74)

    df = load_data()

    print("\nUsing all datapoints (no patient filtering).")
    patient_ids = df["ANON_ADMISSION_ID"].values
    label = df["REVIEW_WITHIN_4HOURS"].values.astype(np.int8)
    positive_mask = label == 1
    print(f"  Rows used:     {len(df):,}")
    print(f"  Positive rows: {positive_mask.sum():,}")

    print("\nComputing snapshot/system-1 scores …")
    pv_scores = compute_fuzzy_scores(df)
    snapshot_total = sum(pv_scores[v] for v in VITALS).astype(np.float32)
    news2 = df["NEWS-2"].values.astype(np.float32)

    print("\nComputing temporal scores (history-aware) …")
    all_slopes, gs, ge = precompute_slopes(df, pv_scores)
    temporal_total = compute_temporal_total(pv_scores, all_slopes, gs, ge)

    results = []
    systems = [
        ("NEWS-2", news2, NEWS_THRESHOLD),
        ("Snapshot Fuzzy EWS", snapshot_total, SNAPSHOT_THRESHOLD),
        ("Temporal Builder", temporal_total, TEMPORAL_THRESHOLD),
    ]
    for name, score, threshold in systems:
        tp, fn, recall = patient_tp_fn(patient_ids, positive_mask, score, threshold)
        results.append((name, threshold, tp, fn, recall))

    print("\n── TP/FN across positive cases only ─────────────────────────")
    print(f"{'System':28s} {'Thresh':>7s} {'TP':>8s} {'FN':>8s} {'TP/(TP+FN)':>12s}")
    for name, threshold, tp, fn, recall in results:
        print(f"{name:28s} {threshold:7.2f} {tp:8d} {fn:8d} {recall:12.4f}")

    print("\nSweeping thresholds for positive-case detection …")
    sweep_news = sweep_thresholds(patient_ids, positive_mask, news2, NEWS_SWEEP)
    sweep_snapshot = sweep_thresholds(patient_ids, positive_mask, snapshot_total, FUZZY_SWEEP)
    sweep_temporal = sweep_thresholds(patient_ids, positive_mask, temporal_total, FUZZY_SWEEP)

    for system_name, sweep_df in [
        ("NEWS-2", sweep_news),
        ("Snapshot Fuzzy EWS", sweep_snapshot),
        ("Temporal Builder", sweep_temporal),
    ]:
        best_idx = sweep_df["tp_over_tp_fn"].idxmax()
        best = sweep_df.loc[best_idx]
        print(
            f"  Best {system_name:20s} threshold={best['threshold']:.2f}  "
            f"TP={int(best['tp'])}  FN={int(best['fn'])}  TP/(TP+FN)={best['tp_over_tp_fn']:.4f}"
        )

    out_df = pd.concat([
        sweep_news.assign(system="NEWS-2"),
        sweep_snapshot.assign(system="Snapshot Fuzzy EWS"),
        sweep_temporal.assign(system="Temporal Builder"),
    ], ignore_index=True)[["system", "threshold", "tp", "fn", "tp_over_tp_fn"]]
    out_dir = Path("threshold_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "positive_case_threshold_sweep.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Saved sweep results to: {out_path}")

    print(f"\nTotal wall time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
