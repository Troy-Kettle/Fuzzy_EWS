#!/usr/bin/env python3
"""
Patient-level AUROC evaluation (one patient = one case).

Temporal score is computed row-by-row with each patient's full history,
then collapsed to one score per patient (max over history).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from grid_search_auroc import VITALS, compute_fuzzy_scores, load_data
from patient_level_auprc import (
    collapse_patient_level,
    compute_temporal_total,
    precompute_slopes,
)

OUT_DIR = Path("auroc_results")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("  Patient-Level AUROC (one patient = one case)")
    print("  Temporal uses full patient history before collapse")
    print("=" * 74)

    df = load_data()

    print("\nComputing NEWS-2, Snapshot, and Temporal row-level scores ...")
    pv_scores = compute_fuzzy_scores(df)
    snapshot_total = sum(pv_scores[v] for v in VITALS).astype(np.float32)
    news2 = df["NEWS-2"].values.astype(np.float32)
    all_slopes, gs, ge = precompute_slopes(df, pv_scores)
    temporal_total = compute_temporal_total(pv_scores, all_slopes, gs, ge).astype(np.float32)

    print("\nCollapsing to one row per patient (max label, max score) ...")
    pat = collapse_patient_level(df, news2, snapshot_total, temporal_total)
    y = pat["label"].values.astype(np.int8)

    models = {
        "NEWS-2": pat["news2"].values.astype(np.float64),
        "Snapshot Fuzzy EWS": pat["snapshot"].values.astype(np.float64),
        "Temporal Builder": pat["temporal"].values.astype(np.float64),
    }

    rows = []
    print(f"Patients: {len(pat):,} | Positives: {y.sum():,} ({100*y.mean():.2f}%)")
    print("\nPatient-level AUROC:")
    for name, score in models.items():
        valid = np.isfinite(score)
        auc = roc_auc_score(y[valid], score[valid])
        rows.append({"Model": name, "Patient_Level_AUROC": round(float(auc), 6)})
        print(f"  {name:22s}  {auc:.6f}")

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "patient_level_auroc_summary.csv"
    summary.to_csv(summary_path, index=False)
    pat_path = OUT_DIR / "patient_level_case_scores.csv"
    pat.to_csv(pat_path, index=False)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {pat_path}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
