#!/usr/bin/env python3
"""
Patient-level rank diagnostics across NEWS-2, Snapshot, and Temporal systems.

Outputs:
  1) Rank correlation (Spearman + Kendall) between model scores.
  2) AUROC for each model (patient-level).
  3) Pairwise ranking reversal rate on positive-vs-negative pairs
     (Monte Carlo estimate for computational tractability).
"""

import time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

from grid_search_auroc import VITALS, compute_fuzzy_scores, load_data
from patient_level_auprc import (
    collapse_patient_level,
    compute_temporal_total,
    precompute_slopes,
)

RNG_SEED = 42
N_PAIR_SAMPLES = 1_000_000
OUT_DIR = Path("threshold_results")


def pairwise_pos_neg_reversal_rate(y, score_a, score_b, n_samples=N_PAIR_SAMPLES, seed=RNG_SEED):
    """
    Estimate ranking reversals between two models on positive-vs-negative pairs.

    For sampled (pos, neg) pairs:
      margin_a = score_a[pos] - score_a[neg]
      margin_b = score_b[pos] - score_b[neg]
    A strict reversal occurs when sign(margin_a) and sign(margin_b) are opposite.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {
            "reversal_rate": np.nan,
            "agree_rate": np.nan,
            "tie_in_any_rate": np.nan,
            "n_samples": 0,
        }

    pos_s = rng.choice(pos_idx, size=n_samples, replace=True)
    neg_s = rng.choice(neg_idx, size=n_samples, replace=True)

    margin_a = score_a[pos_s] - score_a[neg_s]
    margin_b = score_b[pos_s] - score_b[neg_s]

    sign_a = np.sign(margin_a)
    sign_b = np.sign(margin_b)
    tie_any = (sign_a == 0) | (sign_b == 0)
    strict = ~tie_any

    if strict.any():
        agree = (sign_a[strict] == sign_b[strict]).mean()
        reverse = (sign_a[strict] == -sign_b[strict]).mean()
    else:
        agree = np.nan
        reverse = np.nan

    return {
        "reversal_rate": float(reverse),
        "agree_rate": float(agree),
        "tie_in_any_rate": float(tie_any.mean()),
        "n_samples": int(n_samples),
    }


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("  Patient-Level Rank Diagnostics")
    print(f"  Pair samples per model-pair: {N_PAIR_SAMPLES:,}")
    print("=" * 74)

    df = load_data()
    pv_scores = compute_fuzzy_scores(df)
    snapshot_total = sum(pv_scores[v] for v in VITALS).astype(np.float32)
    news2 = df["NEWS-2"].values.astype(np.float32)
    all_slopes, gs, ge = precompute_slopes(df, pv_scores)
    temporal_total = compute_temporal_total(pv_scores, all_slopes, gs, ge)

    pat = collapse_patient_level(df, news2, snapshot_total, temporal_total)
    y = pat["label"].values.astype(np.int8)

    models = {
        "NEWS-2": pat["news2"].values.astype(np.float64),
        "Snapshot Fuzzy EWS": pat["snapshot"].values.astype(np.float64),
        "Temporal Builder": pat["temporal"].values.astype(np.float64),
    }
    names = list(models.keys())

    print(f"Patients: {len(y):,}  positives: {y.sum():,} ({100*y.mean():.2f}%)")
    print("\nAUROC (patient-level):")
    for n in names:
        auc = roc_auc_score(y, models[n])
        print(f"  {n:22s} {auc:.6f}")

    corr_rows = []
    rev_rows = []
    print("\nRank correlation + pairwise reversals:")
    for a, b in combinations(names, 2):
        sa, sb = models[a], models[b]
        rho, _ = spearmanr(sa, sb)
        tau, _ = kendalltau(sa, sb)
        corr_rows.append({
            "model_a": a,
            "model_b": b,
            "spearman_rho": float(rho),
            "kendall_tau": float(tau),
        })

        rev = pairwise_pos_neg_reversal_rate(y, sa, sb)
        rev_rows.append({
            "model_a": a,
            "model_b": b,
            **rev,
        })

        print(
            f"  {a:22s} vs {b:22s} | "
            f"rho={rho:.6f} tau={tau:.6f} | "
            f"reversal={rev['reversal_rate']:.6f} agree={rev['agree_rate']:.6f} "
            f"ties={rev['tie_in_any_rate']:.6f}"
        )

    pd.DataFrame(corr_rows).to_csv(OUT_DIR / "patient_level_rank_correlations.csv", index=False)
    pd.DataFrame(rev_rows).to_csv(OUT_DIR / "patient_level_pairwise_reversals.csv", index=False)
    pat.to_csv(OUT_DIR / "patient_level_scores.csv", index=False)

    print("\nSaved:")
    print(f"  {OUT_DIR / 'patient_level_rank_correlations.csv'}")
    print(f"  {OUT_DIR / 'patient_level_pairwise_reversals.csv'}")
    print(f"  {OUT_DIR / 'patient_level_scores.csv'}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()