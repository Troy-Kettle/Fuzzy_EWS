#!/usr/bin/env python3
"""
AUROC evaluation at optimal temporal context builder parameters
found by grid search:  α=0.70, β=0.5, γ=0.80

Compares:
  1. NEWS-2  (pre-computed in the dataset)
  2. Snapshot Fuzzy EWS  (no temporal adjustment)
  3. Temporal Context Builder  (EWMA + trend, optimal params)

Includes bootstrap 95 % confidence intervals, sensitivity/specificity
at clinically relevant thresholds, and publication-quality plots.
"""

import time
import warnings
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score,
)

import sys
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(over="ignore", invalid="ignore")

# ── Optimal parameters (from grid search) ────────────────────────────
ALPHA = 0.70
BETA  = 0.5
GAMMA = 0.80
WINDOW_HOURS = 24.0

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "20250630_final_observations-sorted_V7_training.csv"
SIGMOID_DIR = SCRIPT_DIR / "generated_membership_data" / "sigmoid"
OUTPUT_DIR = SCRIPT_DIR / "auroc_results"

# ── Re-use core logic from grid_search_auroc ──────────────────────────
from grid_search_auroc import (
    VITALS, VITAL_COL,
    _build_lookup, _ewma_all, _ols_slopes_for_group,
    load_data, compute_fuzzy_scores,
)

N_BOOTSTRAP = 200
RNG_SEED = 42
DIAG_SAMPLE_SIZE = 200_000

# =====================================================================
#  Slope computation (same as grid search, extracted here for clarity)
# =====================================================================

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
        print(f"  {time.time()-t0:.0f}s", flush=True)
    return all_slopes, group_starts, group_ends


# =====================================================================
#  Temporal scoring at fixed parameters
# =====================================================================

def compute_temporal_total(pv_scores, all_slopes, group_starts, group_ends):
    """Compute temporal-adjusted total at optimal α, β, γ."""
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

    if GAMMA == 1.0:
        return sum(adjusted[v] for v in VITALS)
    additive = sum(adjusted[v] for v in VITALS)
    stacked = np.column_stack([adjusted[v] for v in VITALS])
    max_vital = stacked.max(axis=1)
    max_based = (18.0 / 3.0) * max_vital
    return (1.0 - GAMMA) * max_based + GAMMA * additive


# =====================================================================
#  Bootstrap confidence intervals
# =====================================================================

def bootstrap_auroc(label, pred, patient_ids, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Patient-level bootstrap: resample patients (not observations).

    This is the correct approach for clustered data — observations from
    the same patient are kept together, avoiding underestimated variance.
    """
    rng = np.random.default_rng(seed)
    unique_pats = np.unique(patient_ids)
    n_pats = len(unique_pats)

    pat_to_idx = {}
    for i, pid in enumerate(patient_ids):
        pat_to_idx.setdefault(pid, []).append(i)
    pat_idx_arrays = {pid: np.array(idxs) for pid, idxs in pat_to_idx.items()}

    aucs = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique_pats, size=n_pats, replace=True)
        indices = np.concatenate([pat_idx_arrays[p] for p in sampled])
        y, p = label[indices], pred[indices]
        if y.sum() == 0 or y.sum() == len(y):
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(y, p)
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), lo, hi


def diagnose_score_similarity(scores: Dict[str, np.ndarray], seed: int = RNG_SEED):
    """Print diagnostics when models appear to have similar AUROC.

    AUROC is rank-based; different score magnitudes can still yield equal AUROC
    if rank ordering is near-identical. This helper makes that explicit.
    """
    print("\n── Score similarity diagnostics ───────────────────────────")
    names = list(scores.keys())
    rng = np.random.default_rng(seed)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            s1 = scores[n1]
            s2 = scores[n2]
            valid = np.isfinite(s1) & np.isfinite(s2)
            if not valid.any():
                print(f"  {n1} vs {n2}: no finite overlap")
                continue
            a = s1[valid]
            b = s2[valid]
            identical = np.array_equal(a, b)
            n = len(a)
            if n > DIAG_SAMPLE_SIZE:
                idx = rng.choice(n, size=DIAG_SAMPLE_SIZE, replace=False)
                a_s = a[idx]
                b_s = b[idx]
            else:
                a_s = a
                b_s = b

            uniq_a = len(np.unique(a_s))
            uniq_b = len(np.unique(b_s))
            pearson = np.corrcoef(a_s.astype(np.float64), b_s.astype(np.float64))[0, 1]
            rho = pd.Series(a_s).rank(method="average").corr(
                pd.Series(b_s).rank(method="average"), method="pearson"
            )
            print(
                f"  {n1:20s} vs {n2:20s} | identical={identical!s:5s} "
                f"| unique=({uniq_a},{uniq_b}) | pearson={pearson:.6f} | rank_rho={rho:.6f}"
            )


def evaluate_patient_level_auroc(
    label: np.ndarray, patient_ids: np.ndarray, scores: Dict[str, np.ndarray]
):
    """Report patient-level AUROC by collapsing repeated observations per patient.

    Observation-level AUROC can understate practical differences when patients have
    many repeated rows. This diagnostic evaluates one score per patient (max score)
    with one label per patient (any positive event).
    """
    print("\n── Patient-level AUROC (max score per patient) ───────────")
    base = pd.DataFrame({
        "patient_id": patient_ids,
        "label": label.astype(np.int8),
    })
    grouped_label = base.groupby("patient_id", sort=False)["label"].max()
    y_patient = grouped_label.values.astype(np.int8)
    print(
        f"  Patients: {len(grouped_label):,} | "
        f"Positive patients: {y_patient.sum():,} ({100 * y_patient.mean():.2f}%)"
    )

    for name, pred in scores.items():
        tmp = pd.DataFrame({
            "patient_id": patient_ids,
            "pred": pred,
        })
        tmp = tmp[np.isfinite(tmp["pred"])]
        if tmp.empty:
            print(f"  {name:30s}  no finite scores")
            continue
        pred_patient = tmp.groupby("patient_id", sort=False)["pred"].max()
        aligned = pd.concat([grouped_label, pred_patient], axis=1, join="inner").dropna()
        if aligned.empty:
            print(f"  {name:30s}  no overlapping patients")
            continue
        y = aligned["label"].values.astype(np.int8)
        p = aligned["pred"].values.astype(np.float64)
        auc = roc_auc_score(y, p)
        print(f"  {name:30s}  AUROC = {auc:.6f}")


# =====================================================================
#  Visualisations
# =====================================================================

def plot_roc(label, predictors, out_path):
    """Publication-quality ROC comparison."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colours = {"NEWS-2": "#E67E22", "Snapshot Fuzzy EWS": "#3498DB",
               "Temporal Builder": "#27AE60"}

    for name, pred, ci in predictors:
        valid = np.isfinite(pred)
        fpr, tpr, _ = roc_curve(label[valid], pred[valid])
        auc = roc_auc_score(label[valid], pred[valid])
        col = colours.get(name, "grey")
        ci_str = f"[{ci[0]:.4f}–{ci[1]:.4f}]" if ci else ""
        ax.plot(fpr, tpr, color=col, linewidth=2.2,
                label=f"{name}  AUROC={auc:.4f}  {ci_str}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=1)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=13)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    ax.set_title("ROC Curve Comparison — Optimal Temporal Parameters", fontsize=14)
    ax.legend(fontsize=10.5, loc="lower right",
              frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_zoomed(label, predictors, out_path):
    """Zoomed ROC in the high-specificity region (FPR 0–0.3)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = {"NEWS-2": "#E67E22", "Snapshot Fuzzy EWS": "#3498DB",
               "Temporal Builder": "#27AE60"}

    for name, pred, _ in predictors:
        valid = np.isfinite(pred)
        fpr, tpr, _ = roc_curve(label[valid], pred[valid])
        auc = roc_auc_score(label[valid], pred[valid])
        col = colours.get(name, "grey")
        ax.plot(fpr, tpr, color=col, linewidth=2.2,
                label=f"{name}  (AUROC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=1)
    ax.set_xlim(-0.005, 0.30)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=13)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    ax.set_title("ROC — High Specificity Region (FPR ≤ 0.3)", fontsize=14)
    ax.legend(fontsize=10.5, loc="lower right",
              frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(label, predictors, out_path):
    """Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = {"NEWS-2": "#E67E22", "Snapshot Fuzzy EWS": "#3498DB",
               "Temporal Builder": "#27AE60"}
    prevalence = label.mean()

    for name, pred, _ in predictors:
        valid = np.isfinite(pred)
        prec, rec, _ = precision_recall_curve(label[valid], pred[valid])
        ap = average_precision_score(label[valid], pred[valid])
        col = colours.get(name, "grey")
        ax.plot(rec, prec, color=col, linewidth=2.2,
                label=f"{name}  (AP={ap:.4f})")

    ax.axhline(prevalence, color="grey", linestyle=":", alpha=0.5,
               label=f"Prevalence ({prevalence:.4f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=13)
    ax.set_ylabel("Precision (PPV)", fontsize=13)
    ax.set_title("Precision–Recall Curve Comparison", fontsize=14)
    ax.legend(fontsize=10.5, loc="upper right",
              frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(label, predictors, out_path):
    """Score distributions for positive vs negative classes."""
    n = len(predictors)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    colours_pos = "#E74C3C"
    colours_neg = "#3498DB"

    for ax, (name, pred, _) in zip(axes, predictors):
        valid = np.isfinite(pred)
        pos = pred[valid & (label == 1)]
        neg = pred[valid & (label == 0)]
        bins = np.linspace(min(pred[valid].min(), 0), pred[valid].max(), 60)
        ax.hist(neg, bins=bins, alpha=0.6, color=colours_neg,
                label=f"Negative (n={len(neg):,})", density=True)
        ax.hist(pos, bins=bins, alpha=0.6, color=colours_pos,
                label=f"Positive (n={len(pos):,})", density=True)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Density", fontsize=12)
    fig.suptitle("Score Distributions by Outcome", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def sensitivity_at_specificity(label, pred, target_specs=[0.80, 0.85, 0.90, 0.95]):
    """Report sensitivity at given specificity thresholds."""
    valid = np.isfinite(pred)
    fpr, tpr, thresholds = roc_curve(label[valid], pred[valid])
    specificity = 1 - fpr
    rows = []
    for spec_target in target_specs:
        idx = np.argmin(np.abs(specificity - spec_target))
        rows.append({
            "Target Specificity": f"{spec_target:.0%}",
            "Actual Specificity": f"{specificity[idx]:.4f}",
            "Sensitivity (TPR)": f"{tpr[idx]:.4f}",
            "Threshold": f"{thresholds[idx]:.3f}" if idx < len(thresholds) else "—",
        })
    return pd.DataFrame(rows)


# =====================================================================
#  Main
# =====================================================================

def main():
    t_total = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  AUROC Evaluation — Optimal Temporal Parameters")
    print(f"  α={ALPHA}  β={BETA}  γ={GAMMA}  window={WINDOW_HOURS}h")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────
    df = load_data()
    label = df["REVIEW_WITHIN_4HOURS"].values.astype(np.int8)

    # ── Per-vital fuzzy scores ────────────────────────────────────────
    pv_scores = compute_fuzzy_scores(df)
    snapshot_total = sum(pv_scores[v] for v in VITALS).astype(np.float32)

    # ── Slopes ────────────────────────────────────────────────────────
    print("\nPrecomputing slopes …")
    all_slopes, gs, ge = precompute_slopes(df, pv_scores)

    # ── Temporal total ────────────────────────────────────────────────
    print("\nComputing temporal-adjusted total …", flush=True)
    t0 = time.time()
    temporal_total = compute_temporal_total(pv_scores, all_slopes, gs, ge)
    print(f"  Done in {time.time()-t0:.0f}s")

    news2 = df["NEWS-2"].values.astype(np.float32)

    # ── AUROC (point estimates) ───────────────────────────────────────
    print("\n── Point-estimate AUROC ─────────────────────────────────")
    scores = {
        "NEWS-2": news2,
        "Snapshot Fuzzy EWS": snapshot_total,
        "Temporal Builder": temporal_total,
    }
    diagnose_score_similarity(scores)
    for name, pred in scores.items():
        valid = np.isfinite(pred)
        auc = roc_auc_score(label[valid], pred[valid])
        print(f"  {name:30s}  AUROC = {auc:.6f}")
    evaluate_patient_level_auroc(label, df["ANON_ADMISSION_ID"].values, scores)

    # ── Bootstrap CIs ─────────────────────────────────────────────────
    patient_ids = df["ANON_ADMISSION_ID"].values
    print(f"\n── Patient-level bootstrap 95 % CI ({N_BOOTSTRAP} resamples) ──")
    ci_results = {}
    for name, pred in scores.items():
        valid = np.isfinite(pred)
        print(f"  Bootstrapping {name} …", end="", flush=True)
        t0 = time.time()
        mean_auc, lo, hi = bootstrap_auroc(
            label[valid], pred[valid], patient_ids[valid])
        ci_results[name] = (mean_auc, lo, hi)
        print(f"  {mean_auc:.6f}  [{lo:.6f} – {hi:.6f}]  ({time.time()-t0:.0f}s)")

    # ── Sensitivity at key specificity levels ─────────────────────────
    print("\n── Sensitivity at specificity thresholds ────────────────")
    for name, pred in scores.items():
        valid = np.isfinite(pred)
        table = sensitivity_at_specificity(label[valid], pred[valid])
        print(f"\n  {name}:")
        print(table.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n\nGenerating plots …")
    pred_list = [
        ("NEWS-2", news2, (ci_results["NEWS-2"][1], ci_results["NEWS-2"][2])),
        ("Snapshot Fuzzy EWS", snapshot_total,
         (ci_results["Snapshot Fuzzy EWS"][1], ci_results["Snapshot Fuzzy EWS"][2])),
        ("Temporal Builder", temporal_total,
         (ci_results["Temporal Builder"][1], ci_results["Temporal Builder"][2])),
    ]

    plot_roc(label, pred_list, OUTPUT_DIR / "roc_comparison.png")
    print("  Saved roc_comparison.png")

    plot_roc_zoomed(label, pred_list, OUTPUT_DIR / "roc_zoomed.png")
    print("  Saved roc_zoomed.png")

    plot_precision_recall(label, pred_list, OUTPUT_DIR / "precision_recall.png")
    print("  Saved precision_recall.png")

    plot_score_distributions(label, pred_list, OUTPUT_DIR / "score_distributions.png")
    print("  Saved score_distributions.png")

    # ── Summary table ─────────────────────────────────────────────────
    summary_rows = []
    for name, pred in scores.items():
        valid = np.isfinite(pred)
        auc = roc_auc_score(label[valid], pred[valid])
        m, lo, hi = ci_results[name]
        ap = average_precision_score(label[valid], pred[valid])
        summary_rows.append({
            "Model": name,
            "AUROC": round(auc, 6),
            "AUROC 95% CI lower": round(lo, 6),
            "AUROC 95% CI upper": round(hi, 6),
            "Avg Precision (AUPRC)": round(ap, 6),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "auroc_summary.csv", index=False)
    print("\n  Saved auroc_summary.csv")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\n  Δ AUROC (Temporal − NEWS-2):          "
          f"{ci_results['Temporal Builder'][0] - ci_results['NEWS-2'][0]:+.6f}")
    print(f"  Δ AUROC (Snapshot Fuzzy − NEWS-2):     "
          f"{ci_results['Snapshot Fuzzy EWS'][0] - ci_results['NEWS-2'][0]:+.6f}")
    print(f"{'='*70}")
    print(f"\nTotal wall time: {time.time()-t_total:.0f}s")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
