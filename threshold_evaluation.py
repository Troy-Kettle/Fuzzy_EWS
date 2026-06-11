#!/usr/bin/env python3
"""
Binary-threshold evaluation of Temporal Fuzzy EWS vs NEWS-2 for
predicting clinical review within 4 hours.

Thresholds:
  • Fuzzy EWS (temporal-adjusted, 0-18 scale):  ≥ 7
  • NEWS-2 (0-20 scale):                        ≥ 7

Metrics per system:
  Sensitivity, Specificity, PPV, NPV, Accuracy, F1,
  and full ROC curves with threshold marked.

Uses the same data pipeline and optimal temporal parameters
(α=0.70, β=0.5, γ=0.80) as the grid search.
"""

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(over="ignore", invalid="ignore")

# ── Optimal temporal parameters ──────────────────────────────────────
ALPHA = 0.70
BETA  = 0.5
GAMMA = 0.80
WINDOW_HOURS = 24.0

# ── Thresholds ────────────────────────────────────────────────────────
FUZZY_THRESHOLD = 7.0
NEWS_THRESHOLD  = 7

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "threshold_results"

from grid_search_auroc import (
    VITALS, _ewma_all, _ols_slopes_for_group,
    load_data, compute_fuzzy_scores,
)

# =====================================================================
#  Helpers (reused from auroc_optimal.py)
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
            slopes[s:e] = _ols_slopes_for_group(
                t_minutes[s:e], raw[s:e], window_min)
        all_slopes[vital] = slopes
        print(f"  {time.time()-t0:.0f}s", flush=True)
    return all_slopes, group_starts, group_ends


def compute_temporal_total(pv_scores, all_slopes, group_starts, group_ends):
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
    stacked = np.column_stack([adjusted[v] for v in VITALS])
    max_vital = stacked.max(axis=1)
    max_based = (18.0 / 3.0) * max_vital
    total = (1.0 - GAMMA) * max_based + GAMMA * additive
    return np.maximum(total, snapshot).astype(np.float32)


# =====================================================================
#  Metrics at a fixed threshold
# =====================================================================

def threshold_metrics(label, score, threshold, name=""):
    """Compute classification metrics at a fixed threshold."""
    pred_pos = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(label, pred_pos, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    f1          = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "System": name,
        "Threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Sensitivity": round(sensitivity, 6),
        "Specificity": round(specificity, 6),
        "PPV": round(ppv, 6),
        "NPV": round(npv, 6),
        "Accuracy": round(accuracy, 6),
        "F1": round(f1, 6),
    }


# =====================================================================
#  Plots
# =====================================================================

def plot_roc_with_thresholds(label, systems, out_path):
    """ROC curves with the chosen operating-point threshold marked."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colours = {
        "NEWS-2": "#E67E22",
        "Snapshot Fuzzy EWS": "#3498DB",
        "Temporal Fuzzy EWS": "#27AE60",
    }

    for name, score, thresh in systems:
        valid = np.isfinite(score)
        fpr, tpr, thresholds = roc_curve(label[valid], score[valid])
        auc = roc_auc_score(label[valid], score[valid])
        col = colours.get(name, "grey")
        ax.plot(fpr, tpr, color=col, linewidth=2.2,
                label=f"{name}  (AUROC={auc:.4f})")

        # Mark the operating point closest to the threshold
        pred_pos = (score[valid] >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            label[valid], pred_pos, labels=[0, 1]).ravel()
        op_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        op_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        ax.plot(op_fpr, op_tpr, "o", color=col, markersize=10,
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        ax.annotate(
            f"  threshold={thresh}",
            (op_fpr, op_tpr), fontsize=9, color=col,
            textcoords="offset points", xytext=(10, -5),
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=1)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=13)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    ax.set_title("ROC Curves with Operating-Point Thresholds", fontsize=14)
    ax.legend(fontsize=10.5, loc="lower right",
              frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(label, systems, out_path):
    """Side-by-side confusion matrices."""
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, score, thresh) in zip(axes, systems):
        pred_pos = (score >= thresh).astype(int)
        cm = confusion_matrix(label, pred_pos, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        for i in range(2):
            for j in range(2):
                colour = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=colour)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted −", "Predicted +"])
        ax.set_yticklabels(["Actual −", "Actual +"])
        ax.set_title(f"{name}\n(threshold ≥ {thresh})", fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(metrics_list, out_path):
    """Grouped bar chart comparing Sens, Spec, PPV, NPV across systems."""
    metric_names = ["Sensitivity", "Specificity", "PPV", "NPV", "F1"]
    systems = [m["System"] for m in metrics_list]
    colours = ["#E67E22", "#3498DB", "#27AE60"]

    x = np.arange(len(metric_names))
    width = 0.25
    offsets = np.linspace(-width, width, len(systems))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (m, col) in enumerate(zip(metrics_list, colours)):
        vals = [m[mn] for mn in metric_names]
        bars = ax.bar(x + offsets[i], vals, width * 0.9, label=m["System"],
                      color=col, edgecolor="grey", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Classification Metrics at Fixed Thresholds", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(label, systems, out_path):
    """Sensitivity & Specificity vs threshold for each system."""
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, score, chosen_thresh) in zip(axes, systems):
        valid = np.isfinite(score)
        fpr, tpr, thresholds = roc_curve(label[valid], score[valid])
        specificity = 1 - fpr

        ax.plot(thresholds, tpr[:len(thresholds)], color="#27AE60",
                linewidth=2, label="Sensitivity")
        ax.plot(thresholds, specificity[:len(thresholds)], color="#E74C3C",
                linewidth=2, label="Specificity")
        ax.axvline(chosen_thresh, color="black", linestyle="--", alpha=0.6,
                   label=f"Threshold = {chosen_thresh}")

        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=9, loc="center left")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(score[valid].min(), score[valid].max())

    fig.suptitle("Sensitivity & Specificity vs Threshold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(label, systems, out_path):
    """Precision-Recall curves with operating-point thresholds marked."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colours = {
        "NEWS-2": "#E67E22",
        "Snapshot Fuzzy EWS": "#3498DB",
        "Temporal Fuzzy EWS": "#27AE60",
    }
    prevalence = label.mean()

    for name, score, thresh in systems:
        valid = np.isfinite(score)
        prec, rec, _ = precision_recall_curve(label[valid], score[valid])
        ap = average_precision_score(label[valid], score[valid])
        col = colours.get(name, "grey")
        ax.plot(rec, prec, color=col, linewidth=2.2,
                label=f"{name}  (AP={ap:.4f})")

        pred_pos = (score[valid] >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            label[valid], pred_pos, labels=[0, 1]).ravel()
        op_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        op_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        ax.plot(op_rec, op_prec, "o", color=col, markersize=10,
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        ax.annotate(
            f"  threshold={thresh}",
            (op_rec, op_prec), fontsize=9, color=col,
            textcoords="offset points", xytext=(10, -5),
        )

    ax.axhline(prevalence, color="grey", linestyle=":", alpha=0.5,
               label=f"Prevalence ({prevalence:.4f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=13)
    ax.set_ylabel("Precision (PPV)", fontsize=13)
    ax.set_title("Precision-Recall Curves with Operating-Point Thresholds", fontsize=14)
    ax.legend(fontsize=10, loc="upper right",
              frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
#  Main
# =====================================================================

def main():
    t_total = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Threshold Evaluation — Fuzzy EWS ≥ 7  vs  NEWS-2 ≥ 7")
    print(f"  Temporal params: α={ALPHA}  β={BETA}  γ={GAMMA}")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────
    df = load_data()
    label = df["REVIEW_WITHIN_4HOURS"].values.astype(np.int8)

    # ── Per-vital fuzzy scores ────────────────────────────────────────
    pv_scores = compute_fuzzy_scores(df)
    snapshot_total = sum(pv_scores[v] for v in VITALS).astype(np.float32)

    # ── Slopes + temporal total ───────────────────────────────────────
    print("\nPrecomputing slopes …")
    all_slopes, gs, ge = precompute_slopes(df, pv_scores)

    print("\nComputing temporal-adjusted total …", flush=True)
    t0 = time.time()
    temporal_total = compute_temporal_total(pv_scores, all_slopes, gs, ge)
    temporal_total = temporal_total.astype(np.float32)
    print(f"  Done in {time.time()-t0:.0f}s")

    news2 = df["NEWS-2"].values.astype(np.float32)

    # ── Metrics at fixed thresholds ───────────────────────────────────
    print("\n── Classification metrics at fixed thresholds ───────────\n")

    m_news    = threshold_metrics(label, news2, NEWS_THRESHOLD,
                                  f"NEWS-2 (≥{NEWS_THRESHOLD})")
    m_snap    = threshold_metrics(label, snapshot_total, FUZZY_THRESHOLD,
                                  f"Snapshot Fuzzy (≥{FUZZY_THRESHOLD})")
    m_temporal = threshold_metrics(label, temporal_total, FUZZY_THRESHOLD,
                                  f"Temporal Fuzzy (≥{FUZZY_THRESHOLD})")

    metrics_list = [m_news, m_snap, m_temporal]
    metrics_df = pd.DataFrame(metrics_list)

    col_order = [
        "System", "Threshold", "TP", "FP", "FN", "TN",
        "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1",
    ]
    metrics_df = metrics_df[col_order]
    print(metrics_df.to_string(index=False))

    # ── AUROC for reference ───────────────────────────────────────────
    print("\n── AUROC (continuous scores, for reference) ─────────────")
    for name, pred in [("NEWS-2", news2),
                       ("Snapshot Fuzzy", snapshot_total),
                       ("Temporal Fuzzy", temporal_total)]:
        valid = np.isfinite(pred)
        auc = roc_auc_score(label[valid], pred[valid])
        print(f"  {name:25s}  AUROC = {auc:.6f}")

    # ── Save results ──────────────────────────────────────────────────
    metrics_df.to_csv(OUTPUT_DIR / "threshold_metrics.csv", index=False)
    print(f"\n  Saved threshold_metrics.csv")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    systems = [
        ("NEWS-2", news2, NEWS_THRESHOLD),
        ("Snapshot Fuzzy EWS", snapshot_total, FUZZY_THRESHOLD),
        ("Temporal Fuzzy EWS", temporal_total, FUZZY_THRESHOLD),
    ]

    plot_roc_with_thresholds(label, systems, OUTPUT_DIR / "roc_with_thresholds.png")
    print("  Saved roc_with_thresholds.png")

    plot_confusion_matrices(label, systems, OUTPUT_DIR / "confusion_matrices.png")
    print("  Saved confusion_matrices.png")

    plot_metrics_comparison(metrics_list, OUTPUT_DIR / "metrics_comparison.png")
    print("  Saved metrics_comparison.png")

    plot_threshold_sweep(label, systems, OUTPUT_DIR / "threshold_sweep.png")
    print("  Saved threshold_sweep.png")

    plot_precision_recall(label, systems, OUTPUT_DIR / "precision_recall.png")
    print("  Saved precision_recall.png")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for m in metrics_list:
        print(f"\n  {m['System']}:")
        print(f"    Sensitivity  = {m['Sensitivity']:.4f}    "
              f"Specificity = {m['Specificity']:.4f}")
        print(f"    PPV          = {m['PPV']:.4f}    "
              f"NPV         = {m['NPV']:.4f}")
        print(f"    TP={m['TP']:>8,}  FP={m['FP']:>10,}  "
              f"FN={m['FN']:>8,}  TN={m['TN']:>10,}")
    print(f"\n{'='*70}")
    print(f"Total wall time: {time.time()-t_total:.0f}s")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
