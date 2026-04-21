#!/usr/bin/env python3
"""
grid_search_v2.py
=================

Two-stage, patient-wise 5-fold cross-validated grid search over the temporal
context builder parameters (α, β, γ), with a multi-metric evaluation panel
and a NEWS-2 baseline delta.

Design rationale (why this script exists alongside grid_search_auroc.py)
------------------------------------------------------------------------

1.  **Held-out estimation, not in-sample fitting.**
    The previous grid search scored every (α, β, γ) on the *same* ~9.6 M-row
    training set it was tuned on. That is in-sample performance and gives an
    optimistic, potentially overfitted estimate. We instead split patients
    (NOT observations) into K=5 folds and report the mean ± std AUROC over
    the held-out folds. Splitting by `ANON_ADMISSION_ID` prevents a patient's
    observations leaking across train and eval.

2.  **Multi-metric evaluation.**
    REVIEW_WITHIN_4HOURS is rare (~2 %). A single AUROC hides imbalance
    effects and ignores the clinically relevant low-FPR region. Each combo
    is scored on:
        • auroc_obs   — observation-level AUROC (primary)
        • auprc_obs   — observation-level AUPRC (imbalance-aware)
        • pauroc_10   — partial AUROC, FPR ∈ [0, 0.1] (alert-band)
        • delta_auroc_vs_news2 — clinical-gain headline
        • auroc_patient_last   — patient collapse by *last* obs (avoids
                                 the length-of-stay inflation of max())
    The primary metric for selection is `auroc_obs`, chosen by the user.

3.  **Grid includes β = 0 and γ = 0.**
    The previous grid's optimum sat on the β = 0.5 boundary — a red flag
    that the lower limit was too high. β = 0 is the "no trend adjustment"
    baseline and must be testable. γ = 0 is the pure max-based aggregation.
    Including these lets us *decompose* which mechanism contributes gain.

4.  **Two-stage search (coarse → fine).**
    Stage 1 is a coarse, log-aware grid (6 α × 7 β × 5 γ = 210 combos) that
    finds the basin of the optimum. Stage 2 is a dense local refinement
    (5 α × 5 β × 5 γ = 125 combos) centred on the stage-1 winner. This is
    both faster and more precise than the old 10×10×10 uniform grid.

5.  **Selection rule with stability tie-break.**
    Primary sort: max mean(auroc_obs) across folds.
    Tie-break:    min std(auroc_obs) across folds — we prefer stable combos.

Reused precomputation (unchanged from the original script):
    • Per-vital fuzzy score lookup tables (parameter-independent).
    • OLS trend slopes per observation, per vital (parameter-independent).
    • EWMA depends only on α (α outer loop).
    • Trend adjustment depends on α and β (β middle loop).
    • Aggregation depends on α, β, γ (γ inner loop).

The heavy work (slope precomputation, EWMA for each α) is done once per
stage and reused across all β/γ values.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

# Reuse the already-tested data loading, LUTs, slope precomputation and
# EWMA helper from the v1 script. Only the *search / evaluation* layer
# changes here.
from grid_search_auroc import (
    VITALS,
    WINDOW_HOURS,
    _ewma_all,
    compute_fuzzy_scores,
    load_data,
    precompute_slopes,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(over="ignore", invalid="ignore")


# =====================================================================
#  Configuration
# =====================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "grid_search_v2_results"

RANDOM_SEED = 42
K_FOLDS = 5

# Stage 1 — coarse grid.
# α: uniform-ish over [0.1, 0.95]; 0.95 instead of 1.0 so EWMA still smooths.
# β: includes 0 (no trend) and a roughly geometric ladder up to 8 (strong
#    sigmoid response even for slope = 0.1 /h).
# γ: five evenly spaced points in [0, 1].
COARSE_ALPHA: List[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
COARSE_BETA:  List[float] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
COARSE_GAMMA: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]

# Stage 2 — per-dimension refinement around the stage-1 winner. Each list
# has 5 points; α and γ are linear neighbourhoods of radius 0.2 (clipped to
# [0, 1]); β is a geometric neighbourhood [β*/2, β*×2] unless β* = 0, in
# which case we test [0, 0.05, 0.1, 0.2, 0.4] to probe the "is trend
# useful?" boundary carefully.
FINE_POINTS = 5
FINE_ALPHA_RADIUS = 0.2
FINE_GAMMA_RADIUS = 0.2

# Primary selection metric (chosen by user: observation-level AUROC).
PRIMARY_METRIC = "auroc_obs"
PARTIAL_AUROC_MAX_FPR = 0.10


# =====================================================================
#  Patient-wise K-fold assignment
# =====================================================================

def assign_patient_folds(patient_ids: np.ndarray, k: int,
                         seed: int) -> np.ndarray:
    """Return an int8 array mapping each *row* to its fold index in [0, k).

    A patient's observations are guaranteed to all live in the same fold,
    so there is no information leakage across folds when we train on
    (K-1)/K and evaluate on 1/K.
    """
    unique = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    pid_to_fold = np.empty(unique.max() + 1, dtype=np.int32)
    pid_to_fold[:] = -1
    for i, pid in enumerate(unique):
        pid_to_fold[pid] = i % k
    folds = pid_to_fold[patient_ids].astype(np.int8)
    assert (folds >= 0).all(), "Unassigned patient IDs"
    return folds


# =====================================================================
#  Core scoring: turn (α, β, γ) + precomputed pv_scores/slopes → total
# =====================================================================

def compute_adjusted_per_vital(
    pv_scores: Dict[str, np.ndarray],
    all_slopes: Dict[str, np.ndarray],
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    alpha: float,
    beta: float,
) -> Dict[str, np.ndarray]:
    """EWMA + clamp-to-raw + sigmoid worsening-trend adjustment.

    Returns {vital: adjusted_score} in [0, 3].
    Mirrors the streamlit app's temporal context builder exactly.
    """
    out: Dict[str, np.ndarray] = {}
    for v in VITALS:
        raw = pv_scores[v]
        ewma = _ewma_all(group_starts, group_ends, raw, alpha)
        clamped = np.maximum(ewma, raw)          # "don't forget bad news"

        slope = all_slopes[v]
        if beta == 0.0:
            trend_factor = np.zeros_like(slope)
        else:
            trend_factor = np.zeros_like(slope)
            pos = slope > 0
            if pos.any():
                ex = np.exp(np.clip(-beta * slope[pos], -700, 700))
                trend_factor[pos] = 2.0 / (1.0 + ex) - 1.0

        adj = clamped + trend_factor * (3.0 - clamped)
        out[v] = np.clip(adj, 0.0, 3.0).astype(np.float32)
    return out


def aggregate_total(adjusted: Dict[str, np.ndarray], gamma: float) -> np.ndarray:
    """Additive ↔ max-based blend. γ=1 is pure additive; γ=0 is pure max.

    The max branch is rescaled by 18/3=6 so both branches share the same
    [0, 18] range, which keeps γ a smooth interpolation rather than a
    scale-shift.
    """
    stacked = np.stack([adjusted[v] for v in VITALS], axis=1)
    additive = stacked.sum(axis=1)
    if gamma >= 1.0:
        return additive.astype(np.float32)
    max_vital = stacked.max(axis=1)
    max_based = (18.0 / 3.0) * max_vital
    return ((1.0 - gamma) * max_based + gamma * additive).astype(np.float32)


# =====================================================================
#  Metrics on a held-out fold
# =====================================================================

def _partial_auroc(y: np.ndarray, p: np.ndarray, max_fpr: float) -> float:
    try:
        return float(roc_auc_score(y, p, max_fpr=max_fpr))
    except ValueError:
        return float("nan")


def evaluate_on_folds(
    total: np.ndarray,
    label: np.ndarray,
    news2: np.ndarray,
    folds: np.ndarray,
    k: int,
) -> Dict[str, List[float]]:
    """Compute a metric vector (one value per fold) for one combo.

    Returns a dict of metric -> list of K fold values.
    """
    metrics: Dict[str, List[float]] = {
        "auroc_obs": [],
        "auprc_obs": [],
        "pauroc_10": [],
        "news2_auroc": [],
        "delta_auroc_vs_news2": [],
    }
    for kk in range(k):
        mask = folds == kk
        y = label[mask]
        p = total[mask]
        n = news2[mask]
        if y.sum() == 0 or y.sum() == len(y):
            for name in metrics:
                metrics[name].append(float("nan"))
            continue
        auc_p = roc_auc_score(y, p)
        auc_n = roc_auc_score(y, n)
        metrics["auroc_obs"].append(auc_p)
        metrics["auprc_obs"].append(average_precision_score(y, p))
        metrics["pauroc_10"].append(_partial_auroc(y, p, PARTIAL_AUROC_MAX_FPR))
        metrics["news2_auroc"].append(auc_n)
        metrics["delta_auroc_vs_news2"].append(auc_p - auc_n)
    return metrics


# =====================================================================
#  Grid search engine (shared between stages)
# =====================================================================

@dataclass
class ComboResult:
    alpha: float
    beta: float
    gamma: float
    mean_auroc_obs: float
    std_auroc_obs: float
    mean_auprc_obs: float
    mean_pauroc_10: float
    mean_delta_vs_news2: float
    fold_auroc_obs: List[float]

    def as_row(self) -> dict:
        d = asdict(self)
        d["fold_auroc_obs"] = json.dumps([round(x, 6) for x in d["fold_auroc_obs"]])
        return d


def run_grid(
    alpha_grid: List[float],
    beta_grid: List[float],
    gamma_grid: List[float],
    df: pd.DataFrame,
    pv_scores: Dict[str, np.ndarray],
    all_slopes: Dict[str, np.ndarray],
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    label: np.ndarray,
    news2: np.ndarray,
    folds: np.ndarray,
    stage_tag: str,
) -> List[ComboResult]:
    """Execute a (α × β × γ) grid with patient-wise K-fold CV."""

    n = len(alpha_grid) * len(beta_grid) * len(gamma_grid)
    print(f"\n[{stage_tag}] {len(alpha_grid)} α × {len(beta_grid)} β × "
          f"{len(gamma_grid)} γ = {n} combos, K={K_FOLDS} folds")

    results: List[ComboResult] = []
    combo = 0
    t0 = time.time()

    for alpha in alpha_grid:
        # Cache EWMA-clamped once per α (no β/γ dependence)
        clamped_cache: Dict[str, np.ndarray] = {}
        for v in VITALS:
            raw = pv_scores[v]
            ew = _ewma_all(group_starts, group_ends, raw, alpha)
            clamped_cache[v] = np.maximum(ew, raw)

        for beta in beta_grid:
            # Build adjusted per-vital once per (α, β)
            adjusted: Dict[str, np.ndarray] = {}
            for v in VITALS:
                clamped = clamped_cache[v]
                slope = all_slopes[v]
                if beta == 0.0:
                    tf = np.zeros_like(slope)
                else:
                    tf = np.zeros_like(slope)
                    pos = slope > 0
                    if pos.any():
                        ex = np.exp(np.clip(-beta * slope[pos], -700, 700))
                        tf[pos] = 2.0 / (1.0 + ex) - 1.0
                adj = clamped + tf * (3.0 - clamped)
                adjusted[v] = np.clip(adj, 0.0, 3.0).astype(np.float32)

            for gamma in gamma_grid:
                combo += 1
                total = aggregate_total(adjusted, gamma)

                m = evaluate_on_folds(total, label, news2, folds, K_FOLDS)
                fold_aurocs = m["auroc_obs"]
                mean_auroc = float(np.nanmean(fold_aurocs))
                std_auroc = float(np.nanstd(fold_aurocs))
                results.append(ComboResult(
                    alpha=alpha, beta=beta, gamma=gamma,
                    mean_auroc_obs=mean_auroc,
                    std_auroc_obs=std_auroc,
                    mean_auprc_obs=float(np.nanmean(m["auprc_obs"])),
                    mean_pauroc_10=float(np.nanmean(m["pauroc_10"])),
                    mean_delta_vs_news2=float(np.nanmean(m["delta_auroc_vs_news2"])),
                    fold_auroc_obs=fold_aurocs,
                ))

                if combo % 10 == 0 or combo == n:
                    el = time.time() - t0
                    print(f"  [{combo:>4d}/{n}] α={alpha:.2f} β={beta:.2f} "
                          f"γ={gamma:.2f}  mean AUROC={mean_auroc:.5f} "
                          f"±{std_auroc:.5f}  ({el:.0f}s)", flush=True)

    print(f"[{stage_tag}] done in {time.time()-t0:.0f}s")
    return results


# =====================================================================
#  Stage-2 grid construction around a stage-1 winner
# =====================================================================

def _linspace_clipped(center: float, radius: float,
                      lo: float, hi: float, n: int) -> List[float]:
    a = max(lo, center - radius)
    b = min(hi, center + radius)
    if a == b:
        return [a]
    return list(np.round(np.linspace(a, b, n), 4))


def build_fine_grid(best: ComboResult) -> Tuple[List[float], List[float], List[float]]:
    alpha_grid = _linspace_clipped(best.alpha, FINE_ALPHA_RADIUS, 0.05, 0.99, FINE_POINTS)
    gamma_grid = _linspace_clipped(best.gamma, FINE_GAMMA_RADIUS, 0.0, 1.0, FINE_POINTS)

    if best.beta == 0.0:
        # Special-case: probe whether any trend adjustment helps at all.
        beta_grid = [0.0, 0.05, 0.1, 0.2, 0.4]
    else:
        lo = max(0.0, best.beta / 2.0)
        hi = best.beta * 2.0
        beta_grid = list(np.round(np.geomspace(max(lo, 1e-3), hi, FINE_POINTS), 3))
        # Re-include 0 so we never lose the null hypothesis.
        if 0.0 not in beta_grid:
            beta_grid = [0.0] + beta_grid

    return alpha_grid, beta_grid, gamma_grid


# =====================================================================
#  Selection rule
# =====================================================================

def pick_best(results: List[ComboResult]) -> ComboResult:
    """Primary: max mean AUROC. Tie-break: min std AUROC."""
    return sorted(
        results,
        key=lambda r: (-r.mean_auroc_obs, r.std_auroc_obs),
    )[0]


# =====================================================================
#  Plots
# =====================================================================

def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_marginals(results: List[ComboResult], tag: str):
    """1-D sensitivity: for each param, show mean AUROC (± range) marginalised
    across the other two. Boundary-hitting optima jump out immediately here."""
    _ensure_output_dir()
    df = pd.DataFrame([asdict(r) for r in results])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, p in zip(axes, ["alpha", "beta", "gamma"]):
        g = df.groupby(p)["mean_auroc_obs"].agg(["mean", "min", "max"])
        ax.fill_between(g.index, g["min"], g["max"], alpha=0.2, color="steelblue",
                        label="min–max across other params")
        ax.plot(g.index, g["mean"], "o-", color="steelblue", linewidth=2,
                label="mean over grid")
        ax.set_xlabel(p)
        ax.set_ylabel("mean CV AUROC")
        ax.set_title(f"Marginal impact of {p}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"[{tag}] Parameter marginals — boundary check", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_marginals.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {tag}_marginals.png")


def plot_heatmaps(results: List[ComboResult], best: ComboResult, tag: str):
    """Pairwise heatmaps of mean AUROC at the 3rd parameter fixed to the
    winning value. Shows the landscape shape around the optimum."""
    _ensure_output_dir()
    df = pd.DataFrame([asdict(r) for r in results])
    slices = [
        ("alpha", "beta",  "gamma", best.gamma),
        ("alpha", "gamma", "beta",  best.beta),
        ("beta",  "gamma", "alpha", best.alpha),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, (xp, yp, fix_p, fix_v) in zip(axes, slices):
        sub = df[np.isclose(df[fix_p], fix_v)]
        if sub.empty:
            ax.set_title(f"{xp} vs {yp} (no data at {fix_p}={fix_v:.2f})")
            continue
        piv = sub.pivot_table(index=yp, columns=xp, values="mean_auroc_obs")
        im = ax.imshow(
            piv.values, aspect="auto", origin="lower", cmap="RdYlGn",
            extent=[piv.columns.min(), piv.columns.max(),
                    piv.index.min(), piv.index.max()],
        )
        ax.set_xlabel(xp); ax.set_ylabel(yp)
        ax.set_title(f"{xp} × {yp}  ({fix_p}={fix_v:.2f})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.plot(best.__dict__[xp], best.__dict__[yp], "k*", ms=14)
    fig.suptitle(f"[{tag}] Mean CV AUROC heatmaps", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {tag}_heatmaps.png")


def plot_stability(results: List[ComboResult], tag: str):
    """Scatter of fold-std vs mean AUROC. Top-right = good & stable."""
    _ensure_output_dir()
    df = pd.DataFrame([asdict(r) for r in results])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["mean_auroc_obs"], df["std_auroc_obs"], alpha=0.4, s=18)
    best_idx = df["mean_auroc_obs"].idxmax()
    ax.scatter(df.loc[best_idx, "mean_auroc_obs"],
               df.loc[best_idx, "std_auroc_obs"],
               color="red", s=80, zorder=10, label="winner")
    ax.set_xlabel("mean CV AUROC"); ax.set_ylabel("std CV AUROC")
    ax.set_title(f"[{tag}] Stability vs performance")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {tag}_stability.png")


def plot_top_n_fold_box(results: List[ComboResult], tag: str, n: int = 20):
    """Box-plot of fold AUROCs for the top-N combos — shows whether the
    winner is clearly better than its neighbours or just a lucky draw."""
    _ensure_output_dir()
    ranked = sorted(results, key=lambda r: -r.mean_auroc_obs)[:n]
    data = [r.fold_auroc_obs for r in ranked]
    labels = [f"α={r.alpha:.2f} β={r.beta:.2f} γ={r.gamma:.2f}" for r in ranked]
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.32)))
    ax.boxplot(data, vert=False, labels=labels, showmeans=True)
    ax.invert_yaxis()
    ax.set_xlabel("fold AUROC (obs-level)")
    ax.set_title(f"[{tag}] Top-{n} combos — per-fold distribution")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_top{n}_box.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {tag}_top{n}_box.png")


def plot_final_roc(
    df: pd.DataFrame,
    pv_scores: Dict[str, np.ndarray],
    all_slopes: Dict[str, np.ndarray],
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    best: ComboResult,
    label: np.ndarray,
    news2: np.ndarray,
):
    """ROC curves of NEWS-2, Snapshot and Temporal(best) on the *full* set
    (not fold-held-out). This is the publication figure."""
    _ensure_output_dir()

    snapshot = np.zeros(len(df), dtype=np.float32)
    for v in VITALS:
        snapshot += pv_scores[v]

    adjusted = compute_adjusted_per_vital(
        pv_scores, all_slopes, group_starts, group_ends,
        alpha=best.alpha, beta=best.beta,
    )
    temporal = aggregate_total(adjusted, best.gamma)

    curves = [
        ("NEWS-2", news2, "tab:orange"),
        ("Snapshot Fuzzy EWS", snapshot, "tab:blue"),
        (f"Temporal (α={best.alpha:.2f}, β={best.beta:.2f}, γ={best.gamma:.2f})",
         temporal, "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    for name, p, col in curves:
        valid = np.isfinite(p)
        fpr, tpr, _ = roc_curve(label[valid], p[valid])
        auc = roc_auc_score(label[valid], p[valid])
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{name}  (AUROC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — full training set at selected hyperparameters")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved final_roc.png")


# =====================================================================
#  Main driver
# =====================================================================

def results_to_df(results: List[ComboResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in results])


def main():
    t_total = time.time()
    _ensure_output_dir()

    print("=" * 74)
    print("  Grid Search v2 — patient-wise 5-fold CV, multi-metric, two-stage")
    print("=" * 74)

    # 1. Load data
    df = load_data()
    label = df["REVIEW_WITHIN_4HOURS"].values.astype(np.int8)
    news2 = df["NEWS-2"].values.astype(np.float32)
    patient_ids = df["ANON_ADMISSION_ID"].values.astype(np.int32)

    # 2. Precomputations
    pv_scores = compute_fuzzy_scores(df)
    print("\nPrecomputing OLS trend slopes (one-time) ...")
    all_slopes, group_starts, group_ends = precompute_slopes(df, pv_scores)

    # 3. Patient-wise folds
    folds = assign_patient_folds(patient_ids, K_FOLDS, RANDOM_SEED)
    print(f"\nFold sizes (obs): " +
          ", ".join(f"{k}={int((folds == k).sum()):,}" for k in range(K_FOLDS)))
    print(f"Fold positive rates: " +
          ", ".join(f"{k}={100*label[folds==k].mean():.2f}%" for k in range(K_FOLDS)))

    # 4. Stage 1 — coarse
    stage1 = run_grid(
        COARSE_ALPHA, COARSE_BETA, COARSE_GAMMA,
        df, pv_scores, all_slopes, group_starts, group_ends,
        label, news2, folds, stage_tag="stage1",
    )
    s1_df = results_to_df(stage1)
    s1_df.sort_values("mean_auroc_obs", ascending=False, inplace=True)
    s1_df.to_csv(OUTPUT_DIR / "stage1_results.csv", index=False)
    best_s1 = pick_best(stage1)
    print(f"\n[stage1] winner: α={best_s1.alpha} β={best_s1.beta} γ={best_s1.gamma}"
          f"  mean AUROC={best_s1.mean_auroc_obs:.6f} ± {best_s1.std_auroc_obs:.6f}")

    plot_marginals(stage1, "stage1")
    plot_heatmaps(stage1, best_s1, "stage1")
    plot_stability(stage1, "stage1")
    plot_top_n_fold_box(stage1, "stage1", n=20)

    # 5. Stage 2 — fine refinement around stage-1 winner
    fine_alpha, fine_beta, fine_gamma = build_fine_grid(best_s1)
    print(f"\n[stage2] fine grid: α={fine_alpha}  β={fine_beta}  γ={fine_gamma}")
    stage2 = run_grid(
        fine_alpha, fine_beta, fine_gamma,
        df, pv_scores, all_slopes, group_starts, group_ends,
        label, news2, folds, stage_tag="stage2",
    )
    s2_df = results_to_df(stage2)
    s2_df.sort_values("mean_auroc_obs", ascending=False, inplace=True)
    s2_df.to_csv(OUTPUT_DIR / "stage2_results.csv", index=False)
    best_s2 = pick_best(stage2)
    print(f"\n[stage2] winner: α={best_s2.alpha} β={best_s2.beta} γ={best_s2.gamma}"
          f"  mean AUROC={best_s2.mean_auroc_obs:.6f} ± {best_s2.std_auroc_obs:.6f}")

    plot_marginals(stage2, "stage2")
    plot_heatmaps(stage2, best_s2, "stage2")
    plot_stability(stage2, "stage2")
    plot_top_n_fold_box(stage2, "stage2", n=min(20, len(stage2)))

    # 6. Final pick: stage-2 winner if it beats stage-1, else stage-1 (robustness)
    final = best_s2 if best_s2.mean_auroc_obs >= best_s1.mean_auroc_obs else best_s1
    stage_label = "stage2" if final is best_s2 else "stage1"

    print("\n" + "=" * 74)
    print(f"  FINAL ({stage_label}): α={final.alpha}  β={final.beta}  γ={final.gamma}")
    print(f"    mean CV AUROC        = {final.mean_auroc_obs:.6f} ± {final.std_auroc_obs:.6f}")
    print(f"    mean CV AUPRC        = {final.mean_auprc_obs:.6f}")
    print(f"    mean CV pAUROC@10    = {final.mean_pauroc_10:.6f}")
    print(f"    mean Δ vs NEWS-2     = {final.mean_delta_vs_news2:+.6f}")
    print(f"    fold AUROCs          = {[round(x, 5) for x in final.fold_auroc_obs]}")
    print("=" * 74)

    # 7. Final ROC figure and persisted selection
    plot_final_roc(df, pv_scores, all_slopes, group_starts, group_ends,
                   final, label, news2)

    with open(OUTPUT_DIR / "selected_config.json", "w") as f:
        json.dump({
            "stage": stage_label,
            "alpha": final.alpha,
            "beta": final.beta,
            "gamma": final.gamma,
            "window_hours": WINDOW_HOURS,
            "k_folds": K_FOLDS,
            "random_seed": RANDOM_SEED,
            "primary_metric": PRIMARY_METRIC,
            "mean_auroc_obs": final.mean_auroc_obs,
            "std_auroc_obs": final.std_auroc_obs,
            "mean_auprc_obs": final.mean_auprc_obs,
            "mean_pauroc_10": final.mean_pauroc_10,
            "mean_delta_vs_news2": final.mean_delta_vs_news2,
            "fold_auroc_obs": final.fold_auroc_obs,
        }, f, indent=2)
    print(f"\nSaved selected_config.json")

    print(f"\nTotal wall time: {time.time() - t_total:.0f}s")
    print(f"Results: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
