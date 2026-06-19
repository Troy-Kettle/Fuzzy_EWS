#!/usr/bin/env python3
"""
Grid search over temporal context builder parameters (α, β, γ) and
AUROC comparison against NEWS-2.

Optimised for the ~9.6 M-row training set:
  • Per-vital fuzzy scores are precomputed via 1-D lookup tables.
  • OLS trend slopes are computed once (parameter-independent).
  • Only the EWMA (α), sigmoid factor (β) and aggregation (γ) are
    recomputed inside the grid search loop.
"""

import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.metrics import roc_auc_score, roc_curve

import sys
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(over="ignore", invalid="ignore")

# ── paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "20250630_final_observations-sorted_V7_training.csv"
SIGMOID_DIR = SCRIPT_DIR / "generated_membership_data" / "sigmoid"
OUTPUT_DIR = SCRIPT_DIR / "grid_search_results"

# ── grid ─────────────────────────────────────────────────────────────
ALPHA_GRID = np.round(np.arange(0.1, 1.05, 0.1), 2).tolist()   # 10 vals
BETA_GRID  = np.round(np.arange(0.5, 5.05, 0.5), 2).tolist()   # 10 vals
GAMMA_GRID = np.round(np.arange(0.1, 1.05, 0.1), 2).tolist()   # 10 vals
WINDOW_HOURS = 24.0
# Reference spacing for α: α_eff = 1 - (1-α)^(Δt / EWMA_REF_MINUTES)
EWMA_REF_MINUTES = 60.0

# ── vital-sign configuration ────────────────────────────────────────
VITALS = [
    "heart_rate", "blood_pressure", "temperature",
    "respiratory_rate", "oxygen_saturation", "inspired_oxygen",
    "avpu_acvpu",
]
VITAL_COL = {
    "heart_rate":        "HEART_RATE",
    "blood_pressure":    "SYSTOLIC_BP",
    "temperature":       "TEMPERATURE",
    "respiratory_rate":  "RESP_RATE",
    "oxygen_saturation": "SATS_SPO2",
    "inspired_oxygen":   "INSPIRED_O2_TEXT",
    "avpu_acvpu":        "AVPU_ORDINAL",
}
MF_FILE = {
    "heart_rate":        "heart_rate_membership_functions.csv",
    "blood_pressure":    "systolic_blood_pressure_membership_functions.csv",
    "temperature":       "temperature_membership_functions.csv",
    "respiratory_rate":  "respiratory_rate_membership_functions.csv",
    "oxygen_saturation": "oxygen_saturation_membership_functions.csv",
    "inspired_oxygen":   "inspired_oxygen_concentration_membership_functions.csv",
    "avpu_acvpu":        "avpu_acvpu_membership_functions.csv",
}
VITAL_TYPE = {
    "heart_rate": "7var", "blood_pressure": "7var",
    "temperature": "7var", "respiratory_rate": "7var",
    "oxygen_saturation": "3var_down", "inspired_oxygen": "3var_up",
    "avpu_acvpu": "3var_up",
}

AVPU_ORDINAL = {
    "Alert": 0.0,
    "Responds to voice": 1.0,
    "Newly confused / agitated": 2.0,
    "Responds to pain": 3.0,
    "Unresponsive": 3.0,
}
MAX_FUZZY_TOTAL = len(VITALS) * 3.0


def encode_avpu(series: pd.Series) -> np.ndarray:
    """Map AVPU/ACVPU text categories to ordinal 0-3 for fuzzy lookup."""
    cleaned = series.astype(str).str.strip()
    mapped = cleaned.map(AVPU_ORDINAL)
    unknown = mapped.isna() & cleaned.notna() & (cleaned != "nan")
    if unknown.any():
        print(f"  Warning: {unknown.sum():,} unknown AVPU values defaulting to Alert (0)")
    return mapped.fillna(0.0).astype(np.float32).values

LABELS_7 = [
    "Below normal - severe concern", "Below normal - moderate concern",
    "Below normal - mild concern", "No concern",
    "Above normal - mild concern", "Above normal - moderate concern",
    "Above normal - severe concern",
]
LABELS_3_DOWN = [
    "Below normal - severe concern", "Below normal - moderate concern",
    "Below normal - mild concern", "No concern",
]
LABELS_3_UP = [
    "No concern", "Above normal - mild concern",
    "Above normal - moderate concern", "Above normal - severe concern",
]

OUTPUT_MF_DEFS = {
    "No concern":       (-0.5, 0, 0, 0.75),
    "Mild concern":     (0.25, 1, 1, 1.75),
    "Moderate concern": (1.25, 2, 2, 2.75),
    "Severe concern":   (2.25, 3, 3, 3.5),
}

# =====================================================================
#  Fuzzy helpers (mirror streamlit_app.py logic)
# =====================================================================

def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if b <= x <= c:
        return 1.0
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def _interp_lookup(fs: dict, inp: float) -> float:
    if inp in fs:
        return float(fs[inp])
    keys = sorted(fs.keys())
    if not keys:
        return 0.0
    if inp <= keys[0]:
        return float(fs[keys[0]])
    if inp >= keys[-1]:
        return float(fs[keys[-1]])
    for i in range(len(keys) - 1):
        if keys[i] <= inp < keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            t = (inp - lo) / (hi - lo)
            return float(fs[lo]) * (1 - t) + float(fs[hi]) * t
    return 0.0


def _load_mf(vital: str) -> Tuple[List[str], List[dict]]:
    """Return (labels, list-of-membership-dicts) for one vital."""
    df = pd.read_csv(SIGMOID_DIR / MF_FILE[vital])
    keys = df["Value"].values
    vtype = VITAL_TYPE[vital]
    labels = {"7var": LABELS_7, "3var_down": LABELS_3_DOWN, "3var_up": LABELS_3_UP}[vtype]
    fs_list = [dict(zip(keys, df[lab].values)) for lab in labels]
    return labels, fs_list


def _concern_from_memberships(memberships: Dict[str, float]) -> Dict[str, float]:
    concern = {"No concern": 0.0, "Mild concern": 0.0,
               "Moderate concern": 0.0, "Severe concern": 0.0}
    for key, val in memberships.items():
        kl = key.lower()
        if "severe" in kl:
            concern["Severe concern"] = max(concern["Severe concern"], val)
        elif "moderate" in kl:
            concern["Moderate concern"] = max(concern["Moderate concern"], val)
        elif "mild" in kl:
            concern["Mild concern"] = max(concern["Mild concern"], val)
        elif "no concern" in kl:
            concern["No concern"] = max(concern["No concern"], val)
    return concern


# Pre-build the output-MF grid once (301 × 4).
_OUTPUT_X = np.arange(0, 3.01, 0.01)
_OUTPUT_GRID: Dict[str, np.ndarray] = {}
for _lbl, _params in OUTPUT_MF_DEFS.items():
    _OUTPUT_GRID[_lbl] = np.array([_trapezoid(x, *_params) for x in _OUTPUT_X])


def _defuzz_centroid(concern: Dict[str, float]) -> float:
    MIN_FIRING = 0.05
    filtered = {k: (v if v >= MIN_FIRING else 0.0) for k, v in concern.items()}
    if filtered.get("No concern", 0.0) > 0 and all(
        lev == "No concern" or f == 0.0 for lev, f in filtered.items()
    ):
        return 0.0
    agg = np.zeros(301)
    for level, firing in filtered.items():
        if firing > 0:
            np.maximum(agg, np.minimum(firing, _OUTPUT_GRID[level]), out=agg)
    denom = agg.sum()
    if denom == 0:
        return 0.0
    return float(np.dot(_OUTPUT_X, agg) / denom)


def _build_lookup(vital: str) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute input_value → defuzzified score for every grid point."""
    labels, fs_list = _load_mf(vital)
    df = pd.read_csv(SIGMOID_DIR / MF_FILE[vital])
    input_vals = df["Value"].values.astype(float)
    scores = np.empty(len(input_vals))
    for i, v in enumerate(input_vals):
        memberships = {lab: _interp_lookup(fs, v) for lab, fs in zip(labels, fs_list)}
        scores[i] = _defuzz_centroid(_concern_from_memberships(memberships))
    return input_vals, scores

# =====================================================================
#  Data loading
# =====================================================================

def load_data() -> pd.DataFrame:
    print("Loading dataset …")
    t0 = time.time()
    cols_needed = [
        "ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
        "REVIEW_WITHIN_4HOURS", "HEART_RATE", "SYSTOLIC_BP",
        "RESP_RATE", "SATS_SPO2", "INSPIRED_O2_TEXT", "AVPU_ACVPU",
        "TEMPERATURE", "COMPLETE_DATA", "NEWS-2",
    ]
    df = pd.read_csv(DATA_PATH, usecols=cols_needed, low_memory=False)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Coerce vital-sign columns that may contain text ("Refused", blanks, etc.)
    numeric_cols = ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
                    "TEMPERATURE", "COMPLETE_DATA", "REVIEW_WITHIN_4HOURS",
                    "DAYS_SINCE_ADMISSION"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to complete data
    df["COMPLETE_DATA"] = df["COMPLETE_DATA"].fillna(0)
    before = len(df)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    print(f"  Filtered to COMPLETE_DATA=1: {len(df):,} rows (dropped {before-len(df):,})")

    # Drop rows where any vital is still NaN after coercion
    vital_cols = ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2", "TEMPERATURE"]
    before2 = len(df)
    df.dropna(subset=vital_cols + ["REVIEW_WITHIN_4HOURS", "ANON_ADMISSION_ID"], inplace=True)
    if len(df) < before2:
        print(f"  Dropped {before2-len(df):,} rows with NaN vitals after coercion")

    # Cast cleaned columns to efficient types
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df["REVIEW_WITHIN_4HOURS"] = df["REVIEW_WITHIN_4HOURS"].astype("int8")
    for col in vital_cols:
        df[col] = df[col].astype("float32")

    # Inspired O2: coerce to numeric, clamp to [21, 100]
    df["INSPIRED_O2_TEXT"] = pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce").fillna(21.0)
    df["INSPIRED_O2_TEXT"] = df["INSPIRED_O2_TEXT"].clip(lower=21.0, upper=100.0).astype("float32")

    # NEWS-2: coerce blanks
    df["NEWS-2"] = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0).astype("float32")

    df["AVPU_ORDINAL"] = encode_avpu(df["AVPU_ACVPU"])

    # Construct t_minutes from DAYS_SINCE_ADMISSION + OBS_TIME
    obs_time = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    hours = obs_time.dt.hour.fillna(0).astype("float32")
    minutes = obs_time.dt.minute.fillna(0).astype("float32")
    seconds = obs_time.dt.second.fillna(0).astype("float32")
    df["t_minutes"] = (
        df["DAYS_SINCE_ADMISSION"].astype("float32") * 1440.0
        + hours * 60.0 + minutes + seconds / 60.0
    )

    # Sort by patient then time
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    label = df["REVIEW_WITHIN_4HOURS"].values
    print(f"  Positive labels: {label.sum():,} / {len(label):,} "
          f"({100*label.mean():.2f}%)")
    return df

# =====================================================================
#  Vectorised per-vital fuzzy scores
# =====================================================================

def compute_fuzzy_scores(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return {vital_name: score_array} for all observations."""
    print("Precomputing per-vital fuzzy lookup tables …")
    scores: Dict[str, np.ndarray] = {}
    for vital in VITALS:
        t0 = time.time()
        col_vals = df[VITAL_COL[vital]].values.astype(np.float64)
        if vital == "avpu_acvpu":
            scores[vital] = np.clip(col_vals, 0.0, 3.0).astype(np.float32)
            print(f"  {vital:25s}  direct ordinal 0-3  "
                  f"{len(col_vals)/1e6:.1f}M rows  {time.time()-t0:.1f}s")
            continue
        lut_x, lut_y = _build_lookup(vital)
        col_vals = np.clip(col_vals, lut_x[0], lut_x[-1])
        scores[vital] = np.interp(col_vals, lut_x, lut_y).astype(np.float32)
        print(f"  {vital:25s}  LUT {len(lut_x):>4d} pts  "
              f"interp {len(col_vals)/1e6:.1f}M rows  {time.time()-t0:.1f}s")
    return scores

# =====================================================================
#  One-time slope pre-computation (parameter-independent)
# =====================================================================

def _ols_slopes_for_group(times: np.ndarray, raw: np.ndarray,
                          window_min: float) -> np.ndarray:
    """OLS slope of raw scores within a time-based look-back window."""
    n = len(times)
    slopes = np.zeros(n, dtype=np.float32)
    if n < 2:
        return slopes
    left = 0
    for right in range(n):
        while left < right and (times[right] - times[left]) > window_min:
            left += 1
        count = right - left + 1
        if count < 2:
            continue
        t_slice = times[left:right + 1]
        s_slice = raw[left:right + 1]
        t0 = t_slice[0]
        t_h = (t_slice - t0) / 60.0
        mean_t = t_h.mean()
        mean_s = s_slice.mean()
        dt = t_h - mean_t
        ss_tt = (dt * dt).sum()
        if ss_tt == 0:
            continue
        slopes[right] = (dt * (s_slice - mean_s)).sum() / ss_tt
    return slopes


def precompute_slopes(df: pd.DataFrame,
                      pv_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute OLS trend slopes for every observation, for every vital."""
    window_min = WINDOW_HOURS * 60.0
    patient_ids = df["ANON_ADMISSION_ID"].values
    t_minutes = df["t_minutes"].values.astype(np.float64)

    # Build group boundaries (patients are sorted)
    change = np.empty(len(patient_ids), dtype=bool)
    change[0] = True
    change[1:] = patient_ids[1:] != patient_ids[:-1]
    group_starts = np.where(change)[0]
    group_ends = np.append(group_starts[1:], len(patient_ids))
    n_groups = len(group_starts)

    all_slopes: Dict[str, np.ndarray] = {}
    for vital in VITALS:
        print(f"  slopes: {vital:25s}", end="", flush=True)
        t0 = time.time()
        raw = pv_scores[vital].astype(np.float64)
        slopes = np.zeros(len(df), dtype=np.float32)
        for g in range(n_groups):
            s, e = group_starts[g], group_ends[g]
            slopes[s:e] = _ols_slopes_for_group(t_minutes[s:e], raw[s:e], window_min)
        all_slopes[vital] = slopes
        elapsed = time.time() - t0
        print(f"  {elapsed:.0f}s")
    return all_slopes, group_starts, group_ends

# =====================================================================
#  Grid search
# =====================================================================

def _ewma_alpha_eff(dt_minutes: float, alpha: float,
                    ref_minutes: float = EWMA_REF_MINUTES) -> float:
    """Time-adjusted EWMA weight for an irregular gap Δt since the previous obs."""
    if dt_minutes <= 0.0 or ref_minutes <= 0.0:
        return alpha
    return 1.0 - (1.0 - alpha) ** (dt_minutes / ref_minutes)


def _ewma_all(
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    raw: np.ndarray,
    times: np.ndarray,
    alpha: float,
    ref_minutes: float = EWMA_REF_MINUTES,
) -> np.ndarray:
    """EWMA for every observation across all patients for one vital.

    Uses time-adjusted smoothing so irregular observation gaps are handled
    correctly: α_eff = 1 - (1-α)^(Δt / ref_minutes).
    """
    ewma = np.empty_like(raw)
    for g in range(len(group_starts)):
        s, e = group_starts[g], group_ends[g]
        if s >= e:
            continue
        ewma[s] = raw[s]
        for i in range(s + 1, e):
            dt = max(float(times[i] - times[i - 1]), 0.0)
            alpha_eff = _ewma_alpha_eff(dt, alpha, ref_minutes)
            ewma[i] = alpha_eff * raw[i] + (1.0 - alpha_eff) * ewma[i - 1]
    return ewma


def _trend_factor_from_slopes(slopes: np.ndarray, beta: float) -> np.ndarray:
    """Sigmoid worsening-trend factor; zero when slope <= 0."""
    trend_factor = np.zeros_like(slopes)
    pos_mask = slopes > 0
    if pos_mask.any() and beta > 0:
        s = slopes[pos_mask]
        ex = np.exp(np.clip(-beta * s, -700, 700))
        trend_factor[pos_mask] = 2.0 / (1.0 + ex) - 1.0
    return trend_factor


def _aggregate_temporal(
    adjusted_vitals: Dict[str, np.ndarray],
    vitals: List[str],
    gamma: float,
    max_fuzzy_total: float = MAX_FUZZY_TOTAL,
) -> np.ndarray:
    if gamma == 1.0:
        return sum(adjusted_vitals[v] for v in vitals).astype(np.float32)
    additive = sum(adjusted_vitals[v] for v in vitals)
    stacked = np.column_stack([adjusted_vitals[v] for v in vitals])
    max_vital = stacked.max(axis=1)
    max_based = (max_fuzzy_total / 3.0) * max_vital
    return ((1.0 - gamma) * max_based + gamma * additive).astype(np.float32)


def compute_temporal(
    pv_scores: Dict[str, np.ndarray],
    all_slopes: Dict[str, np.ndarray],
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    times: np.ndarray,
    snapshot: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    mode: str = "full",
    vitals: List[str] | None = None,
    max_fuzzy_total: float | None = None,
) -> np.ndarray:
    """Row-level temporal fuzzy total.

    mode:
      - "full"       : EWMA memory + worsening-trend factor (default)
      - "ewma"       : EWMA memory only (no trend adjustment)
      - "trend"      : worsening-trend factor on raw scores only (no EWMA)
    """
    vitals = vitals or VITALS
    max_fuzzy_total = max_fuzzy_total if max_fuzzy_total is not None else MAX_FUZZY_TOTAL
    adjusted_vitals: Dict[str, np.ndarray] = {}
    for vital in vitals:
        raw = pv_scores[vital]
        if mode == "trend":
            base = raw
        else:
            ew = _ewma_all(group_starts, group_ends, raw, times, alpha)
            base = np.maximum(ew, raw)

        if mode == "ewma":
            adjusted_vitals[vital] = np.clip(base, 0.0, 3.0).astype(np.float32)
            continue

        slope = all_slopes[vital]
        trend_factor = _trend_factor_from_slopes(slope, beta)
        adj = base + trend_factor * (3.0 - base)
        adjusted_vitals[vital] = np.clip(adj, 0.0, 3.0).astype(np.float32)

    temporal = _aggregate_temporal(adjusted_vitals, vitals, gamma, max_fuzzy_total)
    return np.maximum(temporal, snapshot).astype(np.float32)


def _patient_level_auroc(
    label: np.ndarray,
    pred: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray,
) -> float:
    """AUROC after collapsing repeated observations to one score per patient."""
    n_groups = len(group_starts)
    y_patient = np.empty(n_groups, dtype=np.int8)
    p_patient = np.empty(n_groups, dtype=np.float64)

    for g in range(n_groups):
        s, e = group_starts[g], group_ends[g]
        y_patient[g] = label[s:e].max()
        p_patient[g] = pred[s:e].max()

    valid = np.isfinite(p_patient)
    y = y_patient[valid]
    p = p_patient[valid]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, p))


def run_grid_search(
    df: pd.DataFrame,
    pv_scores: Dict[str, np.ndarray],
    all_slopes: Dict[str, np.ndarray],
    group_starts: np.ndarray,
    group_ends: np.ndarray,
) -> pd.DataFrame:
    label = df["REVIEW_WITHIN_4HOURS"].values.astype(np.int8)
    t_minutes = df["t_minutes"].values.astype(np.float64)
    n_combos = len(ALPHA_GRID) * len(BETA_GRID) * len(GAMMA_GRID)
    print(f"\nGrid search: {len(ALPHA_GRID)} α × {len(BETA_GRID)} β × "
          f"{len(GAMMA_GRID)} γ = {n_combos} combinations\n")

    results = []
    combo = 0
    t_search_start = time.time()
    snapshot_total = sum(pv_scores[v] for v in VITALS)

    for alpha in ALPHA_GRID:
        t_alpha = time.time()
        # Compute EWMA + clamped for each vital (depends only on α)
        ewma_clamped: Dict[str, np.ndarray] = {}
        for vital in VITALS:
            raw = pv_scores[vital]
            ew = _ewma_all(group_starts, group_ends, raw, t_minutes, alpha)
            ewma_clamped[vital] = np.maximum(ew, raw)

        for beta in BETA_GRID:
            # Compute adjusted per-vital scores (depends on α via EWMA, β via sigmoid)
            adjusted: Dict[str, np.ndarray] = {}
            for vital in VITALS:
                slope = all_slopes[vital]
                clamped = ewma_clamped[vital]
                pos_mask = slope > 0
                trend_factor = np.zeros_like(slope)
                if pos_mask.any():
                    s = slope[pos_mask]
                    ex = np.exp(np.clip(-beta * s, -700, 700))
                    trend_factor[pos_mask] = 2.0 / (1.0 + ex) - 1.0
                adj = clamped + trend_factor * (3.0 - clamped)
                adjusted[vital] = np.clip(adj, 0.0, 3.0).astype(np.float32)

            for gamma in GAMMA_GRID:
                combo += 1
                # Aggregate to total
                if gamma == 1.0:
                    total = sum(adjusted[v] for v in VITALS)
                else:
                    additive = sum(adjusted[v] for v in VITALS)
                    stacked = np.column_stack([adjusted[v] for v in VITALS])
                    max_vital = stacked.max(axis=1)
                    max_based = (MAX_FUZZY_TOTAL / 3.0) * max_vital
                    total = (1.0 - gamma) * max_based + gamma * additive
                total = np.maximum(total, snapshot_total)

                valid = np.isfinite(total)
                obs_auroc = roc_auc_score(label[valid], total[valid])
                patient_auroc = _patient_level_auroc(
                    label=label,
                    pred=total,
                    group_starts=group_starts,
                    group_ends=group_ends,
                )
                results.append({
                    "alpha": alpha, "beta": beta, "gamma": gamma,
                    "auroc_obs": obs_auroc,
                    "auroc_patient": patient_auroc,
                })

                if combo % 50 == 0 or combo == n_combos:
                    elapsed = time.time() - t_search_start
                    print(f"  [{combo:>4d}/{n_combos}]  α={alpha:.1f} β={beta:.1f} "
                          f"γ={gamma:.2f}  AUROC(patient)={patient_auroc:.6f}  "
                          f"({elapsed:.0f}s elapsed)", flush=True)

        print(f"  α={alpha:.1f} done in {time.time()-t_alpha:.0f}s", flush=True)

    res_df = pd.DataFrame(results)
    print(f"\nGrid search finished in {time.time()-t_search_start:.0f}s")
    return res_df

# =====================================================================
#  Visualisations
# =====================================================================

def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_heatmaps(res_df: pd.DataFrame, best: dict):
    """2-D heatmaps: fix one parameter at optimal, show the other two."""
    _ensure_output_dir()
    slices = [
        ("alpha", "beta",  "gamma", best["gamma"]),
        ("alpha", "gamma", "beta",  best["beta"]),
        ("beta",  "gamma", "alpha", best["alpha"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, (xp, yp, fix_p, fix_v) in zip(axes, slices):
        sub = res_df[np.isclose(res_df[fix_p], fix_v)]
        piv = sub.pivot_table(index=yp, columns=xp, values="auroc_patient")
        im = ax.imshow(
            piv.values, aspect="auto", origin="lower",
            cmap="RdYlGn",
            extent=[piv.columns.min(), piv.columns.max(),
                    piv.index.min(), piv.index.max()],
        )
        ax.set_xlabel(xp, fontsize=12)
        ax.set_ylabel(yp, fontsize=12)
        ax.set_title(f"{xp} vs {yp}  ({fix_p}={fix_v:.2f} fixed)", fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.plot(best[xp], best[yp], "k*", markersize=14)

    fig.suptitle("Pairwise AUROC Heatmaps", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved heatmaps.png")


def plot_surfaces(res_df: pd.DataFrame, best: dict):
    """3-D surface plots for each pair of parameters."""
    _ensure_output_dir()
    slices = [
        ("alpha", "beta",  "gamma", best["gamma"]),
        ("alpha", "gamma", "beta",  best["beta"]),
        ("beta",  "gamma", "alpha", best["alpha"]),
    ]
    fig = plt.figure(figsize=(22, 6))
    for i, (xp, yp, fix_p, fix_v) in enumerate(slices, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        sub = res_df[np.isclose(res_df[fix_p], fix_v)]
        piv = sub.pivot_table(index=yp, columns=xp, values="auroc_patient")
        X, Y = np.meshgrid(piv.columns.values, piv.index.values)
        Z = piv.values
        ax.plot_surface(X, Y, Z, cmap="RdYlGn", edgecolor="grey",
                        linewidth=0.3, alpha=0.9)
        ax.set_xlabel(xp, fontsize=10, labelpad=8)
        ax.set_ylabel(yp, fontsize=10, labelpad=8)
        ax.set_zlabel("AUROC", fontsize=10, labelpad=8)
        ax.set_title(f"{xp} vs {yp}  ({fix_p}={fix_v:.2f})", fontsize=11)
        ax.scatter([best[xp]], [best[yp]],
                   [res_df.loc[res_df["auroc_patient"].idxmax(), "auroc_patient"]],
                   color="red", s=80, zorder=10, depthshade=False)
    fig.suptitle("Pairwise AUROC Surfaces", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "surfaces_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved surfaces_3d.png")


def plot_sensitivity(res_df: pd.DataFrame, best: dict):
    """1-D sensitivity: vary one parameter, fix other two at optimal."""
    _ensure_output_dir()
    params = [("alpha", ALPHA_GRID), ("beta", BETA_GRID), ("gamma", GAMMA_GRID)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (p, grid) in zip(axes, params):
        others = [pp for pp, _ in params if pp != p]
        sub = res_df.copy()
        for o in others:
            sub = sub[np.isclose(sub[o], best[o])]
        sub = sub.sort_values(p)
        ax.plot(sub[p], sub["auroc_patient"], "o-", color="steelblue", linewidth=2,
                markersize=6)
        ax.axvline(best[p], color="red", linestyle="--", alpha=0.7,
                   label=f"optimal={best[p]:.2f}")
        ax.set_xlabel(p, fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title(f"Sensitivity to {p}", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Parameter Sensitivity (others fixed at optimal)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sensitivity.png")


def plot_roc_comparison(
    df: pd.DataFrame,
    pv_scores: Dict[str, np.ndarray],
    all_slopes: Dict[str, np.ndarray],
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    best: dict,
):
    """ROC curves: NEWS-2  vs  snapshot fuzzy  vs  temporal-adjusted (best)."""
    _ensure_output_dir()
    label = df["REVIEW_WITHIN_4HOURS"].values.astype(np.int8)
    t_minutes = df["t_minutes"].values.astype(np.float64)

    # Snapshot fuzzy total (no temporal adjustment)
    snapshot_total = sum(pv_scores[v] for v in VITALS)

    # Temporal-adjusted total at optimal parameters
    alpha, beta, gamma = best["alpha"], best["beta"], best["gamma"]
    adjusted_pv: Dict[str, np.ndarray] = {}
    for vital in VITALS:
        raw = pv_scores[vital]
        ew = _ewma_all(group_starts, group_ends, raw, t_minutes, alpha)
        clamped = np.maximum(ew, raw)
        slope = all_slopes[vital]
        pos = slope > 0
        tf = np.zeros_like(slope)
        if pos.any():
            ex = np.exp(np.clip(-beta * slope[pos], -700, 700))
            tf[pos] = 2.0 / (1.0 + ex) - 1.0
        adj = clamped + tf * (3.0 - clamped)
        adjusted_pv[vital] = np.clip(adj, 0.0, 3.0)

    if gamma == 1.0:
        temporal_total = sum(adjusted_pv[v] for v in VITALS)
    else:
        additive = sum(adjusted_pv[v] for v in VITALS)
        stacked = np.column_stack([adjusted_pv[v] for v in VITALS])
        max_vital = stacked.max(axis=1)
        max_based = (18.0 / 3.0) * max_vital
        temporal_total = (1.0 - gamma) * max_based + gamma * additive
    temporal_total = np.maximum(temporal_total, snapshot_total)

    news2 = df["NEWS-2"].values

    curves = [
        ("NEWS-2", news2, "tab:orange"),
        ("Snapshot Fuzzy EWS", snapshot_total, "tab:blue"),
        (f"Temporal (α={alpha:.2f}, β={beta:.1f}, γ={gamma:.2f})",
         temporal_total, "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    for name, pred, col in curves:
        valid = np.isfinite(pred)
        fpr, tpr, _ = roc_curve(label[valid], pred[valid])
        auc = roc_auc_score(label[valid], pred[valid])
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{name}  (AUROC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve Comparison", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved roc_comparison.png")

    for name, pred, _ in curves:
        valid = np.isfinite(pred)
        print(f"  {name:45s}  AUROC = {roc_auc_score(label[valid], pred[valid]):.6f}")


def plot_top_n_heatmap(res_df: pd.DataFrame, n: int = 20):
    """Horizontal bar chart of top-N configurations."""
    _ensure_output_dir()
    top = res_df.nlargest(n, "auroc_patient").reset_index(drop=True)
    labels = [f"α={r.alpha:.1f} β={r.beta:.1f} γ={r.gamma:.2f}"
              for _, r in top.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.RdYlGn(np.linspace(0.4, 1.0, n))[::-1]
    bars = ax.barh(range(n), top["auroc_patient"], color=colors, edgecolor="grey", linewidth=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("AUROC", fontsize=12)
    ax.set_title(f"Top {n} Parameter Configurations", fontsize=13)
    ax.invert_yaxis()
    for bar, val in zip(bars, top["auroc_patient"]):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                f"{val:.6f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_n_configs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved top_n_configs.png")


def plot_parameter_distributions(res_df: pd.DataFrame):
    """Show AUROC distribution marginalised over each parameter."""
    _ensure_output_dir()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, param in zip(axes, ["alpha", "beta", "gamma"]):
        grouped = res_df.groupby(param)["auroc_patient"].agg(["mean", "std", "max", "min"])
        ax.fill_between(grouped.index, grouped["min"], grouped["max"],
                        alpha=0.2, color="steelblue", label="min–max range")
        ax.plot(grouped.index, grouped["mean"], "o-", color="steelblue",
                linewidth=2, label="mean AUROC")
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title(f"AUROC distribution across {param}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Marginal Parameter Impact", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "parameter_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved parameter_distributions.png")

# =====================================================================
#  Main
# =====================================================================

def main():
    t_total = time.time()
    print("=" * 70)
    print("  Temporal Context Builder – Grid Search & AUROC")
    print("=" * 70)

    # 1. Load data
    df = load_data()

    # 2. Compute per-vital fuzzy scores (one-time)
    pv_scores = compute_fuzzy_scores(df)

    # 3. Compute trend slopes (one-time, parameter-independent)
    print("\nPrecomputing OLS trend slopes (one-time) …")
    all_slopes, group_starts, group_ends = precompute_slopes(df, pv_scores)

    # 4. Grid search
    res_df = run_grid_search(df, pv_scores, all_slopes, group_starts, group_ends)

    # 5. Best configuration
    best_idx = res_df["auroc_patient"].idxmax()
    best = res_df.loc[best_idx]
    print(f"\n{'='*70}")
    print(f"  BEST:  α={best.alpha:.2f}  β={best.beta:.1f}  "
          f"γ={best.gamma:.2f}  →  AUROC(patient) = {best.auroc_patient:.6f}")
    print(f"{'='*70}")

    best_dict = {"alpha": best.alpha, "beta": best.beta, "gamma": best.gamma}

    # 6. Save raw results
    _ensure_output_dir()
    res_df.to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)
    print(f"\nSaved grid_search_results.csv ({len(res_df)} rows)")

    # 7. Visualisations
    print("\nGenerating plots …")
    plot_heatmaps(res_df, best_dict)
    plot_surfaces(res_df, best_dict)
    plot_sensitivity(res_df, best_dict)
    plot_top_n_heatmap(res_df)
    plot_parameter_distributions(res_df)
    plot_roc_comparison(df, pv_scores, all_slopes, group_starts, group_ends, best_dict)

    # 8. NEWS-2 baseline AUROC
    label = df["REVIEW_WITHIN_4HOURS"].values
    news2 = df["NEWS-2"].values
    valid_news = np.isfinite(news2)
    news_auroc = roc_auc_score(label[valid_news], news2[valid_news])
    snapshot_total = sum(pv_scores[v] for v in VITALS)
    snap_auroc = roc_auc_score(label, snapshot_total)
    print(f"\n{'='*70}")
    print(f"  NEWS-2 AUROC (baseline)          = {news_auroc:.6f}")
    print(f"  Snapshot Fuzzy EWS AUROC         = {snap_auroc:.6f}")
    print(f"  Temporal Builder AUROC (patient, optimal) = {best.auroc_patient:.6f}")
    print(f"  Δ (Temporal patient − NEWS-2 obs)        = {best.auroc_patient - news_auroc:+.6f}")
    print(f"{'='*70}")
    print(f"\nTotal wall time: {time.time()-t_total:.0f}s")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()