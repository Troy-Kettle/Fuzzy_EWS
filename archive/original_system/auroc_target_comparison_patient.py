"""
PATIENT-LEVEL AUROC validation across three systems and three clinical targets,
run over a full matrix of configurations in a single data load.

Each ADMISSION contributes ONE sample, scored by the MAXIMUM (peak) score the system
reached across that admission's stay, and labelled by whether the admission EVER had
the event (clean per-target pools). This answers "can peak concern separate patients
who deteriorate from those who don't" — the natural patient-level early-warning question.
Mirrors baseline_results/auroc_target_comparison.py exactly except for this aggregation.

Systems compared (per config):
  1. NEWS-2          (pre-scored column; per-patient = peak NEWS-2)
  2. Snapshot Fuzzy  (defuzzified sum, no temporal; per-patient = peak)
  3. Temporal Fuzzy  (time-decay excess-EWMA;       per-patient = peak)

Targets (clean per-target negative pools, no cross-contamination):
  DEATH AUROC: positives = patient EVER DEATH=1   negatives = never DEATH and never ICU
  ICU   AUROC: positives = patient EVER ICU=1     negatives = never ICU   and never DEATH
  EVENT AUROC: positives = patient EVER EVENT=1   negatives = never EVENT

Configuration matrix (8 leaf folders, all under patient_level_results/):
  param sets : event-optimal (patient-level grid) + fixed (0.5, 5.0, 0.75)
  vitals     : 6-vital (main folder)  +  7-vital incl. ACVPU (acvpu/ subfolder)
  temporal   : raise-only      → patient_level_results/results/
               bidirectional   → patient_level_results/results_lowering_included/

Temporal mechanism (time-decay excess-EWMA):
  ewma[i] = a_eff·raw[i] + (1−a_eff)·ewma[i−1],  a_eff = 1 − (1−α)^(Δt/EWMA_REF_MIN)
  raise-only:     signal = max(0, raw − ewma)   adj = clip(raw + β·signal, 0, 3)
                  final  = max(total, snapshot)               (temporal never below snapshot)
  bidirectional:  signal = raw − ewma  (signed)  adj = clip(raw + β·signal, 0, 3)
                  final  = total                              (may fall below snapshot)

Output per leaf folder: auroc_table.csv, auroc_bar_chart.png, roc_{target}.png
"""

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
REPO         = Path(__file__).resolve().parent
DATA_PATH    = REPO / "datasets" / "final_observations_with_targets.csv"
SIGMOID_DIR  = REPO / "membership_functions" / "sigmoid"
PL_BASE      = REPO / "patient_level_results"
GRID_RESULTS = PL_BASE / "results" / "grid_search_excess" / "grid_results.csv"

# ── Constants ────────────────────────────────────────────────────────────────
# Time-decay reference (see grid_search_excess.py): a_eff = 1 − (1−α)^(Δt/EWMA_REF_MIN)
# is proper exponential time-decay for irregular sampling; 360 min ≈ median ward gap.
EWMA_REF_MIN = 360.0
TARGET_ROWS  = 750_000     # max rows per per-target evaluation set
RANDOM_SEED  = 42

# Fixed param set the user wants reported alongside the event-optimal one.
FIXED_PARAMS = (0.5, 5.0, 0.75)

TARGETS = {
    "Death within 24h": "DEATH_WITHIN_24H",
    "ICU within 24h":   "ICU_WITHIN_24H",
    "Event within 24h": "EVENT_FLAG",
}

# ── Vitals ───────────────────────────────────────────────────────────────────
# Base 6 vitals (AVPU/ACVPU excluded). ACVPU is added only for the acvpu/ subfolder.
VITALS_BASE = ["heart_rate", "blood_pressure", "temperature",
               "respiratory_rate", "oxygen_saturation", "inspired_oxygen"]
ACVPU       = "acvpu"

VITAL_COL = {
    "heart_rate":        "HEART_RATE",
    "blood_pressure":    "SYSTOLIC_BP",
    "temperature":       "TEMPERATURE",
    "respiratory_rate":  "RESP_RATE",
    "oxygen_saturation": "SATS_SPO2",
    "inspired_oxygen":   "INSPIRED_O2_TEXT",
    "acvpu":             "ACVPU_NUM",        # built from AVPU_ACVPU text (see ACVPU_MAP)
}

MF_FILE = {
    "heart_rate":        "heart_rate_membership_functions.csv",
    "blood_pressure":    "systolic_blood_pressure_membership_functions.csv",
    "temperature":       "temperature_membership_functions.csv",
    "respiratory_rate":  "respiratory_rate_membership_functions.csv",
    "oxygen_saturation": "oxygen_saturation_membership_functions.csv",
    "inspired_oxygen":   "inspired_oxygen_concentration_membership_functions.csv",
    "acvpu":             "avpu_acvpu_membership_functions.csv",
}

VITAL_TYPE = {
    "heart_rate": "7var", "blood_pressure": "7var",
    "temperature": "7var", "respiratory_rate": "7var",
    "oxygen_saturation": "3var_down", "inspired_oxygen": "3var_up",
    "acvpu": "3var_up",
}

# Graded clinical severity mapping for ACVPU text → fuzzy input value 0..3
# (Alert < Confusion < Voice < Pain ≈ Unresponsive). Mirrors NEWS-2 (Alert=0, else
# abnormal) but graded so the fuzzy system can express intermediate concern.
ACVPU_MAP = {
    "Alert":                     0.0,
    "Newly confused / agitated": 1.0,
    "Responds to voice":         2.0,
    "Responds to pain":          3.0,
    "Unresponsive":              3.0,
}

LABELS_7      = ["Below normal - severe concern", "Below normal - moderate concern",
                 "Below normal - mild concern", "No concern",
                 "Above normal - mild concern", "Above normal - moderate concern",
                 "Above normal - severe concern"]
LABELS_3_DOWN = ["Below normal - severe concern", "Below normal - moderate concern",
                 "Below normal - mild concern", "No concern"]
LABELS_3_UP   = ["No concern", "Above normal - mild concern",
                 "Above normal - moderate concern", "Above normal - severe concern"]

OUTPUT_MF = {
    "No concern":       (-0.5, 0, 0, 0.75),
    "Mild concern":     (0.25, 1, 1, 1.75),
    "Moderate concern": (1.25, 2, 2, 2.75),
    "Severe concern":   (2.25, 3, 3, 3.5),
}
_OUTPUT_X = np.arange(0, 3.01, 0.01)
_OUTPUT_GRID = {
    lbl: np.array([
        (1.0 if b <= x <= c else
         (0.0 if x <= a or x >= d else
          (x-a)/(b-a) if a < x < b else (d-x)/(d-c)))
        for x in _OUTPUT_X
    ])
    for lbl, (a, b, c, d) in OUTPUT_MF.items()
}


# ── Fuzzy LUT machinery ───────────────────────────────────────────────────────

def _defuzz_centroid(memberships: dict) -> float:
    concern = {"No concern": 0.0, "Mild concern": 0.0,
               "Moderate concern": 0.0, "Severe concern": 0.0}
    for key, val in memberships.items():
        kl = key.lower()
        if   "severe"   in kl: concern["Severe concern"]   = max(concern["Severe concern"],   val)
        elif "moderate" in kl: concern["Moderate concern"] = max(concern["Moderate concern"], val)
        elif "mild"     in kl: concern["Mild concern"]     = max(concern["Mild concern"],     val)
        else:                  concern["No concern"]        = max(concern["No concern"],       val)
    agg = np.zeros(301)
    for level, firing in concern.items():
        if firing >= 0.05:
            np.maximum(agg, np.minimum(firing, _OUTPUT_GRID[level]), out=agg)
    denom = agg.sum()
    return 0.0 if denom == 0 else float(np.dot(_OUTPUT_X, agg) / denom)


def _build_lut(vital: str):
    df  = pd.read_csv(SIGMOID_DIR / MF_FILE[vital])
    x   = df["Value"].values.astype(float)
    labels = {"7var": LABELS_7, "3var_down": LABELS_3_DOWN,
              "3var_up": LABELS_3_UP}[VITAL_TYPE[vital]]
    scores = np.array([
        _defuzz_centroid({lab: float(np.interp(v, x, df[lab].values))
                          for lab in labels})
        for v in x
    ])
    return x, scores


def apply_luts(df: pd.DataFrame, luts: dict, vitals) -> dict:
    pv = {}
    for vital in vitals:
        col  = df[VITAL_COL[vital]].values.astype(np.float64)
        x, y = luts[vital]
        pv[vital] = np.interp(np.clip(col, x[0], x[-1]), x, y).astype(np.float32)
    return pv


# ── Time-decay EWMA ───────────────────────────────────────────────────────────

def _ewma_group(times: np.ndarray, raw: np.ndarray, alpha: float) -> np.ndarray:
    ewma    = np.empty_like(raw, dtype=np.float64)
    ewma[0] = raw[0]
    for i in range(1, len(raw)):
        dt      = max(float(times[i] - times[i-1]), 0.0)
        a_eff   = 1.0 - (1.0 - alpha) ** (dt / EWMA_REF_MIN)
        ewma[i] = a_eff * raw[i] + (1.0 - a_eff) * ewma[i-1]
    return ewma


def compute_ewma_all(df, pv, gs, ge, alpha, vitals) -> dict:
    """EWMA per vital at a given α (depends only on α + vital, reused across modes)."""
    t = df["t_minutes"].values.astype(np.float64)
    n = len(df)
    out = {}
    for vital in vitals:
        raw = pv[vital].astype(np.float64)
        ew  = np.empty(n, np.float64)
        for g in range(len(gs)):
            s, e     = gs[g], ge[g]
            ew[s:e]  = _ewma_group(t[s:e], raw[s:e], alpha)
        out[vital] = ew
    return out


def temporal_score(pv, ewma, vitals, beta, gamma, mode) -> np.ndarray:
    """Combine raw fuzzy scores + EWMA baseline into a temporal total.

    mode='raise_only'    : signal = max(0, raw − ewma), final = max(total, snapshot)
    mode='bidirectional' : signal = raw − ewma (signed), final = total (no floor)
    """
    adjusted = {}
    for v in vitals:
        raw = pv[v].astype(np.float64)
        if mode == "raise_only":
            signal = np.maximum(0.0, raw - ewma[v])
        else:  # bidirectional
            signal = raw - ewma[v]
        adjusted[v] = np.clip(raw + beta * signal, 0.0, 3.0).astype(np.float32)

    n_vitals = len(vitals)
    additive = sum(adjusted[v] for v in vitals)
    if gamma == 1.0:
        total = additive
    else:
        max_vital = np.column_stack([adjusted[v] for v in vitals]).max(axis=1)
        total     = (1.0 - gamma) * (n_vitals * max_vital) + gamma * additive

    snapshot = sum(pv[v] for v in vitals)
    if mode == "raise_only":
        return np.maximum(total, snapshot).astype(np.float32)
    return total.astype(np.float32)        # bidirectional: allow lowering below snapshot


def group_boundaries(ids: np.ndarray):
    change     = np.empty(len(ids), dtype=bool)
    change[0]  = True
    change[1:] = ids[1:] != ids[:-1]
    starts = np.where(change)[0]
    ends   = np.append(starts[1:], len(ids))
    return starts, ends


# ── Per-target AUROC (clean pools) ────────────────────────────────────────────

def _pools(d, i, e, target_col):
    if target_col == "DEATH_WITHIN_24H":
        return d == 1, (d == 0) & (i == 0)
    if target_col == "ICU_WITHIN_24H":
        return i == 1, (i == 0) & (d == 0)
    return e == 1, e == 0


def target_auroc(d, i, e, score, target_col) -> float:
    """Patient-level AUROC over the full clean pool (no row downsampling needed —
    there are only ~one sample per admission). d/i/e/score are per-patient arrays."""
    pos_mask, neg_mask = _pools(d, i, e, target_col)
    keep = pos_mask | neg_mask
    y    = pos_mask[keep].astype(np.int8)
    s    = score[keep]
    m    = np.isfinite(s)
    if y[m].sum() == 0 or y[m].sum() == m.sum():
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


# ── Plotting ──────────────────────────────────────────────────────────────────

SYS_COLOR = {"NEWS-2": "#E74C3C", "Snapshot Fuzzy": "#3498DB", "Temporal Fuzzy": "#2ECC71"}
SYS_LS    = {"NEWS-2": "--",      "Snapshot Fuzzy": "-.",       "Temporal Fuzzy": "-"}


def plot_roc(d, i, e, scores, target_col, target_name, out_dir, label):
    pos_mask, neg_mask = _pools(d, i, e, target_col)
    keep = pos_mask | neg_mask
    y    = pos_mask[keep].astype(np.int8)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5)
    for sys_name, s_all in scores.items():
        s = s_all[keep]
        m = np.isfinite(s)
        fpr, tpr, _ = roc_curve(y[m], s[m])
        auc = roc_auc_score(y[m], s[m])
        ax.plot(fpr, tpr, color=SYS_COLOR[sys_name], ls=SYS_LS[sys_name],
                lw=2.0, label=f"{sys_name}  (AUC={auc:.4f})")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"Patient-level ROC — {target_name}\n{label}\n"
                 f"(patients: pos={int(pos_mask.sum()):,}, clean neg={int(neg_mask.sum()):,})",
                 fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{target_col.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(results_df, out_dir, label):
    targets = list(TARGETS.keys())
    systems = ["NEWS-2", "Snapshot Fuzzy", "Temporal Fuzzy"]
    x, w = np.arange(len(targets)), 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, sys in enumerate(systems):
        vals = [results_df.loc[results_df["System"] == sys, t].values[0] for t in targets]
        bars = ax.bar(x + (k-1)*w, vals, w, label=sys,
                      color=SYS_COLOR[sys], edgecolor="white", lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=11)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(f"Patient-level AUROC by System and Target (peak score per admission)\n{label}",
                 fontsize=11)
    ax.legend(fontsize=10)
    lo = results_df[targets].min().min()
    hi = results_df[targets].max().max()
    ax.set_ylim(max(0.5, lo - 0.03), min(1.0, hi + 0.07))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "auroc_bar_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── One leaf-config run ───────────────────────────────────────────────────────

def run_leaf(gs, d_pat, i_pat, e_pat, news2_pat, pv, ewma, vitals,
             beta, gamma, mode, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Row-level system scores, then peak per admission (np.maximum.reduceat over group starts).
    snapshot_row = sum(pv[v] for v in vitals).astype(np.float32)
    temporal_row = temporal_score(pv, ewma, vitals, beta, gamma, mode)
    scores = {
        "NEWS-2":         news2_pat,                                  # already per-patient peak
        "Snapshot Fuzzy": np.maximum.reduceat(snapshot_row, gs),
        "Temporal Fuzzy": np.maximum.reduceat(temporal_row, gs),
    }

    rows = []
    for sys_name, s in scores.items():
        row = {"System": sys_name}
        for tname, tcol in TARGETS.items():
            row[tname] = round(target_auroc(d_pat, i_pat, e_pat, s, tcol), 6)
        rows.append(row)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "auroc_table.csv", index=False)

    for tname, tcol in TARGETS.items():
        plot_roc(d_pat, i_pat, e_pat, scores, tcol, tname, out_dir, label)
    plot_bar(results_df, out_dir, label)

    rel = out_dir.relative_to(REPO)
    print(f"\n  ── {label}  →  {rel}")
    print(results_df.to_string(index=False))
    return results_df


# ══════════════════════════════════════════════════════════════════════════════

def pick_event_optimal():
    grid = pd.read_csv(GRID_RESULTS)
    best = grid.loc[grid["event"].idxmax()]
    return float(best["alpha"]), float(best["beta"]), float(best["gamma"])


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Param sets ───────────────────────────────────────────────────────────
    # (display name, folder name, (α, β, γ))
    opt = pick_event_optimal()
    param_sets = [
        ("event-optimal", "auroc_grid_search_event_params", opt),
        ("Sherif's",      "auroc_sherifs_params",           FIXED_PARAMS),
    ]
    print("Param sets:")
    for name, folder, (a, b, g) in param_sets:
        print(f"  {name:14s}  α={a}  β={b}  γ={g}  →  {folder}")

    # ── Load + clean ─────────────────────────────────────────────────────────
    print("\nLoading dataset…")
    t0 = time.time()
    cols = ["ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
            "HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
            "INSPIRED_O2_TEXT", "AVPU_ACVPU", "TEMPERATURE",
            "COMPLETE_DATA", "NEWS-2",
            "DEATH_WITHIN_24H", "ICU_WITHIN_24H", "EVENT_FLAG"]
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    print(f"  After COMPLETE_DATA filter: {len(df):,} rows")

    num_cols = ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
                "TEMPERATURE", "DAYS_SINCE_ADMISSION"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=num_cols, inplace=True)

    df["INSPIRED_O2_TEXT"] = (pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce")
                              .fillna(21.0).clip(21.0, 100.0))
    df["NEWS-2"] = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0)
    df["ACVPU_NUM"] = df["AVPU_ACVPU"].map(ACVPU_MAP).fillna(0.0)   # text → 0..3 (Alert=0)

    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"] * 1440.0
                       + obs.dt.hour.fillna(0) * 60.0
                       + obs.dt.minute.fillna(0)
                       + obs.dt.second.fillna(0) / 60.0).astype(np.float32)

    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Fuzzy LUTs + per-vital raw scores (7 vitals, α-independent) ──────────
    print("\nBuilding fuzzy LUTs (7 vitals incl. ACVPU)…")
    vitals_full = VITALS_BASE + [ACVPU]
    luts = {v: _build_lut(v) for v in vitals_full}
    pv   = apply_luts(df, luts, vitals_full)

    gs, ge = group_boundaries(df["ANON_ADMISSION_ID"].values)

    # ── Patient-level labels (one sample per admission) ──────────────────────
    d_pat = np.maximum.reduceat(df["DEATH_WITHIN_24H"].values, gs)   # ever death-within-24h
    i_pat = np.maximum.reduceat(df["ICU_WITHIN_24H"].values,  gs)    # ever icu-within-24h
    e_pat = np.maximum.reduceat(df["EVENT_FLAG"].values,      gs)    # ever event
    news2_pat = np.maximum.reduceat(df["NEWS-2"].values.astype(np.float64), gs)  # peak NEWS-2
    print(f"  Patients: {len(gs):,}  (event pos={int(e_pat.sum()):,}, "
          f"neg={int((e_pat==0).sum()):,})")

    # ── Config matrix ────────────────────────────────────────────────────────
    MODE_DIR = {"raise_only": "results", "bidirectional": "results_lowering_included"}
    vital_variants = {"":      VITALS_BASE,            # main folder: 6 vitals
                      "acvpu": vitals_full}            # subfolder:   7 vitals

    for pname, folder, (alpha, beta, gamma) in param_sets:
        print(f"\n{'='*70}\nParam set: {pname}  (α={alpha}, β={beta}, γ={gamma})\n{'='*70}")
        print("  Computing EWMA (7 vitals at this α)…")
        t1 = time.time()
        ewma = compute_ewma_all(df, pv, gs, ge, alpha, vitals_full)
        print(f"  EWMA done in {time.time()-t1:.0f}s")

        for mode, root in MODE_DIR.items():
            for sub, vitals in vital_variants.items():
                out_dir = PL_BASE / root / folder / sub if sub else PL_BASE / root / folder
                vlabel  = "7 vitals (incl. ACVPU)" if sub else "6 vitals (no ACVPU)"
                mlabel  = "raise-only" if mode == "raise_only" else "bidirectional (raise+lower)"
                label   = f"{pname}: α={alpha} β={beta} γ={gamma} | {vlabel} | {mlabel}"
                run_leaf(gs, d_pat, i_pat, e_pat, news2_pat, pv, ewma, vitals,
                         beta, gamma, mode, out_dir, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
