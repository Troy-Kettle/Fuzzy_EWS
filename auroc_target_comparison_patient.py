"""
PATIENT-LEVEL + ROW-LEVEL validation — improved system.

Metrics per configuration leaf:
  AUROC        — row-level and patient-level
  AUPRC        — row-level and patient-level
  Sens / Spec  — at Youden's-J optimal threshold, both levels

Lead-time table (primary config: event-optimal, +ACVPU):
  At matched detection rates 60/70/80/90%:
    median lead time (h) and false-alarm rate (% non-event patients alerting)

Temporal formula: EWMA + sigmoid worsening-trend (engine_scoring.temporal_score),
matching app/streamlit_app.py — the adjusted score is structurally >= the snapshot
total, so there is no separate raise_only/bidirectional mode to sweep.

Config matrix:
  param sets : event-optimal (from improved grid search)  +  Sherif's (0.5, 5.0, 0.75)
  vitals     : 6-vital  +  7-vital (+ACVPU)

Outputs per leaf folder:
  auroc_table.csv        — AUROC (legacy-compatible format)
  metrics_full.csv       — all metrics, both levels
  roc_{target}.png       — ROC curves
  pr_{target}.png        — precision-recall curves

Top-level outputs:
  results/current/lead_time_table.csv
  results/current/lead_time_plot.png
  results/current/all_metrics_summary.csv
"""

import sys, time, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, roc_curve,
                              average_precision_score, precision_recall_curve)

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "engine"))
import engine_scoring as es

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

DATA_PATH    = REPO / "datasets" / "final_observations_with_targets.csv"
PL_BASE      = REPO / "results" / "24thJuly"
GRID_RESULTS = PL_BASE / "grid_search" / "grid_results.csv"

RANDOM_SEED  = 42
FIXED_PARAMS = (0.5, 5.0, 0.75)
ROW_NEG_CAP  = 500_000   # cap on non-event rows for row-level metrics

TARGETS = {
    "Death within 24h": "DEATH_WITHIN_24H",
    "ICU within 24h":   "ICU_WITHIN_24H",
    "Event within 24h": "EVENT_FLAG",
}
TARGET_SHORT = {"DEATH_WITHIN_24H": "death", "ICU_WITHIN_24H": "icu", "EVENT_FLAG": "event"}

O2CAT_CONCERN = {"Low": 1.0, "Low-moderate": 1.5, "Moderate": 2.0,
                 "High": 2.5, "Very high": 3.0}

# Temporal excess-EWMA applies ONLY to continuously-varying physiological vitals.
# inspired_oxygen (categorical INSP_O2_CAT) and ACVPU (categorical) are clinical
# step-signals whose transitions produce artefactual EWMA excesses; they stay at
# snapshot value and are excluded from the temporal boost.
TEMPORAL_VITALS = es.TEMPORAL_VITALS_DEFAULT   # canonical set (incl. inspired_oxygen);
# defined once in engine_scoring so the app and the pipeline cannot drift apart again.

SYS_COLOR = {"NEWS-2": "#E74C3C", "Snapshot Fuzzy": "#3498DB", "Temporal Fuzzy": "#2ECC71"}
SYS_LS    = {"NEWS-2": "--",      "Snapshot Fuzzy": "-.",       "Temporal Fuzzy": "-"}
SYSTEMS   = ["NEWS-2", "Snapshot Fuzzy", "Temporal Fuzzy"]


# ── Scoring ────────────────────────────────────────────────────────────────────

def build_pv(df, luts, vitals):
    pv = es.apply_luts(df, luts, vitals)
    pv["inspired_oxygen"] = df["O2_CONCERN"].values.astype(np.float32)
    return pv


def temporal_score_row(pv, ewma, slopes, vitals, beta, gamma):
    """Thin wrapper over engine_scoring.temporal_score (single source of truth,
    matching app/streamlit_app.py's EWMA + sigmoid-trend formula).

    The categorical-exclusion rule (inspired_oxygen / acvpu stay at snapshot) lives
    in the engine via ``temporal_vitals``."""
    return es.temporal_score(pv, ewma, slopes, vitals, beta, gamma,
                             method="additive", temporal_vitals=TEMPORAL_VITALS)


# ── Metrics ────────────────────────────────────────────────────────────────────

def clean_pool(d, i, e, target_col):
    if target_col == "DEATH_WITHIN_24H": return d == 1, (d == 0) & (i == 0)
    if target_col == "ICU_WITHIN_24H":   return i == 1, (i == 0) & (d == 0)
    return e == 1, e == 0


def compute_metrics(d, i, e, score, target_col):
    """AUROC, AUPRC, sens, spec at Youden's J threshold. Returns None if degenerate."""
    pos_mask, neg_mask = clean_pool(d, i, e, target_col)
    keep = pos_mask | neg_mask
    y = pos_mask[keep].astype(np.int8)
    s = score[keep].astype(np.float64)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or y.sum() == len(y):
        return None

    fpr, tpr, thresholds = roc_curve(y, s)
    auroc_val = float(roc_auc_score(y, s))
    j_idx = int(np.argmax(tpr - fpr))
    sens = float(tpr[j_idx])
    spec = float(1.0 - fpr[j_idx])
    thr  = float(thresholds[j_idx])
    auprc_val = float(average_precision_score(y, s))
    return {
        "auroc": round(auroc_val, 5), "auprc": round(auprc_val, 5),
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "threshold": round(thr, 4),
        "_fpr": fpr, "_tpr": tpr, "_y": y, "_s": s,   # retained for plotting only
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_roc(metrics_by_sys, target_col, target_name, out_dir, label):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5)
    for sys_name in SYSTEMS:
        m = metrics_by_sys.get(sys_name)
        if not m: continue
        ax.plot(m["_fpr"], m["_tpr"], color=SYS_COLOR[sys_name], ls=SYS_LS[sys_name],
                lw=2.0, label=f"{sys_name}  (AUC={m['auroc']:.4f})")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"Patient-level ROC — {target_name}\n{label}", fontsize=11)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{target_col.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr(metrics_by_sys, target_col, target_name, out_dir, label):
    fig, ax = plt.subplots(figsize=(7, 6))
    for sys_name in SYSTEMS:
        m = metrics_by_sys.get(sys_name)
        if not m: continue
        prec, rec, _ = precision_recall_curve(m["_y"], m["_s"])
        ax.plot(rec, prec, color=SYS_COLOR[sys_name], ls=SYS_LS[sys_name],
                lw=2.0, label=f"{sys_name}  (AP={m['auprc']:.4f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)",      fontsize=12)
    ax.set_title(f"Patient-level Precision-Recall — {target_name}\n{label}", fontsize=11)
    ax.legend(fontsize=10, loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_{target_col.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(df_metrics, col, out_dir, label, ylabel):
    targets = list(TARGETS.keys())
    x, w = np.arange(len(targets)), 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, sys in enumerate(SYSTEMS):
        sub  = df_metrics[df_metrics["system"] == sys]
        vals = [sub.loc[sub["target_name"] == t, col].values[0]
                if (sub["target_name"] == t).any() else np.nan
                for t in targets]
        bars = ax.bar(x + (k-1)*w, vals, w, label=sys,
                      color=SYS_COLOR[sys], edgecolor="white", lw=0.5)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Patient-level {ylabel} — improved system\n{label}", fontsize=11)
    ax.legend(fontsize=10)
    valid = df_metrics[col].dropna()
    if len(valid):
        ax.set_ylim(max(0.0, valid.min() - 0.03), min(1.0, valid.max() + 0.07))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / f"bar_{col}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Leaf runner ───────────────────────────────────────────────────────────────

def run_leaf(gs, d_pat, i_pat, e_pat, news2_pat,
             pv, ewma, slopes, vitals, beta, gamma,
             out_dir, label,
             d_row, i_row, e_row, news2_row, row_idx):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Row-level scores (full dataset)
    snap_row_full = sum(pv[v] for v in vitals).astype(np.float32)
    temp_row_full = temporal_score_row(pv, ewma, slopes, vitals, beta, gamma)

    # Patient-level scores (peak per admission)
    snap_pat = es.patient_peak(snap_row_full, gs)
    temp_pat = es.patient_peak(temp_row_full, gs)

    # Row-level downsampled subset
    snap_row_s  = snap_row_full[row_idx]
    temp_row_s  = temp_row_full[row_idx]
    news2_row_s = news2_row[row_idx]
    d_s = d_row[row_idx]; i_s = i_row[row_idx]; e_s = e_row[row_idx]

    all_rows = []

    for tname, tcol in TARGETS.items():
        tshort = TARGET_SHORT[tcol]

        pat_scores = {"NEWS-2": news2_pat,
                      "Snapshot Fuzzy": snap_pat, "Temporal Fuzzy": temp_pat}
        row_scores = {"NEWS-2": news2_row_s,
                      "Snapshot Fuzzy": snap_row_s, "Temporal Fuzzy": temp_row_s}

        # Patient-level metrics (also used for ROC / PR curves)
        pat_metrics = {}
        for sys_name, sc in pat_scores.items():
            m = compute_metrics(d_pat, i_pat, e_pat, sc, tcol)
            pat_metrics[sys_name] = m
            if m:
                all_rows.append({"level": "patient", "system": sys_name,
                                  "target": tshort, "target_name": tname,
                                  **{k: m[k] for k in
                                     ["auroc","auprc","sensitivity","specificity","threshold"]}})

        # Row-level metrics
        for sys_name, sc in row_scores.items():
            m = compute_metrics(d_s, i_s, e_s, sc, tcol)
            if m:
                all_rows.append({"level": "row", "system": sys_name,
                                  "target": tshort, "target_name": tname,
                                  **{k: m[k] for k in
                                     ["auroc","auprc","sensitivity","specificity","threshold"]}})

        plot_roc(pat_metrics, tcol, tname, out_dir, label)
        plot_pr(pat_metrics, tcol, tname, out_dir, label)

    mdf = pd.DataFrame(all_rows)
    mdf.to_csv(out_dir / "metrics_full.csv", index=False)

    # Legacy-compatible auroc_table.csv
    auroc_rows = []
    for sys_name in SYSTEMS:
        row = {"System": sys_name}
        for tname in TARGETS:
            hit = mdf[(mdf["system"]==sys_name) & (mdf["level"]=="patient") &
                      (mdf["target_name"]==tname)]
            row[tname] = hit["auroc"].values[0] if len(hit) else float("nan")
        auroc_rows.append(row)
    pd.DataFrame(auroc_rows).to_csv(out_dir / "auroc_table.csv", index=False)

    # Bar charts (patient-level)
    pat_df = mdf[mdf["level"] == "patient"]
    plot_bar(pat_df, "auroc",  out_dir, label, "AUROC")
    plot_bar(pat_df, "auprc",  out_dir, label, "AUPRC")

    # Console summary
    rel = out_dir.relative_to(REPO)
    print(f"\n  ── {label}  →  {rel}")
    print(f"  {'System':<18} {'Level':<10} {'Target':<8}"
          f" {'AUROC':>8} {'AUPRC':>8} {'Sens':>7} {'Spec':>7}")
    print(f"  {'-'*68}")
    for r in all_rows:
        print(f"  {r['system']:<18} {r['level']:<10} {r['target']:<8}"
              f" {r['auroc']:>8.5f} {r['auprc']:>8.5f}"
              f" {r['sensitivity']:>7.4f} {r['specificity']:>7.4f}")
    return mdf


# ── Lead time ──────────────────────────────────────────────────────────────────

def _detection_curves(score, times, gs, ge, e_pat, e_row):
    """Per-patient detection rate, median lead, false-alarm rate vs threshold."""
    n = len(gs)
    onset = np.full(n, np.nan)
    pre_cummax = [None] * n
    pre_times  = [None] * n
    peak = np.empty(n)
    for g in range(n):
        s, e = gs[g], ge[g]
        tt = times[s:e]; sc = score[s:e]
        peak[g] = sc.max()
        ew = np.where(e_row[s:e] == 1)[0]
        if len(ew):
            on = tt[ew[0]]; onset[g] = on
            mask = tt <= on
            if mask.any():
                ts = tt[mask]; ss = sc[mask]
                order = np.argsort(ts)
                ts = ts[order]; ss = ss[order]
                pre_cummax[g] = np.maximum.accumulate(ss)
                pre_times[g]  = ts

    ev  = np.where(e_pat == 1)[0]
    ne  = np.where(e_pat == 0)[0]
    neg_peak = peak[ne]
    thr = np.unique(np.round(np.quantile(score, np.linspace(0.50, 0.9995, 80)), 4))

    det = []; lead = []; fa = []
    for T in thr:
        d = 0; leads = []
        for g in ev:
            cm = pre_cummax[g]
            if cm is None or cm[-1] < T: continue
            idx = np.searchsorted(cm, T, side="left")
            d += 1; leads.append((onset[g] - pre_times[g][idx]) / 60.0)
        det.append(d / len(ev))
        lead.append(np.median(leads) if leads else np.nan)
        fa.append(float(np.mean(neg_peak >= T)))

    return np.array(thr), np.array(det), np.array(lead), np.array(fa)


def run_lead_time(df, pv, ewma, slopes, vitals, beta, gamma,
                  news2_row, gs, ge, e_pat, e_row):
    """Compute and save lead time table for the primary (event-optimal) config."""
    times = df["t_minutes"].values.astype(np.float64)
    snap  = sum(pv[v] for v in vitals).astype(np.float32)
    temp  = temporal_score_row(pv, ewma, slopes, vitals, beta, gamma)

    systems = {"NEWS-2": news2_row.astype(np.float64),
               "Snapshot": snap.astype(np.float64),
               "Temporal": temp.astype(np.float64)}
    curves = {name: _detection_curves(sc, times, gs, ge, e_pat, e_row)
              for name, sc in systems.items()}

    # Only the lead-time-vs-detection-accuracy view is kept; the false-alarm-rate
    # view (computed above by _detection_curves but unused here) was dropped.
    det_targets = [0.60, 0.70, 0.80, 0.90]
    rows = []
    for name, (thr, det, ld, fa_arr) in curves.items():
        order = np.argsort(det)
        for D in det_targets:
            li = float(np.interp(D, det[order], ld[order]))
            rows.append({"system": name, "detection_rate": D,
                         "median_lead_h": round(li, 1)})
    tab = pd.DataFrame(rows)
    tab.to_csv(PL_BASE / "lead_time_table.csv", index=False)

    print(f"\n{'='*70}")
    print("Lead time at matched detection accuracy (+ACVPU, event-optimal):")
    print(tab.pivot_table(index="detection_rate", columns="system",
                          values="median_lead_h").to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {"NEWS-2": "#E74C3C", "Snapshot": "#3498DB", "Temporal": "#2ECC71"}
    for name, (thr, det, ld, fa_arr) in curves.items():
        o = np.argsort(det)
        ax.plot(det[o]*100, ld[o], "o-", ms=3, lw=2, color=colors[name], label=name)
    ax.set_xlabel("Detection accuracy (% event patients caught before onset)", fontsize=11)
    ax.set_ylabel("Median lead time (h)", fontsize=11)
    ax.set_title("Lead time vs detection accuracy", fontsize=11)
    ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Time-to-alert at matched detection — improved system (+ACVPU, event-optimal)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PL_BASE / "lead_time_plot.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → lead_time_table.csv  +  lead_time_plot.png")
    return tab


# ══════════════════════════════════════════════════════════════════════════════

def pick_event_optimal():
    if not GRID_RESULTS.exists():
        raise FileNotFoundError(
            f"Grid results not found: {GRID_RESULTS}\n"
            "Run grid_search_excess_patient.py first.")
    grid = pd.read_csv(GRID_RESULTS)
    best = grid.loc[grid["event"].idxmax()]
    return float(best["alpha"]), float(best["beta"]), float(best["gamma"])


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    opt_alpha, opt_beta, opt_gamma = pick_event_optimal()
    opt = (opt_alpha, opt_beta, opt_gamma)
    param_sets = [
        ("event-optimal", "event_optimal", opt),
        ("Sherif's",      "sherifs_params", FIXED_PARAMS),
    ]
    print("Param sets:")
    for name, folder, (a, b, g) in param_sets:
        print(f"  {name:14s}  α={a}  β={b}  γ={g}  →  {folder}")

    # ── Load ─────────────────────────────────────────────────────────────────
    print("\nLoading dataset…"); t0 = time.time()
    cols = ["ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
            "HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
            "INSPIRED_O2_TEXT", "INSP_O2_CAT", "AVPU_ACVPU", "TEMPERATURE",
            "COMPLETE_DATA", "NEWS-2", "ACVPU_SCORE",
            "DEATH_WITHIN_24H", "ICU_WITHIN_24H", "EVENT_FLAG"]
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    print(f"  After COMPLETE_DATA filter: {len(df):,} rows")

    for c in ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
              "TEMPERATURE", "DAYS_SINCE_ADMISSION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE",
                      "SATS_SPO2", "TEMPERATURE"], inplace=True)

    df["INSPIRED_O2_TEXT"] = (pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce")
                              .fillna(21.0).clip(21.0, 100.0))
    df["NEWS-2"]     = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0)
    # ACVPU_SCORE is the NEWS-2 consciousness sub-score (0 if Alert, 3 otherwise);
    # subtracted for the 6-vital leaves so NEWS-2 excludes consciousness there too.
    df["ACVPU_SCORE"] = pd.to_numeric(df["ACVPU_SCORE"], errors="coerce").fillna(0.0)
    df["ACVPU_NUM"]  = df["AVPU_ACVPU"].map(es.ACVPU_MAP).fillna(0.0)
    df["O2_CONCERN"] = df["INSP_O2_CAT"].map(O2CAT_CONCERN).fillna(0.0).astype(np.float32)

    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"] * 1440.0
                       + obs.dt.hour.fillna(0) * 60.0
                       + obs.dt.minute.fillna(0)
                       + obs.dt.second.fillna(0) / 60.0).astype(np.float32)
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    on_o2 = df["O2_CONCERN"].values > 0
    print(f"  O2 via INSP_O2_CAT: {on_o2.sum():,} rows ({100*on_o2.mean():.1f}%)")

    # ── LUTs + per-vital scores ───────────────────────────────────────────────
    print("\nBuilding fuzzy LUTs (engine, five-set SBP, 6 vitals)…")
    # ACVPU is not a scored vital (flag only) — this stays the 6-vital set
    vitals_full = es.VITALS_BASE
    luts = {v: es.build_lut(v) for v in vitals_full}
    pv = build_pv(df, luts, vitals_full)

    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)

    # ── Labels ───────────────────────────────────────────────────────────────
    d_row    = df["DEATH_WITHIN_24H"].values
    i_row    = df["ICU_WITHIN_24H"].values
    e_row    = df["EVENT_FLAG"].values
    news2_row = df["NEWS-2"].values.astype(np.float64)
    # Consciousness-stripped NEWS-2 for the 6-vital (no-ACVPU) leaves: a matched
    # comparison where NEWS-2, Snapshot and Temporal all exclude ACVPU. The full
    # NEWS-2 is kept for the 7-vital (+ACVPU) leaves.
    news2_row_noacvpu = np.maximum(0.0, news2_row - df["ACVPU_SCORE"].values.astype(np.float64))
    d_pat = np.maximum.reduceat(d_row, gs)
    i_pat = np.maximum.reduceat(i_row, gs)
    e_pat = np.maximum.reduceat(e_row, gs)
    news2_pat = np.maximum.reduceat(news2_row, gs)
    news2_pat_noacvpu = np.maximum.reduceat(news2_row_noacvpu, gs)
    print(f"  Patients: {len(gs):,}  "
          f"(event pos={int(e_pat.sum()):,}, neg={int((e_pat==0).sum()):,})")

    # ── Row-level downsampled index (all events + capped non-events) ──────────
    pos_idx = np.where(e_row == 1)[0]
    neg_idx = np.where(e_row == 0)[0]
    neg_samp = rng.choice(neg_idx, min(ROW_NEG_CAP, len(neg_idx)), replace=False)
    row_idx = np.sort(np.concatenate([pos_idx, neg_samp]))
    print(f"  Row-level eval set: {len(row_idx):,} rows "
          f"({len(pos_idx):,} event + {len(neg_samp):,} non-event sample)")

    # ── Config matrix ─────────────────────────────────────────────────────────
    vital_variants = {"": es.VITALS_BASE, "acvpu": vitals_full}

    all_mdf = []
    primary_leaf = None   # store (pv_vitals, ewma, slopes, vitals, beta, gamma) for lead time

    for pname, folder, (alpha, beta, gamma) in param_sets:
        print(f"\n{'='*70}\nParam set: {pname}  (α={alpha}, β={beta}, γ={gamma})")
        alphas = {v: float(alpha) for v in vitals_full}
        refs   = {v: es.EWMA_REF_DEFAULT for v in vitals_full}
        times  = df["t_minutes"].values.astype(np.float64)
        t1 = time.time()
        ewma = es.compute_ewma(times, pv, gs, ge, vitals_full, alphas, refs)
        slopes = es.compute_slopes(times, pv, gs, ge, vitals_full)
        print(f"  EWMA + slopes done in {time.time()-t1:.0f}s")

        for sub, vitals in vital_variants.items():
            out_dir = PL_BASE / folder / sub if sub else PL_BASE / folder
            vlabel  = ("7 vitals (incl. ACVPU)" if sub else
                       "6 vitals (no ACVPU; NEWS-2 excl. consciousness)")
            label   = f"{pname}: α={alpha} β={beta} γ={gamma} | {vlabel}"

            # 6-vital leaves: NEWS-2 also excludes consciousness (matched comparison)
            news2_pat_use = news2_pat if sub else news2_pat_noacvpu
            news2_row_use = news2_row if sub else news2_row_noacvpu

            mdf = run_leaf(
                gs, d_pat, i_pat, e_pat, news2_pat_use,
                {v: pv[v] for v in vitals}, ewma, slopes, vitals,
                beta, gamma,
                out_dir, label,
                d_row, i_row, e_row, news2_row_use, row_idx)

            mdf["config"]      = label
            mdf["param_set"]   = pname
            mdf["vitals"]      = vlabel
            all_mdf.append(mdf)

            # primary config for lead time
            if pname == "event-optimal" and sub == "acvpu":
                primary_leaf = ({v: pv[v] for v in vitals}, ewma, slopes,
                                vitals, float(beta), float(gamma))

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary = pd.concat(all_mdf, ignore_index=True)
    summary_clean = summary.drop(columns=["_fpr","_tpr","_y","_s"], errors="ignore")
    summary_clean.to_csv(PL_BASE / "all_metrics_summary.csv", index=False)
    print(f"\nSaved all_metrics_summary.csv ({len(summary_clean)} rows)")

    # ── Lead time analysis (primary config) ───────────────────────────────────
    if primary_leaf is not None:
        pv_lt, ewma_lt, slopes_lt, vitals_lt, beta_lt, gamma_lt = primary_leaf
        run_lead_time(df, pv_lt, ewma_lt, slopes_lt, vitals_lt, beta_lt, gamma_lt,
                      news2_row, gs, ge, e_pat, e_row)

    print("\nDone.")


if __name__ == "__main__":
    main()
