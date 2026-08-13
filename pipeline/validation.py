"""Validation leaves for results/main — AUROC, AUPRC and lead time at both levels.

Reads the winning α/β/γ from grid_search_main.py's best_configs.csv and evaluates
three systems (NEWS-2, Snapshot Fuzzy, Temporal Fuzzy) against three targets
(death / ICU / event within 24 h) under every combination of:

  param set : the patient-level event-AUROC winner, the patient-level event-AUPRC
              winner (dropped if identical), and Sherif's fixed (0.5, 5.0, 0.75)
  ACVPU     : 6 vitals with NEWS-2's consciousness sub-score stripped (the matched
              like-for-like comparison, written to the leaf root) and 7 vitals with
              full NEWS-2 (written to ACVPU included/)
  level     : patient (peak score per admission) and row (per observation)

Two things differ from the previous generation of this folder:

  • No negative sampling anywhere in the metrics. Patient level uses all admissions
    as before, and row level now uses all 9.3M observations rather than a 500k
    non-event cap. The cap was defensible for AUROC but silently inflated AUPRC by
    raising apparent prevalence ~1.2% → ~18%. scored_data_full.csv is still written
    from a capped sample (file size only) — the metrics are not.
  • Lead time is measured backward from the true event time and reported at BOTH
    levels; see common.lead_time_patient / lead_time_row.

Outputs → results/main_dataset/validation/{patient_level, row_level}/
"""

import sys, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# Set the paths explicitly rather than relying on the interpreter adding this file's
# directory, or on common.py's own sys.path side effect, to find engine/.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C                                              # noqa: E402
import engine_scoring as es                                     # noqa: E402
from stats import delong_auc_ci, delong_roc_test                # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

SCORED_ROW_CAP = 500_000     # non-event rows written to scored_data_full.csv (export only)


# ── Metrics ───────────────────────────────────────────────────────────────────

def leaf_metrics(d, i, e, scores: dict):
    """AUROC (+DeLong 95% CI), AUPRC, sens/spec at Youden's J, for each system."""
    out = {}
    for name, sc in scores.items():
        per_target = {}
        for tname, tcol in C.TARGETS.items():
            pm, nm = es.pools(d, i, e, C.TARGET_SHORT[tcol])
            keep = pm | nm
            y = pm[keep].astype(np.int8)
            s = np.asarray(sc)[keep].astype(np.float64)
            m = np.isfinite(s)
            y, s = y[m], s[m]
            if y.sum() == 0 or y.sum() == len(y):
                per_target[tname] = None
                continue
            auc, lo, hi = delong_auc_ci(y, s)
            fpr, tpr, thr = roc_curve(y, s)
            j = int(np.argmax(tpr - fpr))
            per_target[tname] = {
                "auroc": auc, "ci_lo": lo, "ci_hi": hi,
                "auprc": float(average_precision_score(y, s)),
                "sensitivity": float(tpr[j]), "specificity": float(1.0 - fpr[j]),
                "threshold": float(thr[j]), "n": int(len(y)), "n_pos": int(y.sum()),
                "prevalence": float(y.mean()),
            }
        out[name] = per_target
    return out


def paired_tests(d, i, e, scores: dict):
    """Paired DeLong: Temporal vs Snapshot, and Snapshot vs NEWS-2."""
    rows = []
    pairs = [("Temporal Fuzzy", "Snapshot Fuzzy"), ("Snapshot Fuzzy", "NEWS-2")]
    for tname, tcol in C.TARGETS.items():
        pm, nm = es.pools(d, i, e, C.TARGET_SHORT[tcol])
        keep = pm | nm
        y = pm[keep].astype(np.int8)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for a, b in pairs:
            sa = np.asarray(scores[a])[keep].astype(np.float64)
            sb = np.asarray(scores[b])[keep].astype(np.float64)
            r = delong_roc_test(y, sa, sb)
            rows.append({"Target": tname, "Comparison": f"{a} - {b}",
                         "AUROC A": round(r["auc_a"], 5),
                         "AUROC B": round(r["auc_b"], 5),
                         "Delta AUROC": round(r["delta"], 5),
                         "Delta 95% CI": f"{r['ci95'][0]:+.5f}–{r['ci95'][1]:+.5f}",
                         "p (two-sided)": f"{r['p']:.3g}",
                         "Significant at 0.05": "yes" if r["p"] < 0.05 else "no"})
    return pd.DataFrame(rows)


def write_auroc_csv(metrics, out_dir):
    rows = []
    for name in C.SYSTEMS:
        row = {"System": name}
        for tname in C.TARGETS:
            m = metrics[name][tname]
            row[tname] = round(m["auroc"], 4) if m else np.nan
            row[f"{tname} 95% CI"] = f"{m['ci_lo']:.4f}–{m['ci_hi']:.4f}" if m else ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "AUROC.csv", index=False)


def write_metrics_full(metrics, out_dir):
    rows = []
    for name in C.SYSTEMS:
        for tname in C.TARGETS:
            m = metrics[name][tname]
            if not m:
                continue
            rows.append({"System": name, "Target": tname,
                         "AUROC": round(m["auroc"], 5),
                         "AUROC 95% CI low": round(m["ci_lo"], 5),
                         "AUROC 95% CI high": round(m["ci_hi"], 5),
                         "AUPRC": round(m["auprc"], 5),
                         "Prevalence": round(m["prevalence"], 6),
                         "Sensitivity at Youden J": round(m["sensitivity"], 4),
                         "Specificity at Youden J": round(m["specificity"], 4),
                         "Threshold at Youden J": round(m["threshold"], 4),
                         "N": m["n"], "N positive": m["n_pos"]})
    pd.DataFrame(rows).to_csv(out_dir / "metrics_full.csv", index=False)


def plot_bar(metrics, key, ylabel, out_dir, label):
    targets = list(C.TARGETS.keys())
    x, w = np.arange(len(targets)), 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    allv = []
    for k, name in enumerate(C.SYSTEMS):
        vals = [metrics[name][t][key] if metrics[name][t] else np.nan for t in targets]
        allv += [v for v in vals if np.isfinite(v)]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=name,
                      color=C.SYS_COLOR[name], edgecolor="white", lw=0.5)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} — {label}", fontsize=10)
    ax.legend(fontsize=10)
    if allv:
        ax.set_ylim(max(0.0, min(allv) - 0.05), min(1.0, max(allv) + 0.10))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"bar_{key}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lead_time(curves, out_dir, label, level):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"NEWS-2": "#E74C3C", "Snapshot": "#3498DB", "Temporal": "#2ECC71"}
    for name, (thr, det, lead, fa) in curves.items():
        o = np.argsort(det)
        axes[0].plot(det[o] * 100, lead[o], "o-", ms=3, lw=2, color=colors[name], label=name)
        axes[1].plot(fa[o] * 100, det[o] * 100, "o-", ms=3, lw=2, color=colors[name], label=name)
    unit = "event admissions" if level == "patient" else "pre-event observations"
    axes[0].set_xlabel(f"Sensitivity (% {unit} detected)", fontsize=10)
    axes[0].set_ylabel("Median lead before event (h)", fontsize=10)
    axes[0].set_title("Lead time vs sensitivity", fontsize=10)
    axes[1].set_xlabel("False-alarm rate (%)", fontsize=10)
    axes[1].set_ylabel(f"Sensitivity (% {unit} detected)", fontsize=10)
    axes[1].set_title("Sensitivity vs false alarms", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(f"Time to alert before the event — {level} level\n{label}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "lead_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── Param sets ────────────────────────────────────────────────────────────────

def load_param_sets():
    """Patient-level event winners under each metric, plus Sherif's fixed set."""
    best_path = C.GRID_PATIENT_DIR / "best_configs.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} missing — run pipeline/grid_search_main.py first.")
    best = pd.read_csv(best_path)
    sel = best[(best["level"] == "patient") & (best["target"] == "event")]

    sets, seen = [], {}
    for metric, folder in (("AUROC", "Grid search params (AUROC)"),
                           ("AUPRC", "Grid search params (AUPRC)")):
        r = sel[sel["metric"] == metric].iloc[0]
        key = (round(float(r["alpha"]), 3), round(float(r["beta"]), 3), round(float(r["gamma"]), 3))
        if key in seen:
            # Both metrics picked the same point — collapse to one folder named for both.
            sets[seen[key]] = ("Grid search params (AUROC + AUPRC)", key,
                               "patient-level event optimum under both metrics")
            continue
        seen[key] = len(sets)
        sets.append((folder, key, f"patient-level event {metric} optimum"))
    sets.append(("Sherifs params", C.FIXED_PARAMS, "fixed prior baseline"))
    return sets


# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    rng = np.random.default_rng(C.RANDOM_SEED)

    param_sets = load_param_sets()
    print("Param sets:")
    for folder, (a, b, g), why in param_sets:
        print(f"  {folder:38s} α={a} β={b} γ={g}   ({why})")

    df = C.load(all_columns=True)
    # ACVPU is not a scored vital (flag only) — this stays the 6-vital set
    vitals_full = es.VITALS_BASE
    print("\nBuilding fuzzy LUTs (engine, five-set SBP, 6 vitals)…")
    luts = C.build_luts(vitals_full)
    pv = C.build_pv(df, luts, vitals_full)

    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    times = df["t_minutes"].values

    d_row = df["DEATH_WITHIN_24H"].values
    i_row = df["ICU_WITHIN_24H"].values
    e_row = df["EVENT_FLAG"].values
    news2_row = df["NEWS-2"].values.astype(np.float64)
    # 6-vital leaves compare against NEWS-2 with its consciousness sub-score removed,
    # so no system has a consciousness input the others lack.
    news2_row_noacvpu = np.maximum(0.0, news2_row - df["ACVPU_SCORE"].values.astype(np.float64))

    d_pat = np.maximum.reduceat(d_row, gs)
    i_pat = np.maximum.reduceat(i_row, gs)
    e_pat = np.maximum.reduceat(e_row, gs)
    news2_pat = np.maximum.reduceat(news2_row, gs)
    news2_pat_noacvpu = np.maximum.reduceat(news2_row_noacvpu, gs)
    print(f"  Patients: {len(gs):,}  (event pos={int(e_pat.sum()):,} = "
          f"{100*e_pat.mean():.2f}%)   rows: {len(df):,} (event {100*e_row.mean():.2f}%)")

    print("\nDeriving true event times (death = last obs; ICU = obs before the >24h gap)…")
    death_t, icu_t, event_t = C.derive_event_times(df, gs, ge)
    have = np.isfinite(event_t) & (e_pat == 1)
    print(f"  event admissions with a recoverable event time: {int(have.sum()):,} / "
          f"{int(e_pat.sum()):,}")

    # Export sample for scored_data_full.csv (file size only — NOT the metric basis)
    pos_idx = np.flatnonzero(e_row == 1)
    neg_idx = np.flatnonzero(e_row == 0)
    neg_samp = rng.choice(neg_idx, min(SCORED_ROW_CAP, len(neg_idx)), replace=False)
    export_idx = np.sort(np.concatenate([pos_idx, neg_samp]))
    print(f"  scored_data_full.csv export sample: {len(export_idx):,} rows")

    print("\nPrecomputing OLS trend slopes (α/β/γ-independent, computed once)…")
    t0 = time.time()
    slopes = es.compute_slopes(times, pv, gs, ge, vitals_full)
    print(f"  done in {time.time()-t0:.0f}s")

    auprc_rows = {"patient": [], "row": []}
    lead_headline = {}

    for folder, (alpha, beta, gamma), why in param_sets:
        print(f"\n{'='*76}\n{folder}:  α={alpha}  β={beta}  γ={gamma}   ({why})")
        t0 = time.time()
        ewma = es.compute_ewma(times, pv, gs, ge, vitals_full,
                               {v: float(alpha) for v in vitals_full},
                               {v: es.EWMA_REF_DEFAULT for v in vitals_full})
        print(f"  EWMA in {time.time()-t0:.0f}s")

        for acvpu in (False, True):
            vitals = vitals_full if acvpu else es.VITALS_BASE
            vlabel = ("7 vitals (incl. ACVPU)" if acvpu else
                      "6 vitals (no ACVPU; NEWS-2 excl. consciousness)")
            label = f"{folder}: α={alpha} β={beta} γ={gamma} | {vlabel}"
            sub = "ACVPU included" if acvpu else ""

            snap_row = sum(pv[v] for v in vitals).astype(np.float64)
            temp_row = C.temporal({v: pv[v] for v in vitals}, ewma, slopes,
                                  vitals, beta, gamma).astype(np.float64)
            n2_row = news2_row if acvpu else news2_row_noacvpu
            n2_pat = news2_pat if acvpu else news2_pat_noacvpu
            snap_pat = np.maximum.reduceat(snap_row, gs)
            temp_pat = np.maximum.reduceat(temp_row, gs)

            level_data = {
                "patient": (C.PATIENT_DIR, d_pat, i_pat, e_pat,
                            {"NEWS-2": n2_pat, "Snapshot Fuzzy": snap_pat,
                             "Temporal Fuzzy": temp_pat}),
                "row": (C.ROW_DIR, d_row, i_row, e_row,
                        {"NEWS-2": n2_row, "Snapshot Fuzzy": snap_row,
                         "Temporal Fuzzy": temp_row}),
            }

            for level, (base, d, i, e, scores) in level_data.items():
                out_dir = base / folder / sub if sub else base / folder
                out_dir.mkdir(parents=True, exist_ok=True)
                t1 = time.time()

                metrics = leaf_metrics(d, i, e, scores)
                write_auroc_csv(metrics, out_dir)
                write_metrics_full(metrics, out_dir)
                plot_bar(metrics, "auroc", "AUROC", out_dir, label)
                plot_bar(metrics, "auprc", "AUPRC", out_dir, label)
                paired_tests(d, i, e, scores).to_csv(
                    out_dir / "paired_delong_tests.csv", index=False)

                for name in C.SYSTEMS:
                    r = {"Params": folder, "Level": f"{level} level",
                         "ACVPU included": "Yes" if acvpu else "No (default)",
                         "System": name}
                    for tname in C.TARGETS:
                        m = metrics[name][tname]
                        r[tname] = round(m["auprc"], 4) if m else np.nan
                    auprc_rows[level].append(r)

                # ── Lead time, measured backward from the true event ──────────
                # Always fed the ROW-level trajectories: lead time is a timing
                # question, so even the patient-level table walks each admission's
                # observations — the peak scores would have no time axis at all.
                lt_scores = {"NEWS-2": n2_row, "Snapshot": snap_row, "Temporal": temp_row}
                curves = {}
                for name, sc in lt_scores.items():
                    if level == "patient":
                        curves[name] = C.lead_time_patient(sc, times, gs, ge, event_t, e_pat)
                    else:
                        curves[name] = C.lead_time_row(sc, times, gs, ge, event_t, e_row)
                tab = C.lead_time_table(curves)
                tab.to_csv(out_dir / "lead_time.csv", index=False)
                plot_lead_time(curves, out_dir, label, level)
                if acvpu:
                    lead_headline[(level, folder)] = (tab, label)

                # ── scored_data_full.csv ──────────────────────────────────────
                if level == "patient":
                    n_obs = ge - gs
                    los = np.maximum.reduceat(df["DAYS_SINCE_ADMISSION"].values, gs)
                    pat = pd.DataFrame({
                        "ANON_ADMISSION_ID": df["ANON_ADMISSION_ID"].values[gs],
                        "N_OBSERVATIONS": n_obs,
                        "LENGTH_OF_STAY_DAYS": los,
                        "DISCHARGING_SPECIALTY": df["DISCHARGING_SPECIALTY"].values[gs],
                        "DIED_FLAG": df["DIED_FLAG"].values[gs],
                        "ICU_FLAG": df["ICU_FLAG"].values[gs],
                        "EVENT_TIME_MINUTES": event_t,
                        "NEWS-2 (peak, used in this leaf)": n2_pat,
                        "Snapshot Fuzzy (peak)": snap_pat,
                        "Temporal Fuzzy (peak)": temp_pat,
                        "Death 24h": d_pat, "ICU 24h": i_pat, "Event 24h": e_pat})
                    pat.to_csv(out_dir / "scored_data_full.csv", index=False)
                else:
                    exp = df.iloc[export_idx].copy()
                    exp["EVENT_TIME_MINUTES"] = np.repeat(event_t, ge - gs)[export_idx]
                    exp["NEWS-2 (used in this leaf)"] = n2_row[export_idx]
                    exp["Snapshot Fuzzy"] = snap_row[export_idx]
                    exp["Temporal Fuzzy"] = temp_row[export_idx]
                    exp.to_csv(out_dir / "scored_data_full.csv", index=False)

                print(f"  {level:8s} {sub or 'no ACVPU':15s} "
                      f"AUROC event: " + "  ".join(
                          f"{n.split()[0]}={metrics[n]['Event 24h']['auroc']:.4f}"
                          for n in C.SYSTEMS)
                      + f"   ({time.time()-t1:.0f}s)", flush=True)

    # Headline lead-time copy = the recommended config (+ACVPU). AUPRC picks it when
    # the two metrics disagree: under this much class imbalance precision-recall area
    # is the more informative ranking, and AUROC's winner is reported in its own leaf.
    grid_folders = [f for f, _, _ in param_sets if f.startswith("Grid search")]
    headline_folder = next((f for f in grid_folders if "AUPRC" in f),
                           grid_folders[0] if grid_folders else None)

    for level, base in (("patient", C.PATIENT_DIR), ("row", C.ROW_DIR)):
        pd.DataFrame(auprc_rows[level]).to_csv(base / "AUPRC_summary.csv", index=False)
        key = (level, headline_folder)
        if key in lead_headline:
            lead_headline[key][0].to_csv(base / "lead_time_summary.csv", index=False)

    print(f"\nDone in {time.time()-t_total:.0f}s.")
    print(f"Headline lead-time config: {headline_folder} (+ACVPU)")
    for level in ("patient", "row"):
        key = (level, headline_folder)
        if key in lead_headline:
            tab, label = lead_headline[key]
            print(f"\n── Lead time, {level} level ({label}) ──")
            print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
