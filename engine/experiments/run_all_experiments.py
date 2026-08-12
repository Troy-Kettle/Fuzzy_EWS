"""
Improvement-experiment driver. Loads the dataset ONCE and runs a cumulative ladder
of experiments, each evaluated at BOTH row-level and patient-level AUROC plus a
time-to-alert (lead-time) analysis, for two parameter sets (grid-optimal + Sherif's
α=0.5/β=5/γ=0.75). Uses experiment_code/engine_scoring.py as the single source of
truth (A1 zero-rule, A2 ACVPU map, A3 shared scoring).

Experiment ladder (each adds one improvement layer):
  01_fidelity            A1+A2+A3, additive, global excess-EWMA
  02_aggregation         + aggregation-method search (additive/multiplicative/nonlinear)  [B1/B2]
  03_sharper_sbp         02-best method + sharper SBP membership                           [C1]
  04_pervital_temporal   02-best method + per-vital α & EWMA ref (physio) + excess scaling [D1/D2/D3]
  05_combined_best       best method + sharper SBP + per-vital temporal

Outputs → improved_results/<exp>/:
  grid_results.csv, summary_bars.png,
  auroc_event_optimal.csv, auroc_sherifs.csv      (row & patient columns side by side)
  time_to_alert.csv, time_to_alert.png
A machine-readable improved_results/ALL_RESULTS.csv aggregates everything for the
top-level comparison document.
"""

import sys, time, warnings, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine_scoring as es

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

REPO      = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO / "datasets" / "final_observations_with_targets.csv"
OUT_ROOT  = REPO / "results" / "experiments"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED        = 42
NE_PATIENTS = 22_336
ROW_NEG_CAP = 500_000          # row-level eval negative cap
SHERIF      = dict(alpha=0.5, beta=5.0, gamma=0.75, method="additive",
                   profile="global", excess_mode="absolute", power=2.0)

# compact but representative grids
A_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
B_GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5]
G_GRID = [0.5, 0.7, 0.9, 1.0]
TARGETS = ["death", "icu", "event"]


# ───────────────────────── data ──────────────────────────
def load():
    print("Loading dataset…"); t0 = time.time()
    cols = ["ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
            "HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
            "INSPIRED_O2_TEXT", "AVPU_ACVPU", "TEMPERATURE",
            "COMPLETE_DATA", "NEWS-2", "DEATH_WITHIN_24H", "ICU_WITHIN_24H", "EVENT_FLAG"]
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    for c in ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2", "TEMPERATURE", "DAYS_SINCE_ADMISSION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2", "TEMPERATURE"], inplace=True)
    df["INSPIRED_O2_TEXT"] = pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce").fillna(21.).clip(21, 100)
    df["NEWS-2"] = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0)
    df["ACVPU_NUM"] = df["AVPU_ACVPU"].map(es.ACVPU_MAP).fillna(0.0)
    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"] * 1440. + obs.dt.hour.fillna(0)*60.
                       + obs.dt.minute.fillna(0) + obs.dt.second.fillna(0)/60.).astype(np.float32)
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df):,} rows in {time.time()-t0:.0f}s")
    return df


# ───────────────── temporal profile helpers ─────────────────
def alphas_refs(cfg, vitals):
    if cfg["profile"] == "physio":
        return ({v: es.PHYSIO_ALPHA[v] for v in vitals},
                {v: es.PHYSIO_REF[v]   for v in vitals})
    return ({v: cfg["alpha"] for v in vitals},
            {v: es.EWMA_REF_DEFAULT for v in vitals})


# ───────────────── grid search (patient-level event AUROC) ─────────────────
def grid_search(times, pv_std, pv_shp, gs, ge, e_pat, exp):
    """Return best cfg dict + full results DataFrame. Scored by patient-level event AUROC."""
    methods   = exp["methods"]
    profiles  = exp["profiles"]
    excess_md = exp["excess_modes"]
    sbp_var   = exp["sharper_sbp"]
    vitals    = es.VITALS_BASE                      # grid on 6 vitals (no ACVPU)
    pv        = pv_shp if sbp_var else pv_std

    rows = []; best = None; best_auc = -1
    # EWMA cache keyed by (profile, alpha)
    cache = {}
    def get_ewma(profile, alpha):
        key = (profile, alpha)
        if key not in cache:
            cfg = dict(profile=profile, alpha=alpha)
            a, r = alphas_refs(cfg, vitals)
            cache[key] = es.compute_ewma(times, pv, gs, ge, vitals, a, r)
        return cache[key]

    alpha_grid = A_GRID if "global" in profiles else [None]
    for profile in profiles:
        a_list = A_GRID if profile == "global" else [None]
        for alpha in a_list:
            ew = get_ewma(profile, alpha if alpha is not None else 0.0) if profile == "global" \
                 else get_ewma(profile, 0.0)
            for method in methods:
                for beta in B_GRID:
                    for gamma in G_GRID:
                        for em in excess_md:
                            total = es.temporal_score(pv, ew, vitals, beta, gamma,
                                                      "raise_only", method, em_power(method), em)
                            pscore = es.patient_peak(total, gs)
                            auc = es.auroc(e_pat, e_pat, e_pat, pscore, "event")
                            rec = dict(profile=profile, alpha=alpha, method=method,
                                       beta=beta, gamma=gamma, excess_mode=em, event_auroc=auc)
                            rows.append(rec)
                            if auc > best_auc:
                                best_auc, best = auc, rec
    res = pd.DataFrame(rows)
    bestcfg = dict(alpha=best["alpha"] if best["alpha"] is not None else 0.3,
                   beta=best["beta"], gamma=best["gamma"], method=best["method"],
                   profile=best["profile"], excess_mode=best["excess_mode"], power=2.0)
    return bestcfg, res, best_auc


def em_power(method):
    return 2.0


# ───────────────── evaluation (row + patient) ─────────────────
def eval_config(df, pv, vitals, cfg, gs, ge, times,
                d_row, i_row, e_row, news2_row, d_pat, i_pat, e_pat, news2_pat,
                row_eval_idx):
    a, r = alphas_refs(cfg, vitals)
    ew = es.compute_ewma(times, pv, gs, ge, vitals, a, r)
    snap = es.snapshot_score(pv, vitals, cfg["method"], cfg["gamma"], cfg["power"])
    temp = es.temporal_score(pv, ew, vitals, cfg["beta"], cfg["gamma"],
                             "raise_only", cfg["method"], cfg["power"], cfg["excess_mode"])
    out = {}
    # patient level (peak), full population
    snap_p = es.patient_peak(snap, gs); temp_p = es.patient_peak(temp, gs)
    for sysname, srow, spat in [("NEWS-2", news2_row, news2_pat),
                                ("Snapshot", snap, snap_p),
                                ("Temporal", temp, temp_p)]:
        for tgt in TARGETS:
            out[(sysname, "patient", tgt)] = es.auroc(d_pat, i_pat, e_pat, spat, tgt)
            out[(sysname, "row", tgt)] = es.auroc(d_row[row_eval_idx], i_row[row_eval_idx],
                                                  e_row[row_eval_idx], srow[row_eval_idx], tgt)
    return out, (snap, temp)


# ───────────────── time-to-alert (lead time) ─────────────────
def time_to_alert(df, scores_by_sys, gs, ge, times, e_pat, e_row, target_alert_rate=0.20):
    """For each system, set threshold so a fixed fraction of NON-event patients ever alert,
    then measure lead time (hours from first alert to the event-window onset) for event
    patients. Returns dict per system: detection rate among event patients + median lead h."""
    # event onset proxy per admission = first time EVENT_FLAG==1 within the admission
    n_pat = len(gs)
    onset = np.full(n_pat, np.nan)
    for g in range(n_pat):
        s, e = gs[g], ge[g]
        ew = np.where(e_row[s:e] == 1)[0]
        if len(ew):
            onset[g] = times[s + ew[0]]
    results = {}
    for sysname, score in scores_by_sys.items():
        peak = es.patient_peak(score, gs)
        # threshold from non-event patients' peak distribution
        ne_peak = peak[e_pat == 0]
        thr = np.quantile(ne_peak, 1.0 - target_alert_rate)
        det = 0; leads = []
        ev_idx = np.where(e_pat == 1)[0]
        for g in ev_idx:
            s, e = gs[g], ge[g]
            sc = score[s:e]; tt = times[s:e]
            cross = np.where(sc >= thr)[0]
            if len(cross) == 0:
                continue
            t_alert = tt[cross[0]]
            if not np.isnan(onset[g]) and t_alert <= onset[g]:
                det += 1
                leads.append((onset[g] - t_alert) / 60.0)
        results[sysname] = dict(
            threshold=float(thr),
            detected_frac=det / len(ev_idx),
            median_lead_h=float(np.median(leads)) if leads else 0.0,
            mean_lead_h=float(np.mean(leads)) if leads else 0.0,
            alert_rate_neg=target_alert_rate)
    return results


# ───────────────── plotting ─────────────────
def bars(table_optimal, table_sherif, out_png, title):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (tbl, lab) in zip(axes, [(table_optimal, "event-optimal"), (table_sherif, "Sherif's")]):
        systems = ["NEWS-2", "Snapshot", "Temporal"]
        x = np.arange(len(TARGETS)); w = 0.27
        for k, sysn in enumerate(systems):
            vals = [tbl[(sysn, "patient", t)] for t in TARGETS]
            b = ax.bar(x + (k-1)*w, vals, w, label=sysn)
            for bar, v in zip(b, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([t.upper() for t in TARGETS])
        ax.set_ylim(0.6, 0.95); ax.set_ylabel("Patient-level AUROC")
        ax.set_title(f"{lab}"); ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)


def tta_plot(tta6, tta7, out_png, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (tta, lab) in zip(axes, [(tta6, "6 vitals"), (tta7, "+ACVPU")]):
        sysn = list(tta.keys())
        det = [tta[s]["detected_frac"]*100 for s in sysn]
        lead = [tta[s]["median_lead_h"] for s in sysn]
        x = np.arange(len(sysn))
        ax.bar(x-0.2, det, 0.4, label="Detected % (≤onset)", color="#3498DB")
        ax2 = ax.twinx()
        ax2.bar(x+0.2, lead, 0.4, label="Median lead (h)", color="#E67E22")
        ax.set_xticks(x); ax.set_xticklabels(sysn); ax.set_ylabel("Detected %")
        ax2.set_ylabel("Median lead (h)"); ax.set_title(lab)
        ax.set_ylim(0, 100)
    fig.suptitle(title + "  (@20% non-event alert rate)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)


# ───────────────── experiment definitions ─────────────────
EXPERIMENTS = [
    dict(name="01_fidelity",          methods=["additive"], profiles=["global"],
         excess_modes=["absolute"], sharper_sbp=False),
    dict(name="02_aggregation",       methods=["additive", "multiplicative", "nonlinear"],
         profiles=["global"], excess_modes=["absolute"], sharper_sbp=False),
    dict(name="03_sharper_sbp",       methods=["BEST02"], profiles=["global"],
         excess_modes=["absolute"], sharper_sbp=True),
    dict(name="04_pervital_temporal", methods=["BEST02"], profiles=["physio"],
         excess_modes=["absolute", "relative"], sharper_sbp=False),
    dict(name="05_combined_best",     methods=["BEST02"], profiles=["global", "physio"],
         excess_modes=["absolute", "relative"], sharper_sbp=True),
]


def main():
    rng = np.random.default_rng(SEED)
    df = load()

    vitals_full = es.VITALS_BASE + [es.ACVPU]
    print("Building LUTs (standard + sharper SBP)…")
    luts_std = {v: es.build_lut(v) for v in vitals_full}
    luts_shp = dict(luts_std); luts_shp["blood_pressure"] = es.build_lut("blood_pressure", sharper_sbp=True)
    pv_std = es.apply_luts(df, luts_std, vitals_full)
    pv_shp = es.apply_luts(df, luts_shp, vitals_full)

    times = df["t_minutes"].values.astype(np.float64)
    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    d_row = df["DEATH_WITHIN_24H"].values; i_row = df["ICU_WITHIN_24H"].values
    e_row = df["EVENT_FLAG"].values; news2_row = df["NEWS-2"].values.astype(np.float64)
    d_pat = np.maximum.reduceat(d_row, gs); i_pat = np.maximum.reduceat(i_row, gs)
    e_pat = np.maximum.reduceat(e_row, gs); news2_pat = np.maximum.reduceat(news2_row, gs)
    print(f"  Patients: {len(gs):,} (event {int(e_pat.sum()):,})")

    # row-level eval subset (all positives + capped negatives)
    pos = np.where(e_row == 1)[0]; neg = np.where(e_row == 0)[0]
    negk = rng.choice(neg, min(ROW_NEG_CAP, len(neg)), replace=False)
    row_eval_idx = np.sort(np.concatenate([pos, negk]))

    # patient sample for grid search
    ev_ids = set(df.loc[df["EVENT_FLAG"] == 1, "ANON_ADMISSION_ID"].unique())
    ne_ids = list(set(df["ANON_ADMISSION_ID"].unique()) - ev_ids)
    ne_samp = rng.choice(ne_ids, min(NE_PATIENTS, len(ne_ids)), replace=False)
    keep = df["ANON_ADMISSION_ID"].isin(set(ev_ids) | set(ne_samp)).values
    samp_idx = np.where(keep)[0]
    gss, ges = es.group_boundaries(df["ANON_ADMISSION_ID"].values[samp_idx])
    times_s = times[samp_idx]
    pv_std_s = {v: pv_std[v][samp_idx] for v in vitals_full}
    pv_shp_s = {v: pv_shp[v][samp_idx] for v in vitals_full}
    e_pat_s = np.maximum.reduceat(e_row[samp_idx], gss)
    print(f"  Grid sample: {len(samp_idx):,} rows, {len(gss):,} patients\n")

    best02_method = None
    all_rows = []   # for ALL_RESULTS.csv

    for exp in EXPERIMENTS:
        name = exp["name"]; t0 = time.time()
        print(f"{'='*64}\n{name}\n{'='*64}")
        exp = dict(exp)
        if exp["methods"] == ["BEST02"]:
            exp["methods"] = [best02_method or "additive"]
        out_dir = OUT_ROOT / name; out_dir.mkdir(parents=True, exist_ok=True)

        # grid search → optimal cfg
        bestcfg, res, best_auc = grid_search(times_s, pv_std_s, pv_shp_s, gss, ges, e_pat_s, exp)
        res.to_csv(out_dir / "grid_results.csv", index=False)
        print(f"  grid best: {bestcfg}  patient-event AUROC(sample)={best_auc:.4f}  ({time.time()-t0:.0f}s)")
        if name == "02_aggregation":
            best02_method = bestcfg["method"]

        sharper = exp["sharper_sbp"]
        pv_full = pv_shp if sharper else pv_std

        sherif = dict(SHERIF)   # additive, global α0.5 β5 γ0.75 (fixed reference)
        param_sets = [("event_optimal", bestcfg), ("sherifs", sherif)]

        # compute each (param_set, vital_set) ONCE; stash table + snap/temp arrays
        evals = {}
        for setname, cfg in param_sets:
            for vits, tag in [(es.VITALS_BASE, "6vital"), (vitals_full, "+ACVPU")]:
                out, (snap, temp) = eval_config(
                    df, {v: pv_full[v] for v in vits}, vits, cfg, gs, ge, times,
                    d_row, i_row, e_row, news2_row, d_pat, i_pat, e_pat, news2_pat, row_eval_idx)
                evals[(setname, tag)] = (out, snap, temp)

        for setname, cfg in param_sets:
            recs = []
            for tag in ["6vital", "+ACVPU"]:
                out = evals[(setname, tag)][0]
                for (sysn, lvl, tgt), v in out.items():
                    recs.append(dict(system=sysn, level=lvl, vitals=tag, target=tgt, auroc=round(v, 5)))
                    all_rows.append(dict(experiment=name, param_set=setname, vitals=tag,
                        system=sysn, level=lvl, target=tgt, auroc=round(v, 5),
                        **{k: cfg[k] for k in ["alpha","beta","gamma","method","profile","excess_mode"]}))
            pd.DataFrame(recs).pivot_table(index=["vitals","level","system"], columns="target",
                values="auroc").to_csv(out_dir / f"auroc_{setname}.csv")

        bars(evals[("event_optimal","6vital")][0], evals[("sherifs","6vital")][0],
             out_dir / "summary_bars.png", f"{name}  (patient-level AUROC, 6 vitals)")

        # time-to-alert (event-optimal), 6-vital and +ACVPU, reusing stored arrays
        _, snap6, temp6 = evals[("event_optimal","6vital")]
        _, snap7, temp7 = evals[("event_optimal","+ACVPU")]
        tta6 = time_to_alert(df, {"NEWS-2": news2_row, "Snapshot": snap6, "Temporal": temp6},
                             gs, ge, times, e_pat, e_row)
        tta7 = time_to_alert(df, {"NEWS-2": news2_row, "Snapshot": snap7, "Temporal": temp7},
                             gs, ge, times, e_pat, e_row)
        ttarecs = []
        for tag, tta in [("6vital", tta6), ("+ACVPU", tta7)]:
            for s, dd in tta.items():
                ttarecs.append(dict(vitals=tag, system=s, **dd))
                all_rows.append(dict(experiment=name, param_set="event_optimal", vitals=tag,
                    system=s, level="tta_detected_frac", target="event", auroc=round(dd["detected_frac"],5),
                    **{k: bestcfg[k] for k in ["alpha","beta","gamma","method","profile","excess_mode"]}))
                all_rows.append(dict(experiment=name, param_set="event_optimal", vitals=tag,
                    system=s, level="tta_median_lead_h", target="event", auroc=round(dd["median_lead_h"],3),
                    **{k: bestcfg[k] for k in ["alpha","beta","gamma","method","profile","excess_mode"]}))
        pd.DataFrame(ttarecs).to_csv(out_dir / "time_to_alert.csv", index=False)
        tta_plot(tta6, tta7, out_dir / "time_to_alert.png", f"{name} — time-to-alert")

        with open(out_dir / "best_config.json", "w") as f:
            json.dump(bestcfg, f, indent=2)
        print(f"  {name} done in {time.time()-t0:.0f}s\n")

    pd.DataFrame(all_rows).to_csv(OUT_ROOT / "ALL_RESULTS.csv", index=False)
    print(f"Saved {OUT_ROOT/'ALL_RESULTS.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
