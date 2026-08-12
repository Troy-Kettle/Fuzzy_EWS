"""
Experiment 07_oxygen_fix

Fixes a real data bug in the inspired-oxygen vital. INSPIRED_O2_TEXT mixes two
units: low values (1,2,3,4,6,15 …) are oxygen FLOW in L/min, higher values
(28,35,40,60 …) are FiO2 %. The pipeline's .clip(21,100) collapsed every
low-flow value up to 21 ("room air"), silently discarding ~1.3M supplemental-
oxygen observations (we saw 2.3% on O2; the chart shows 16.4%).

Fix: score the oxygen vital from the RECORDED clinical category INSP_O2_CAT
(Low / Low-moderate / Moderate / High / Very high, blank = room air) instead of
the broken mixed-unit numeric field. This is observed clinical data, not NEWS-2's
derived score, and it is naturally graded — suited to a fuzzy concern gradient.

Category → concern is a fixed, clinically-reasoned EVEN ramp (NOT tuned to AUROC):
  room air = 0, Low = 1.0, Low-moderate = 1.5, Moderate = 2.0, High = 2.5, Very high = 3.0
Only the usual temporal params (α/β/γ) are grid-searched, as in every prior experiment.

Everything else inherits the best config: additive, sharper SBP, +ACVPU, temporal.
Supersedes 06's oxygen handling (which used a binary flag on the wrong field).

Outputs → improved_results/07_oxygen_fix/ (same structure as prior experiments).
"""

import time, warnings, json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd

import engine_scoring as es
from run_all_experiments import (alphas_refs, eval_config, time_to_alert,
                                  bars, tta_plot, grid_search, SHERIF,
                                  ROW_NEG_CAP, NE_PATIENTS)

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

REPO    = Path(__file__).resolve().parent.parent
DATA    = REPO / "datasets" / "final_observations_with_targets.csv"
OUT_DIR = REPO / "improved_results" / "07_oxygen_fix"; OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED    = 42

# Fixed, untuned clinical ramp for recorded oxygen-delivery category
O2CAT_CONCERN = {"Low": 1.0, "Low-moderate": 1.5, "Moderate": 2.0,
                 "High": 2.5, "Very high": 3.0}   # blank/NaN (room air) -> 0.0


def load():
    print("Loading dataset…"); t0 = time.time()
    cols = ["ANON_ADMISSION_ID","OBS_TIME","DAYS_SINCE_ADMISSION","HEART_RATE","SYSTOLIC_BP",
            "RESP_RATE","SATS_SPO2","INSPIRED_O2_TEXT","INSP_O2_CAT","AVPU_ACVPU","TEMPERATURE",
            "COMPLETE_DATA","NEWS-2","DEATH_WITHIN_24H","ICU_WITHIN_24H","EVENT_FLAG"]
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    for c in ["HEART_RATE","SYSTOLIC_BP","RESP_RATE","SATS_SPO2","TEMPERATURE","DAYS_SINCE_ADMISSION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["HEART_RATE","SYSTOLIC_BP","RESP_RATE","SATS_SPO2","TEMPERATURE"], inplace=True)
    # keep INSPIRED_O2_TEXT only as fallback; the oxygen vital now comes from INSP_O2_CAT
    df["INSPIRED_O2_TEXT"] = pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce").fillna(21.).clip(21, 100)
    df["NEWS-2"] = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0)
    df["ACVPU_NUM"] = df["AVPU_ACVPU"].map(es.ACVPU_MAP).fillna(0.0)
    df["O2_CONCERN"] = df["INSP_O2_CAT"].map(O2CAT_CONCERN).fillna(0.0).astype(np.float32)
    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"]*1440. + obs.dt.hour.fillna(0)*60.
                       + obs.dt.minute.fillna(0) + obs.dt.second.fillna(0)/60.).astype(np.float32)
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID","t_minutes"], inplace=True); df.reset_index(drop=True, inplace=True)
    print(f"  {len(df):,} rows in {time.time()-t0:.0f}s")
    on_o2 = (df["O2_CONCERN"].values > 0)
    print(f"  On supplemental O2 (INSP_O2_CAT): {on_o2.sum():,} rows ({100*on_o2.mean():.1f}%)")
    return df


def build_pv(df, luts, vitals):
    """LUT-score every vital, then OVERRIDE inspired_oxygen with the category concern."""
    pv = es.apply_luts(df, luts, vitals)
    pv["inspired_oxygen"] = df["O2_CONCERN"].values.astype(np.float32)   # the fix
    return pv


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    vitals_full = es.VITALS_BASE + [es.ACVPU]

    print("Building LUTs (sharper SBP)…")
    luts = {v: es.build_lut(v) for v in vitals_full}
    luts["blood_pressure"] = es.build_lut("blood_pressure", sharper_sbp=True)

    pv = build_pv(df, luts, vitals_full)

    times = df["t_minutes"].values.astype(np.float64)
    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    d_row = df["DEATH_WITHIN_24H"].values; i_row = df["ICU_WITHIN_24H"].values
    e_row = df["EVENT_FLAG"].values;       news2_row = df["NEWS-2"].values.astype(np.float64)
    d_pat = np.maximum.reduceat(d_row, gs); i_pat = np.maximum.reduceat(i_row, gs)
    e_pat = np.maximum.reduceat(e_row, gs); news2_pat = np.maximum.reduceat(news2_row, gs)
    print(f"  Patients: {len(gs):,} (event {int(e_pat.sum()):,})")

    pos = np.where(e_row == 1)[0]; neg = np.where(e_row == 0)[0]
    row_eval_idx = np.sort(np.concatenate([pos,
        rng.choice(neg, min(ROW_NEG_CAP, len(neg)), replace=False)]))

    # patient sample for grid search
    e_idx = np.where(e_pat == 1)[0]; ne_idx = np.where(e_pat == 0)[0]
    ne_k = rng.choice(ne_idx, min(NE_PATIENTS, len(ne_idx)), replace=False)
    samp = np.sort(np.concatenate([e_idx, ne_k]))
    samp_rows = np.concatenate([np.arange(gs[g], ge[g]) for g in samp])
    df_s = df.iloc[samp_rows].reset_index(drop=True)
    gs_s, ge_s = es.group_boundaries(df_s["ANON_ADMISSION_ID"].values)
    times_s = df_s["t_minutes"].values.astype(np.float64)
    pv_s = build_pv(df_s, luts, vitals_full)
    e_pat_s = np.maximum.reduceat(df_s["EVENT_FLAG"].values.astype(np.int8), gs_s)
    print(f"  Grid sample: {len(df_s):,} rows, {len(gs_s):,} patients")

    t0 = time.time()
    exp_def = dict(name="07_oxygen_fix", methods=["additive"], profiles=["global"],
                   excess_modes=["absolute", "relative"], sharper_sbp=False)  # sharper SBP baked into LUT
    bestcfg, grid_res, best_auc = grid_search(times_s, pv_s, pv_s, gs_s, ge_s, e_pat_s, exp_def)
    grid_res.to_csv(OUT_DIR / "grid_results.csv", index=False)
    print(f"  grid best: {bestcfg}  patient-event AUROC(sample)={best_auc:.4f}  ({time.time()-t0:.0f}s)")

    sherif = dict(SHERIF)
    param_sets = [("event_optimal", bestcfg), ("sherifs", sherif)]

    evals = {}
    for setname, cfg in param_sets:
        for vits, tag in [(es.VITALS_BASE, "6vital"), (vitals_full, "+ACVPU")]:
            out, (snap, temp) = eval_config(
                df, {v: pv[v] for v in vits}, vits, cfg, gs, ge, times,
                d_row, i_row, e_row, news2_row, d_pat, i_pat, e_pat, news2_pat, row_eval_idx)
            evals[(setname, tag)] = (out, snap, temp)

    all_rows = []
    for setname, cfg in param_sets:
        recs = []
        for tag in ["6vital", "+ACVPU"]:
            out = evals[(setname, tag)][0]
            for (sysn, lvl, tgt), v in out.items():
                recs.append(dict(system=sysn, level=lvl, vitals=tag, target=tgt, auroc=round(v, 5)))
                all_rows.append(dict(experiment="07_oxygen_fix", param_set=setname, vitals=tag,
                    system=sysn, level=lvl, target=tgt, auroc=round(v, 5),
                    **{k: cfg[k] for k in ["alpha","beta","gamma","method","profile","excess_mode"]}))
        pd.DataFrame(recs).pivot_table(index=["vitals","level","system"], columns="target",
            values="auroc").to_csv(OUT_DIR / f"auroc_{setname}.csv")

    bars(evals[("event_optimal","6vital")][0], evals[("sherifs","6vital")][0],
         OUT_DIR / "summary_bars.png", "07_oxygen_fix  (patient-level AUROC, 6 vitals)")

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
            all_rows.append(dict(experiment="07_oxygen_fix", param_set="event_optimal", vitals=tag,
                system=s, level="tta_detected_frac", target="event", auroc=round(dd["detected_frac"],5),
                **{k: bestcfg[k] for k in ["alpha","beta","gamma","method","profile","excess_mode"]}))
            all_rows.append(dict(experiment="07_oxygen_fix", param_set="event_optimal", vitals=tag,
                system=s, level="tta_median_lead_h", target="event", auroc=round(dd["median_lead_h"],3),
                **{k: bestcfg[k] for k in ["alpha","beta","gamma","method","profile","excess_mode"]}))
    pd.DataFrame(ttarecs).to_csv(OUT_DIR / "time_to_alert.csv", index=False)
    tta_plot(tta6, tta7, OUT_DIR / "time_to_alert.png", "07_oxygen_fix — time-to-alert")

    with open(OUT_DIR / "best_config.json", "w") as f:
        json.dump(bestcfg, f, indent=2)

    print("\n=== Patient-level AUROC (event-optimal, 6 vitals) ===")
    for (sysn, lvl, tgt), v in evals[("event_optimal","6vital")][0].items():
        if lvl == "patient": print(f"  {sysn:12s} {tgt:6s}: {v:.4f}")
    print("\n=== Patient-level AUROC (event-optimal, +ACVPU) ===")
    for (sysn, lvl, tgt), v in evals[("event_optimal","+ACVPU")][0].items():
        if lvl == "patient": print(f"  {sysn:12s} {tgt:6s}: {v:.4f}")

    all_csv = REPO / "improved_results" / "ALL_RESULTS.csv"
    existing = pd.read_csv(all_csv) if all_csv.exists() else pd.DataFrame()
    if len(existing): existing = existing[existing["experiment"] != "07_oxygen_fix"]
    pd.concat([existing, pd.DataFrame(all_rows)], ignore_index=True).to_csv(all_csv, index=False)

    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
