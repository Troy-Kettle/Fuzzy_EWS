"""
Experiment 06_supplemental_o2

NEWS-2 applies a hard +2 bonus for ANY supplemental oxygen (FiO2 > 21%).
This experiment adds that same binary flag ON TOP of the existing graded FiO2
fuzzy score: whenever a patient is on supplemental O2, their inspired_oxygen
vital score becomes min(3, fuzzy_fio2 + 2).

This preserves interpretability:
  - Room air (FiO2=21): score unchanged (graded, 0-3 as before)
  - Any O2 (FiO2>21): graded score + 2, capped at 3

Everything else inherits the best config from 05_combined_best:
  additive aggregation, α=0.1, β=0.5, γ=1.0, relative excess.

Outputs → improved_results/06_supplemental_o2/ (same structure as prior experiments).
"""

import time, warnings, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import engine_scoring as es
# reuse helpers from run_all_experiments
from run_all_experiments import (load, alphas_refs, eval_config, time_to_alert,
                                  bars, tta_plot, grid_search, SHERIF,
                                  A_GRID, B_GRID, G_GRID, TARGETS,
                                  ROW_NEG_CAP, NE_PATIENTS)

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

REPO     = Path(__file__).resolve().parent.parent
OUT_DIR  = REPO / "improved_results" / "06_supplemental_o2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED     = 42


def apply_o2_flag(pv: dict, on_o2: np.ndarray) -> dict:
    """Return a copy of pv with inspired_oxygen score boosted +2 (capped 3) on O2."""
    pv2 = dict(pv)
    base = pv["inspired_oxygen"].astype(np.float32)
    pv2["inspired_oxygen"] = np.where(on_o2, np.clip(base + 2.0, 0.0, 3.0), base).astype(np.float32)
    return pv2


def main():
    rng = np.random.default_rng(SEED)
    df = load()

    vitals_full = es.VITALS_BASE + [es.ACVPU]

    print("Building LUTs (five-set SBP)…")
    luts = {v: es.build_lut(v) for v in vitals_full}

    # Standard LUT-scored vitals
    pv_base = es.apply_luts(df, luts, vitals_full)

    # O2 flag: True wherever patient is on any supplemental oxygen
    on_o2 = (df["INSPIRED_O2_TEXT"].values > 21.0)
    print(f"  Rows on supplemental O2: {on_o2.sum():,} / {len(df):,} ({100*on_o2.mean():.1f}%)")

    # Apply flag → boosted FiO2 scores
    pv_o2 = apply_o2_flag(pv_base, on_o2)

    times = df["t_minutes"].values.astype(np.float64)
    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    d_row = df["DEATH_WITHIN_24H"].values; i_row = df["ICU_WITHIN_24H"].values
    e_row = df["EVENT_FLAG"].values;       news2_row = df["NEWS-2"].values.astype(np.float64)
    d_pat = np.maximum.reduceat(d_row, gs); i_pat = np.maximum.reduceat(i_row, gs)
    e_pat = np.maximum.reduceat(e_row, gs); news2_pat = np.maximum.reduceat(news2_row, gs)
    print(f"  Patients: {len(gs):,} (event {int(e_pat.sum()):,})")

    # Row-level subset
    pos = np.where(e_row == 1)[0]; neg = np.where(e_row == 0)[0]
    row_eval_idx = np.sort(np.concatenate([pos,
        rng.choice(neg, min(ROW_NEG_CAP, len(neg)), replace=False)]))

    # Patient sample for grid search
    e_idx = np.where(e_pat == 1)[0]
    ne_idx = np.where(e_pat == 0)[0]
    ne_k = rng.choice(ne_idx, min(NE_PATIENTS, len(ne_idx)), replace=False)
    samp_idx = np.sort(np.concatenate([e_idx, ne_k]))
    samp_rows = np.concatenate([np.arange(gs[g], ge[g]) for g in samp_idx])
    df_s = df.iloc[samp_rows].reset_index(drop=True)
    gs_s, ge_s = es.group_boundaries(df_s["ANON_ADMISSION_ID"].values)
    times_s = df_s["t_minutes"].values.astype(np.float64)
    on_o2_s = (df_s["INSPIRED_O2_TEXT"].values > 21.0)
    pv_base_s = es.apply_luts(df_s, luts, es.VITALS_BASE)
    pv_o2_s = apply_o2_flag(pv_base_s, on_o2_s)
    e_pat_s = np.maximum.reduceat(
        df_s["EVENT_FLAG"].values.astype(np.int8), gs_s)
    print(f"  Grid sample: {len(df_s):,} rows, {len(gs_s):,} patients")

    # Grid search on sample with O2-flagged vitals
    t0 = time.time()
    exp_def = dict(name="06_supplemental_o2",
                   methods=["additive"], profiles=["global"],
                   excess_modes=["absolute", "relative"], sharper_sbp=False)
    bestcfg, grid_res, best_auc = grid_search(times_s, pv_o2_s, pv_o2_s,
                                               gs_s, ge_s, e_pat_s, exp_def)
    grid_res.to_csv(OUT_DIR / "grid_results.csv", index=False)
    print(f"  grid best: {bestcfg}  patient-event AUROC(sample)={best_auc:.4f}  ({time.time()-t0:.0f}s)")

    # Full-data evaluation for both param sets
    sherif = dict(SHERIF)
    param_sets = [("event_optimal", bestcfg), ("sherifs", sherif)]

    evals = {}
    for setname, cfg in param_sets:
        for vits, tag in [(es.VITALS_BASE, "6vital"), (vitals_full, "+ACVPU")]:
            pv_full = {v: pv_o2[v] for v in vits}
            out, (snap, temp) = eval_config(
                df, pv_full, vits, cfg, gs, ge, times,
                d_row, i_row, e_row, news2_row,
                d_pat, i_pat, e_pat, news2_pat, row_eval_idx)
            evals[(setname, tag)] = (out, snap, temp)

    all_rows = []
    for setname, cfg in param_sets:
        recs = []
        for tag in ["6vital", "+ACVPU"]:
            out = evals[(setname, tag)][0]
            for (sysn, lvl, tgt), v in out.items():
                recs.append(dict(system=sysn, level=lvl, vitals=tag, target=tgt, auroc=round(v, 5)))
                all_rows.append(dict(experiment="06_supplemental_o2", param_set=setname, vitals=tag,
                    system=sysn, level=lvl, target=tgt, auroc=round(v, 5),
                    **{k: cfg[k] for k in ["alpha","beta","gamma","method","profile","excess_mode"]}))
        pd.DataFrame(recs).pivot_table(index=["vitals","level","system"], columns="target",
            values="auroc").to_csv(OUT_DIR / f"auroc_{setname}.csv")

    bars(evals[("event_optimal","6vital")][0], evals[("sherifs","6vital")][0],
         OUT_DIR / "summary_bars.png", "06_supplemental_o2  (patient-level AUROC, 6 vitals)")

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
            all_rows.append(dict(experiment="06_supplemental_o2", param_set="event_optimal", vitals=tag,
                system=s, level="tta_detected_frac", target="event", auroc=round(dd["detected_frac"],5),
                **{k: bestcfg[k] for k in ["alpha","beta","gamma","method","profile","excess_mode"]}))
            all_rows.append(dict(experiment="06_supplemental_o2", param_set="event_optimal", vitals=tag,
                system=s, level="tta_median_lead_h", target="event", auroc=round(dd["median_lead_h"],3),
                **{k: bestcfg[k] for k in ["alpha","beta","gamma","method","profile","excess_mode"]}))
    pd.DataFrame(ttarecs).to_csv(OUT_DIR / "time_to_alert.csv", index=False)
    tta_plot(tta6, tta7, OUT_DIR / "time_to_alert.png", "06_supplemental_o2 — time-to-alert")

    with open(OUT_DIR / "best_config.json", "w") as f:
        json.dump(bestcfg, f, indent=2)

    # Print key numbers
    print("\n=== Patient-level AUROC (event-optimal, 6 vitals) ===")
    for (sysn, lvl, tgt), v in evals[("event_optimal","6vital")][0].items():
        if lvl == "patient":
            print(f"  {sysn:12s} {tgt:6s}: {v:.4f}")
    print("\n=== Patient-level AUROC (event-optimal, +ACVPU) ===")
    for (sysn, lvl, tgt), v in evals[("event_optimal","+ACVPU")][0].items():
        if lvl == "patient":
            print(f"  {sysn:12s} {tgt:6s}: {v:.4f}")
    print("\n=== Time-to-alert (6 vitals, 20% non-event alert rate) ===")
    for s, d in tta6.items():
        print(f"  {s:12s}  detected {d['detected_frac']*100:.1f}%  lead {d['median_lead_h']:.1f}h")

    # Append to ALL_RESULTS.csv
    all_csv = REPO / "improved_results" / "ALL_RESULTS.csv"
    existing = pd.read_csv(all_csv) if all_csv.exists() else pd.DataFrame()
    existing = existing[existing["experiment"] != "06_supplemental_o2"] if len(existing) else existing
    pd.concat([existing, pd.DataFrame(all_rows)], ignore_index=True).to_csv(all_csv, index=False)

    print(f"\nSaved → {OUT_DIR}")
    print(f"Done in {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
