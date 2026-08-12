"""TRAJECTORY-LEVEL validation — does temporal scoring carry signal AUROC can see?

Motivation (see results/trajectory/README and MODEL_COMPARISON.md "Methods caveat"):
the headline snapshot-vs-temporal AUROCs can end up close because both scores are
often reduced to a per-patient PEAK, which is structurally >= snapshot (EWMA + sigmoid
worsening-trend formula, matching app/streamlit_app.py — see engine_scoring.py) — so
peak-AUROC, being rank-only, may not distinguish them well. This script checks that
WITHOUT touching results/current/:

  * trajectory patient aggregates (not just peak)   — keep the deterioration path (#3)
  * coupling diagnostic + DeLong significance       — measure the (non)difference (#1, #7)
  * stay-length-stratified AUROC                    — expose short-stay inertness (#6)

NOTE: this previously also swept boost_mode (clip/headroom/uncapped), an excess-only
channel, and an EWMA-seed comparison — all specific to the old excess-EWMA formula
(`raw + β·max(0, raw−EWMA)`). Those concepts don't exist under the EWMA + sigmoid-trend
formula (the adjusted score is structurally >= raw with no [0,3] clip discontinuity to
work around), so they were removed rather than ported.

Outputs:
  results/trajectory/auroc_variants.csv      AUROC for every (system × aggregate × target)
  results/trajectory/coupling.csv            Spearman / %identical / %same-peak-row
  results/trajectory/delong.csv              snapshot-peak vs temporal variant, Δ + p-value
  results/trajectory/stratified_auroc.csv    AUROC by stay-length bin
"""
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "engine"))
import engine_scoring as es
import diagnostics as dg
import stats as st

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

DATA_PATH = REPO / "datasets" / "final_observations_with_targets.csv"
TRAJ_DIR  = REPO / "results" / "trajectory"
GRID_RES  = REPO / "results" / "current" / "grid_search" / "grid_results.csv"

RANDOM_SEED = 42
NE_PATIENTS = 25_000          # all event patients + this many controls (patient-level)
O2CAT = {"Low": 1.0, "Low-moderate": 1.5, "Moderate": 2.0, "High": 2.5, "Very high": 3.0}
TEMPORAL_VITALS = es.TEMPORAL_VITALS_DEFAULT   # canonical set (incl. inspired_oxygen);
# defined once in engine_scoring so the app and the pipeline cannot drift apart again.
TARGETS = {"death": "DEATH_WITHIN_24H", "icu": "ICU_WITHIN_24H", "event": "EVENT_FLAG"}


def pick_event_optimal():
    grid = pd.read_csv(GRID_RES)
    best = grid.loc[grid["event"].idxmax()]
    return float(best["alpha"]), float(best["beta"]), float(best["gamma"])


def load():
    print("Loading dataset…"); t0 = time.time()
    cols = ["ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
            "HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
            "INSPIRED_O2_TEXT", "INSP_O2_CAT", "AVPU_ACVPU", "TEMPERATURE",
            "COMPLETE_DATA", "NEWS-2", "DEATH_WITHIN_24H", "ICU_WITHIN_24H", "EVENT_FLAG"]
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    for c in ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
              "TEMPERATURE", "DAYS_SINCE_ADMISSION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2", "TEMPERATURE"],
              inplace=True)
    df["INSPIRED_O2_TEXT"] = (pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce")
                              .fillna(21.0).clip(21, 100))
    df["NEWS-2"] = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0)
    df["ACVPU_NUM"] = df["AVPU_ACVPU"].map(es.ACVPU_MAP).fillna(0.0)
    df["O2_CONCERN"] = df["INSP_O2_CAT"].map(O2CAT).fillna(0.0).astype(np.float32)
    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"] * 1440.0 + obs.dt.hour.fillna(0) * 60.0
                       + obs.dt.minute.fillna(0) + obs.dt.second.fillna(0) / 60.0).astype(np.float32)
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")

    # sample: all event patients + NE_PATIENTS controls (patient-level evaluation)
    rng = np.random.default_rng(RANDOM_SEED)
    ev_ids = set(df.loc[df["EVENT_FLAG"] == 1, "ANON_ADMISSION_ID"].unique())
    ne_ids = list(set(df["ANON_ADMISSION_ID"].unique()) - ev_ids)
    ne = rng.choice(ne_ids, size=min(NE_PATIENTS, len(ne_ids)), replace=False)
    df = df[df["ANON_ADMISSION_ID"].isin(ev_ids | set(ne))].copy()
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df):,} rows, {df['ANON_ADMISSION_ID'].nunique():,} patients "
          f"in {time.time()-t0:.0f}s")
    return df


def patient_labels(df, gs):
    return {k: np.maximum.reduceat(df[col].values, gs) for k, col in TARGETS.items()}


def auroc_targets(scores_pat, labels):
    """{system: patient_score} → list of {system,target,auroc} rows."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for sysname, sc in scores_pat.items():
        for tgt, y in labels.items():
            s = np.asarray(sc, np.float64); ok = np.isfinite(s)
            au = (float(roc_auc_score(y[ok], s[ok]))
                  if 0 < y[ok].sum() < ok.sum() else float("nan"))
            rows.append({"system": sysname, "target": tgt,
                         "auroc": round(au, 5) if au == au else au})
    return rows


def main():
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    alpha, beta, gamma = pick_event_optimal()
    print(f"Event-optimal params: α={alpha} β={beta} γ={gamma}")

    df = load()
    # ACVPU is not a scored vital (flag only) — this stays the 6-vital set
    vitals = es.VITALS_BASE
    luts = {v: es.build_lut(v) for v in vitals}
    pv = es.apply_luts(df, luts, vitals)
    pv["inspired_oxygen"] = df["O2_CONCERN"].values.astype(np.float32)

    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    times = df["t_minutes"].values.astype(np.float64)
    labels = patient_labels(df, gs)
    stay_len = (ge - gs).astype(np.int64)
    print(f"  event pos={int(labels['event'].sum()):,} "
          f"neg={int((labels['event']==0).sum()):,}")

    alphas = {v: alpha for v in vitals}; refs = {v: es.EWMA_REF_DEFAULT for v in vitals}
    print("Computing EWMA + trend slopes…"); t1 = time.time()
    ewma = es.compute_ewma(times, pv, gs, ge, vitals, alphas, refs)
    slopes = es.compute_slopes(times, pv, gs, ge, vitals)
    print(f"  done in {time.time()-t1:.0f}s")

    # ── row-level score variants ──────────────────────────────────────────────
    snap_row = es.snapshot_score(pv, vitals).astype(np.float32)
    temp_row = es.temporal_score(pv, ewma, slopes, vitals, beta, gamma,
                                 temporal_vitals=TEMPORAL_VITALS)

    row_variants = {"snapshot": snap_row, "temporal": temp_row}

    # ── patient-level: every variant × every aggregate ────────────────────────
    AGG = ["peak", "mean", "area", "pre_peak_slope", "time_above", "score_at_lead"]
    THR = float(np.quantile(snap_row, 0.90))   # threshold for time_above
    print("Aggregating to patient level (this loops per patient)…"); t2 = time.time()
    pat = {}   # (variant, agg) -> patient vector
    for name, rowsc in row_variants.items():
        for agg in AGG:
            pat[(name, agg)] = es.patient_aggregate(rowsc, gs, ge, agg, times=times,
                                                    threshold=THR)
    print(f"  aggregation done in {time.time()-t2:.0f}s")

    # AUROC variants table
    auroc_rows = []
    for (name, agg), sc in pat.items():
        for r in auroc_targets({name: sc}, labels):
            auroc_rows.append({"aggregate": agg, **r})
    av = pd.DataFrame(auroc_rows)
    av.to_csv(TRAJ_DIR / "auroc_variants.csv", index=False)
    print(f"\nSaved auroc_variants.csv ({len(av)} rows)")

    snap_peak = pat[("snapshot", "peak")]

    # ── coupling diagnostic (issue #1, #7) ────────────────────────────────────
    pairs = {}
    for agg in AGG:
        pairs[f"snapshot_vs_temporal[{agg}]"] = (
            pat[("snapshot", agg)], pat[("temporal", agg)])
    ctab = dg.coupling_table(pairs, snap_row=snap_row,
                             temp_rows={"snapshot_vs_temporal[peak]": temp_row},
                             gs=gs, ge=ge)
    ctab.to_csv(TRAJ_DIR / "coupling.csv", index=False)
    print(f"Saved coupling.csv ({len(ctab)} rows)")

    # ── DeLong significance vs snapshot-peak (issue #7) ───────────────────────
    delong_rows = []
    for tgt, y in labels.items():
        for (name, agg), sc in pat.items():
            if name == "snapshot" and agg == "peak":
                continue
            try:
                r = st.delong_roc_test(y, snap_peak, sc)
            except ValueError:
                continue
            delong_rows.append({
                "target": tgt, "variant": name, "aggregate": agg,
                "auroc_snapshot_peak": round(r["auc_a"], 5),
                "auroc_variant": round(r["auc_b"], 5),
                "delta_variant_minus_snap": round(r["auc_b"] - r["auc_a"], 5),
                "p_value": round(r["p"], 4),
                "significant_0p05": bool(r["p"] < 0.05)})
    pd.DataFrame(delong_rows).to_csv(TRAJ_DIR / "delong.csv", index=False)
    print(f"Saved delong.csv ({len(delong_rows)} rows)")

    # ── stay-length stratified AUROC (issue #6) ───────────────────────────────
    strat = dg.stratified_auroc(
        labels["event"],
        {"snapshot_peak": snap_peak, "temporal_peak": pat[("temporal", "peak")]},
        stay_len)
    strat.to_csv(TRAJ_DIR / "stratified_auroc.csv", index=False)
    print(f"Saved stratified_auroc.csv ({len(strat)} rows)")

    # ── console headline ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Event-target AUROC by variant (peak aggregate) vs snapshot peak:")
    ev = av[(av["target"] == "event") & (av["aggregate"] == "peak")]
    print(ev[["system", "auroc"]].to_string(index=False))
    print("\nCoupling Spearman(snapshot, temporal) by aggregate:")
    cc = ctab[ctab["comparison"].str.startswith("snapshot_vs_temporal")]
    print(cc[["comparison", "spearman", "pct_identical"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
