"""
Build a scored-observations spreadsheet from the annotated event dataset
(datasets/Annotated dataset_training_anonymised_V5_Troy (1).xlsx, sheet
"Annotated training") — the companion run to build_scored_sample_spreadsheet.py
on the main event dataset, using the same methodology.

This dataset has only 55,271 complete-eligible rows (829 admissions), under
the 500,000-row target, so every row is used (no sampling) — whole admissions
are naturally kept intact.

Per observation:
  - the raw vitals (heart rate, systolic BP, respiratory rate, SpO2,
    temperature, inspired oxygen — as recorded, see below — ACVPU)
  - Chronic_Resp (0/1): this dataset carries it natively as
    CHRONIC_RESP_OBS_SET (observation-level, not just per-admission — 72/829
    admissions have it change mid-stay), used directly.
  - NEWS-2, chronic-aware (Scale 1 unless Chronic_Resp=1, then Scale 2), with NO
    consciousness sub-score: ACVPU is out-of-band for every system here, so neither
    NEWS_2 nor the fuzzy totals get points for it (see below). The
    source file's own NEWS_2_SCORE is also carried through as
    NEWS_2_Score_Source for comparison. NOTE that source column is precomputed
    dataset content and almost certainly DOES include consciousness points, so it
    is not matched with the NEWS_2 column computed here — use NEWS_2 for any
    like-for-like comparison against the fuzzy scores.
  - the snapshot fuzzy total and the temporal (EWMA + worsening-trend) fuzzy
    total, over the SIX scored vitals. ACVPU is not one of them and adds nothing
    to either total.
  - ACVPU_Deterioration_Flag: Yes/No — any reading other than Alert makes the row
    a positive flag of deterioration on its own, which is the whole of ACVPU's
    contribution — no system awards points for it, so the fuzzy scores and the
    NEWS_2 baseline are matched on inputs.

Inspired oxygen: INSPIRED_O2 is recorded in mixed units (INSPIRED_O2_UNITS is
either "%" FiO2 or "litres" of supplementary flow). The two units are kept
SEPARATE and reported in their own columns — Inspired_O2_FiO2_Pct and
Supplementary_O2_L_min, each blank on rows recorded in the other unit — and each
row is scored through the membership function for its own unit ("%" via
inspired_oxygen_concentration_membership_functions.csv, "litres" via
supplementary_oxygen_lmin_membership_functions.csv). See
engine_scoring.split_inspired_oxygen / inspired_oxygen_concern.

An earlier version of this script instead converted every flow reading to a
pseudo-FiO2 (21 + 4*L/min) and reported only that single column. That fabricated
concentrations the source never recorded (0.5 L/min appeared as "23%", 1 L/min as
"25%", …) and is not used: no interconversion is performed.

Temporal parameters: alpha=0.7 (fixed per request), beta/gamma taken from the
most recent row-level AUPRC-optimal grid search for this dataset
(results/annotated/grid_search_rowlevel_auprc/grid_results_rowlevel_auprc.csv,
best row by auprc_event: alpha=0.8, beta=0.0, gamma=0.9 — alpha overridden).

Run:  python3 build_scored_sample_spreadsheet_annotated.py
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "engine"))
import engine_scoring as es

DATA_PATH = REPO / "datasets" / "Annotated dataset_training_anonymised_V5_Troy (1).xlsx"
OUT_PATH  = REPO / "fuzzy_ews_scored_annotated.xlsx"

N_ROWS_CAP  = 500_000   # dataset is smaller than this, so every row is used
RANDOM_SEED = 42

# AUPRC-optimal temporal parameters for this dataset (row-level grid search,
# results/annotated/grid_search_rowlevel_auprc/grid_results_rowlevel_auprc.csv,
# best row by auprc_event). alpha is overridden to 0.7 per request.
ALPHA, BETA, GAMMA = 0.7, 0.0, 0.9

VITALS = es.VITALS_BASE   # 6 scored vitals. ACVPU is NOT one of them: any non-Alert
# reading flags the whole row as deterioration and adds nothing to the score.
TEMPORAL_VITALS = es.TEMPORAL_VITALS_DEFAULT   # canonical set (incl. inspired_oxygen);
# defined once in engine_scoring so the app and the pipeline cannot drift apart again.

# Per-vital temporal-score output columns (label used in the "Temporal_Score_*"
# column names); covers all VITALS, not just TEMPORAL_VITALS, since
# temporal_score() still returns a (snapshot) component for ACVPU, the one
# vital excluded from the EWMA/trend adjustment.
TEMPORAL_VITAL_LABELS = {
    "heart_rate":        "Heart_Rate",
    "blood_pressure":    "Systolic_BP",
    "temperature":       "Temperature",
    "respiratory_rate":  "Respiratory_Rate",
    "oxygen_saturation": "SpO2",
    "inspired_oxygen":   "Inspired_O2",
}

# This dataset spells the confusion category differently from the main
# dataset's AVPU_ACVPU ("Newly confused / agitated"); alias it onto the
# canonical engine map rather than duplicating the map.
ACVPU_ALIASES = {"Confused": "Newly confused / agitated"}


def score_resp(x):
    return np.select([x <= 8, x <= 11, x <= 20, x <= 24], [3, 1, 0, 2], default=3)


def score_temp(x):
    return np.select([x <= 35.0, x <= 36.0, x <= 38.0, x <= 39.0], [3, 1, 0, 1], default=2)


def score_bp(x):
    return np.select([x <= 90, x <= 100, x <= 110, x <= 219], [3, 2, 1, 0], default=3)


def score_hr(x):
    return np.select([x <= 40, x <= 50, x <= 90, x <= 110, x <= 130], [3, 1, 0, 1, 2], default=3)


def load():
    print("Loading annotated dataset…"); t0 = time.time()
    df = pd.read_excel(DATA_PATH, sheet_name="Annotated training")
    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    for c in ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "O2_SATS", "TEMPERATURE",
              "OBS_DAYS_SINCE_ADMISSION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "O2_SATS", "TEMPERATURE"],
              inplace=True)

    # Kept in their own units, no interconversion (see module docstring).
    fio2, flow = es.split_inspired_oxygen(df["INSPIRED_O2"], df["INSPIRED_O2_UNITS"])
    df["FIO2_PCT"], df["SUPP_O2_LMIN"] = fio2, flow
    df["ACVPU_CLEAN"] = df["ACVPU"].replace(ACVPU_ALIASES).fillna("Alert")
    df["ACVPU_NUM"] = df["ACVPU_CLEAN"].map(es.ACVPU_MAP).fillna(0.0).astype(np.float32)
    df["CHRONIC_RESP"] = pd.to_numeric(df["CHRONIC_RESP_OBS_SET"], errors="coerce").fillna(0).astype(np.int8)

    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["OBS_DAYS_SINCE_ADMISSION"] * 1440.0
                       + obs.dt.hour.fillna(0) * 60.0
                       + obs.dt.minute.fillna(0)
                       + obs.dt.second.fillna(0) / 60.0).astype(np.float32)
    df["ADMISSION_ID"] = df["ADMISSION_ID"].astype("int32")
    df.sort_values(["ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df):,} complete rows in {time.time()-t0:.0f}s, "
          f"{df['ADMISSION_ID'].nunique():,} admissions "
          f"(no sampling needed: below the {N_ROWS_CAP:,}-row target)")
    return df


def main():
    t_total = time.time()
    df = load()

    print("Building fuzzy LUTs (five-set SBP, + supplementary-O2 L/min)…")
    luts = {v: es.build_lut(v) for v in VITALS + [es.SUPP_O2_LMIN]}

    lut_df = df.rename(columns={"O2_SATS": "SATS_SPO2"})
    lut_df["ACVPU_NUM"] = df["ACVPU_NUM"].values
    # inspired oxygen is scored per row in its own recorded unit, so it is handled
    # separately rather than through the single-column apply_luts path
    pv = es.apply_luts(lut_df, luts, [v for v in VITALS if v != "inspired_oxygen"])
    pv["inspired_oxygen"] = es.inspired_oxygen_concern(
        df["FIO2_PCT"].values, df["SUPP_O2_LMIN"].values,
        luts["inspired_oxygen"], luts[es.SUPP_O2_LMIN])

    gs, ge = es.group_boundaries(df["ADMISSION_ID"].values)
    t_arr = df["t_minutes"].values.astype(np.float64)
    acvpu_raw = df["ACVPU_NUM"].values

    print("Computing EWMA + trend slopes…")
    ewma = es.compute_ewma(
        t_arr, pv, gs, ge, VITALS,
        {v: ALPHA for v in VITALS},
        {v: es.EWMA_REF_DEFAULT for v in VITALS},
    )
    slopes = es.compute_slopes(t_arr, pv, gs, ge, VITALS)

    snapshot_total = es.snapshot_score(pv, VITALS, method="additive", gamma=1.0)
    temporal_total, temporal_components = es.temporal_score(
        pv, ewma, slopes, VITALS, BETA, GAMMA,
        method="additive", temporal_vitals=TEMPORAL_VITALS, gs=gs, ge=ge,
        return_components=True,
    )

    on_oxygen = es.on_supplemental_oxygen(df["FIO2_PCT"].values, df["SUPP_O2_LMIN"].values)
    news2 = (score_resp(df["RESP_RATE"].values)
             + es.news2_spo2_score(df["O2_SATS"].values, df["CHRONIC_RESP"].values, on_oxygen)
             + score_temp(df["TEMPERATURE"].values)
             + score_bp(df["SYSTOLIC_BP"].values)
             + score_hr(df["HEART_RATE"].values)
             + np.where(on_oxygen, 2, 0)
             + es.news2_consciousness_score(acvpu_raw))

    deteriorating = es.acvpu_deterioration_flag(acvpu_raw)

    out = pd.DataFrame({
        "Patient_ID":                 df["ADMISSION_ID"].values,
        "Day_Since_Admission":        df["OBS_DAYS_SINCE_ADMISSION"].values,
        "Obs_Time":                   df["OBS_TIME"].values,
        "Heart_Rate":                 df["HEART_RATE"].values,
        "Systolic_BP":                df["SYSTOLIC_BP"].values,
        "Respiratory_Rate":           df["RESP_RATE"].values,
        "SpO2":                       df["O2_SATS"].values,
        "Temperature":                df["TEMPERATURE"].values,
        # both as recorded; each blank on rows recorded in the other unit
        "Inspired_O2_FiO2_Pct":       df["FIO2_PCT"].values,
        "Supplementary_O2_L_min":     df["SUPP_O2_LMIN"].values,
        "ACVPU":                      df["ACVPU_CLEAN"].values,
        "Chronic_Resp":               df["CHRONIC_RESP"].values,
        "ACVPU_Deterioration_Flag":   np.where(deteriorating, "Yes", "No"),
        "NEWS_2":                     news2,
        "NEWS_2_Score_Source":        df["NEWS_2_SCORE"].values,
        "Snapshot_Fuzzy_Score":       snapshot_total,
        "Temporal_Fuzzy_Score":       temporal_total,
    })
    for v, label in TEMPORAL_VITAL_LABELS.items():
        out[f"Temporal_Score_{label}"] = temporal_components[v]

    print(f"Writing {len(out):,} rows → {OUT_PATH.name}")
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Scored Observations", index=False)
    print(f"Done in {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
