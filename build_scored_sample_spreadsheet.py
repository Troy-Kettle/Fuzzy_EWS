"""
Build a 500,000-row spreadsheet of scored observations from the main event
dataset (datasets/final_observations_with_targets.csv).

For a random sample of patients (whole admissions kept intact so the temporal
EWMA + trend calculations stay causally correct), this writes one row per
observation with:

  - the raw vitals (heart rate, systolic BP, respiratory rate, SpO2,
    temperature, inspired-oxygen category, ACVPU)
  - Chronic_Resp (0/1): NEWS-2 Scale 2 applies when this is 1. This dataset
    does not carry chronic-respiratory status itself — it is joined in from
    datasets/Annotated dataset_training_anonymised_V5_Troy (1).xlsx on
    (admission, day, obs time) where an exact match exists (a small minority
    of admissions overlap the two datasets); every other row defaults to 0
    (Scale 1). See the "chronic resp coverage" line printed at run time for
    how many rows actually got a real (matched) value.
  - NEWS-2, chronic-aware (Scale 1 unless Chronic_Resp=1, then Scale 2), with NO
    consciousness sub-score: ACVPU is out-of-band for every system here, so neither
    NEWS_2 nor the fuzzy totals get points for it (see below).
  - the snapshot fuzzy total and the temporal (EWMA + worsening-trend) fuzzy
    total, over the SIX scored vitals. ACVPU is not one of them and adds nothing
    to either total.
  - ACVPU_Deterioration_Flag: Yes/No — any reading other than Alert makes the row
    a positive flag of deterioration on its own, which is the whole of ACVPU's
    contribution — no system awards points for it, so the fuzzy scores and the
    NEWS_2 baseline are matched on inputs.

Temporal parameters: alpha=0.7 (fixed per request), beta/gamma taken from the
most recent row-level AUPRC-optimal grid search for this dataset
(results/current/grid_search_rowlevel_auprc/grid_results_rowlevel_auprc.csv,
best row by auprc_event: alpha=0.1, beta=0.5, gamma=0.9 — alpha overridden).

Run:  python3 build_scored_sample_spreadsheet.py
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "engine"))
import engine_scoring as es

DATA_PATH     = REPO / "datasets" / "final_observations_with_targets.csv"
CHRONIC_PATH  = REPO / "datasets" / "Annotated dataset_training_anonymised_V5_Troy (1).xlsx"
OUT_PATH      = REPO / "fuzzy_ews_scored_500k.xlsx"

N_ROWS      = 500_000
RANDOM_SEED = 42

# AUPRC-optimal temporal parameters for this dataset (row-level grid search,
# results/current/grid_search_rowlevel_auprc/grid_results_rowlevel_auprc.csv,
# best row by auprc_event). alpha is overridden to 0.7 per request.
ALPHA, BETA, GAMMA = 0.7, 0.5, 0.9

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

O2CAT_CONCERN = {"Low": 1.0, "Low-moderate": 1.5, "Moderate": 2.0,
                  "High": 2.5, "Very high": 3.0}


def score_resp(x):
    return np.select([x <= 8, x <= 11, x <= 20, x <= 24], [3, 1, 0, 2], default=3)


def score_temp(x):
    return np.select([x <= 35.0, x <= 36.0, x <= 38.0, x <= 39.0], [3, 1, 0, 1], default=2)


def score_bp(x):
    return np.select([x <= 90, x <= 100, x <= 110, x <= 219], [3, 2, 1, 0], default=3)


def score_hr(x):
    return np.select([x <= 40, x <= 50, x <= 90, x <= 110, x <= 130], [3, 1, 0, 1, 2], default=3)


def load_chronic_lookup():
    """Chronic-respiratory (0/1) keyed by (ADMISSION_ID, DAYS_SINCE_ADMISSION,
    OBS_TIME) from the annotated dataset — the only source of this field."""
    df = pd.read_excel(CHRONIC_PATH, sheet_name="Annotated training",
                       usecols=["ADMISSION_ID", "OBS_DAYS_SINCE_ADMISSION",
                                "OBS_TIME", "CHRONIC_RESP_OBS_SET"])
    df = df.drop_duplicates(subset=["ADMISSION_ID", "OBS_DAYS_SINCE_ADMISSION", "OBS_TIME"])
    return df.set_index(["ADMISSION_ID", "OBS_DAYS_SINCE_ADMISSION", "OBS_TIME"])["CHRONIC_RESP_OBS_SET"]


def load():
    print("Loading dataset…"); t0 = time.time()
    cols = ["ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
            "HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
            "INSPIRED_O2_TEXT", "INSP_O2_CAT", "AVPU_ACVPU", "TEMPERATURE", "COMPLETE_DATA"]
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
    df["O2_CONCERN"] = df["INSP_O2_CAT"].map(O2CAT_CONCERN).fillna(0.0).astype(np.float32)
    df["ACVPU_NUM"] = df["AVPU_ACVPU"].map(es.ACVPU_MAP).fillna(0.0).astype(np.float32)
    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"] * 1440.0
                       + obs.dt.hour.fillna(0) * 60.0
                       + obs.dt.minute.fillna(0)
                       + obs.dt.second.fillna(0) / 60.0).astype(np.float32)
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df):,} complete rows in {time.time()-t0:.0f}s, "
          f"{df['ANON_ADMISSION_ID'].nunique():,} patients")
    return df


def attach_chronic_resp(df):
    lookup = load_chronic_lookup()
    key = list(zip(df["ANON_ADMISSION_ID"].values, df["DAYS_SINCE_ADMISSION"].astype("int64").values,
                    df["OBS_TIME"].values))
    matched = pd.Series(key, index=df.index).map(lookup)
    n_matched = matched.notna().sum()
    df["CHRONIC_RESP"] = matched.fillna(0).astype(np.int8)
    print(f"  Chronic-resp coverage: {n_matched:,}/{len(df):,} rows matched to the annotated "
          f"dataset (this dataset does not carry the field itself); all other rows default to "
          f"0 = Scale 1.")
    return df


def sample_patients(df, n_rows, seed):
    """Keep whole admissions (so EWMA/trend stay causal), stop once n_rows is hit."""
    rng = np.random.default_rng(seed)
    ids = df["ANON_ADMISSION_ID"].unique()
    rng.shuffle(ids)
    counts = df.groupby("ANON_ADMISSION_ID").size()
    picked, total = [], 0
    for pid in ids:
        if total >= n_rows:
            break
        picked.append(pid)
        total += int(counts[pid])
    out = df[df["ANON_ADMISSION_ID"].isin(picked)].copy()
    out.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out.iloc[:n_rows].copy()


def main():
    t_total = time.time()
    df = load()
    df = sample_patients(df, N_ROWS, RANDOM_SEED)
    df = attach_chronic_resp(df)
    print(f"Sampled {len(df):,} rows across {df['ANON_ADMISSION_ID'].nunique():,} patients")

    print("Building fuzzy LUTs (sharper SBP)…")
    luts = {v: es.build_lut(v) for v in VITALS}
    luts["blood_pressure"] = es.build_lut("blood_pressure", sharper_sbp=True)

    pv = es.apply_luts(df, luts, VITALS)
    pv["inspired_oxygen"] = df["O2_CONCERN"].values.astype(np.float32)

    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
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

    # NEWS-2 "on supplemental oxygen" comes from the clinical category, NOT from
    # INSPIRED_O2_TEXT: that field mixes L/min flow with FiO2%, so the .clip(21, 100)
    # in load() collapses every flow reading to 21 ("room air"). Testing it with > 21
    # therefore missed ~1.31M of the 1.52M on-oxygen rows in the source data, losing
    # both the +2 supplemental-O2 point and the on-oxygen Scale 2 SpO2 thresholds.
    on_oxygen = df["O2_CONCERN"].values > 0
    news2 = (score_resp(df["RESP_RATE"].values)
             + es.news2_spo2_score(df["SATS_SPO2"].values, df["CHRONIC_RESP"].values, on_oxygen)
             + score_temp(df["TEMPERATURE"].values)
             + score_bp(df["SYSTOLIC_BP"].values)
             + score_hr(df["HEART_RATE"].values)
             + np.where(on_oxygen, 2, 0)
             + es.news2_consciousness_score(acvpu_raw))

    deteriorating = es.acvpu_deterioration_flag(acvpu_raw)

    out = pd.DataFrame({
        "Patient_ID":                 df["ANON_ADMISSION_ID"].values,
        "Day_Since_Admission":        df["DAYS_SINCE_ADMISSION"].values,
        "Obs_Time":                   df["OBS_TIME"].values,
        "Heart_Rate":                 df["HEART_RATE"].values,
        "Systolic_BP":                df["SYSTOLIC_BP"].values,
        "Respiratory_Rate":           df["RESP_RATE"].values,
        "SpO2":                       df["SATS_SPO2"].values,
        "Temperature":                df["TEMPERATURE"].values,
        "Inspired_O2_Category":       df["INSP_O2_CAT"].fillna("Room air").values,
        "ACVPU":                      df["AVPU_ACVPU"].fillna("Alert").values,
        "Chronic_Resp":               df["CHRONIC_RESP"].values,
        "ACVPU_Deterioration_Flag":   np.where(deteriorating, "Yes", "No"),
        "NEWS_2":                     news2,
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
