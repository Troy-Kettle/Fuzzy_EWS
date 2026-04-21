#!/usr/bin/env python3
"""
Standalone AUROC script for Snapshot Fuzzy EWS on 1,000,000 rows.

No imports from other project scripts.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(over="ignore", invalid="ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "20250630_final_observations-sorted_V7_training_balanced.csv"
SIGMOID_DIR = SCRIPT_DIR / "generated_membership_data" / "sigmoid"

# Sampling
SAMPLE_SIZE = 1000000
RANDOM_SEED = 42

# Vitals and membership files
VITALS = [
    "heart_rate",
    "blood_pressure",
    "temperature",
    "respiratory_rate",
    "oxygen_saturation",
    "inspired_oxygen",
]

VITAL_COL = {
    "heart_rate": "HEART_RATE",
    "blood_pressure": "SYSTOLIC_BP",
    "temperature": "TEMPERATURE",
    "respiratory_rate": "RESP_RATE",
    "oxygen_saturation": "SATS_SPO2",
    "inspired_oxygen": "INSPIRED_O2_TEXT",
}

MF_FILE = {
    "heart_rate": "heart_rate_membership_functions.csv",
    "blood_pressure": "systolic_blood_pressure_membership_functions.csv",
    "temperature": "temperature_membership_functions.csv",
    "respiratory_rate": "respiratory_rate_membership_functions.csv",
    "oxygen_saturation": "oxygen_saturation_membership_functions.csv",
    "inspired_oxygen": "inspired_oxygen_concentration_membership_functions.csv",
}

VITAL_TYPE = {
    "heart_rate": "7var",
    "blood_pressure": "7var",
    "temperature": "7var",
    "respiratory_rate": "7var",
    "oxygen_saturation": "3var_down",
    "inspired_oxygen": "3var_up",
}

LABELS_7 = [
    "Below normal - severe concern",
    "Below normal - moderate concern",
    "Below normal - mild concern",
    "No concern",
    "Above normal - mild concern",
    "Above normal - moderate concern",
    "Above normal - severe concern",
]

LABELS_3_DOWN = [
    "Below normal - severe concern",
    "Below normal - moderate concern",
    "Below normal - mild concern",
    "No concern",
]

LABELS_3_UP = [
    "No concern",
    "Above normal - mild concern",
    "Above normal - moderate concern",
    "Above normal - severe concern",
]

OUTPUT_MF_DEFS = {
    "No concern": (-0.5, 0, 0, 0.75),
    "Mild concern": (0.25, 1, 1, 1.75),
    "Moderate concern": (1.25, 2, 2, 2.75),
    "Severe concern": (2.25, 3, 3, 3.5),
}


def _trapezoid(x, a, b, c, d):
    if b <= x <= c:
        return 1.0
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def _interp_lookup(fs, inp):
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


def _load_mf(vital):
    df = pd.read_csv(SIGMOID_DIR / MF_FILE[vital])
    keys = df["Value"].values
    vtype = VITAL_TYPE[vital]
    labels = {"7var": LABELS_7, "3var_down": LABELS_3_DOWN, "3var_up": LABELS_3_UP}[vtype]
    fs_list = [dict(zip(keys, df[label].values)) for label in labels]
    return labels, fs_list


def _concern_from_memberships(memberships):
    concern = {"No concern": 0.0, "Mild concern": 0.0, "Moderate concern": 0.0, "Severe concern": 0.0}
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


_OUTPUT_X = np.arange(0, 3.01, 0.01)
_OUTPUT_GRID = {
    name: np.array([_trapezoid(x, *params) for x in _OUTPUT_X])
    for name, params in OUTPUT_MF_DEFS.items()
}


def _defuzz_centroid(concern):
    min_firing = 0.05
    filtered = {k: (v if v >= min_firing else 0.0) for k, v in concern.items()}

    if filtered.get("No concern", 0.0) > 0 and all(
        level == "No concern" or firing == 0.0 for level, firing in filtered.items()
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


def load_data():
    print("Loading dataset ...")
    t0 = time.time()
    cols_needed = [
        "ANON_ADMISSION_ID",
        "OBS_TIME",
        "DAYS_SINCE_ADMISSION",
        "REVIEW_WITHIN_4HOURS",
        "HEART_RATE",
        "SYSTOLIC_BP",
        "RESP_RATE",
        "SATS_SPO2",
        "INSPIRED_O2_TEXT",
        "TEMPERATURE",
        "COMPLETE_DATA",
        "NEWS-2",
    ]
    df = pd.read_csv(DATA_PATH, usecols=cols_needed, low_memory=False)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    numeric_cols = [
        "HEART_RATE",
        "SYSTOLIC_BP",
        "RESP_RATE",
        "SATS_SPO2",
        "TEMPERATURE",
        "COMPLETE_DATA",
        "REVIEW_WITHIN_4HOURS",
        "DAYS_SINCE_ADMISSION",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["COMPLETE_DATA"] = df["COMPLETE_DATA"].fillna(0)
    before = len(df)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    print(f"  Filtered to COMPLETE_DATA=1: {len(df):,} rows (dropped {before-len(df):,})")

    vital_cols = ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2", "TEMPERATURE"]
    before2 = len(df)
    df.dropna(subset=vital_cols + ["REVIEW_WITHIN_4HOURS", "ANON_ADMISSION_ID"], inplace=True)
    if len(df) < before2:
        print(f"  Dropped {before2-len(df):,} rows with NaN vitals after coercion")

    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df["REVIEW_WITHIN_4HOURS"] = df["REVIEW_WITHIN_4HOURS"].astype("int8")
    for col in vital_cols:
        df[col] = df[col].astype("float32")

    df["INSPIRED_O2_TEXT"] = pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce").fillna(21.0)
    df["INSPIRED_O2_TEXT"] = df["INSPIRED_O2_TEXT"].clip(lower=21.0, upper=100.0).astype("float32")
    df["NEWS-2"] = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0).astype("float32")

    obs_time = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    hours = obs_time.dt.hour.fillna(0).astype("float32")
    minutes = obs_time.dt.minute.fillna(0).astype("float32")
    seconds = obs_time.dt.second.fillna(0).astype("float32")
    df["t_minutes"] = (
        df["DAYS_SINCE_ADMISSION"].astype("float32") * 1440.0
        + hours * 60.0
        + minutes
        + seconds / 60.0
    )

    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    label = df["REVIEW_WITHIN_4HOURS"].values
    print(f"  Positive labels: {label.sum():,} / {len(label):,} ({100*label.mean():.2f}%)")
    return df


def compute_snapshot_scores_row_by_row(df):
    print("\nBuilding membership functions for row-wise scoring ...")
    mf_by_vital = {}
    for vital in VITALS:
        labels, fs_list = _load_mf(vital)
        mf_by_vital[vital] = (labels, fs_list)

    n = len(df)
    scores = np.zeros(n, dtype=np.float32)
    progress_every = 50_000
    print(f"Scoring {n:,} rows one-by-one ...")
    t0 = time.time()

    for i, row in enumerate(df.itertuples(index=False), start=1):
        total = 0.0
        for vital in VITALS:
            labels, fs_list = mf_by_vital[vital]
            value = float(getattr(row, VITAL_COL[vital]))
            memberships = {label: _interp_lookup(fs, value) for label, fs in zip(labels, fs_list)}
            concern = _concern_from_memberships(memberships)
            total += _defuzz_centroid(concern)

        scores[i - 1] = total
        if i % progress_every == 0 or i == n:
            elapsed = time.time() - t0
            rps = i / elapsed if elapsed > 0 else 0.0
            print(f"  {i:,}/{n:,} rows ({100*i/n:.1f}%)  ~{rps:,.0f} rows/s")

    return scores


def main():
    t0 = time.time()
    print("=" * 68)
    print(f"Standalone Snapshot Fuzzy EWS AUROC on {SAMPLE_SIZE:,} rows")
    print("=" * 68)

    df = load_data()
    total_rows = len(df)

    if total_rows > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).copy()
        print(f"Sampled {len(df):,} rows from {total_rows:,} cleaned rows")
    else:
        print(f"Dataset has {total_rows:,} rows; using all available rows")

    label = df["REVIEW_WITHIN_4HOURS"].values.astype(np.int8)
    print(f"Positive labels in sample: {label.sum():,} / {len(label):,} ({100*label.mean():.2f}%)")

    snapshot_total = compute_snapshot_scores_row_by_row(df)

    valid = np.isfinite(snapshot_total)
    snap_auroc = roc_auc_score(label[valid], snapshot_total[valid])
    print(f"\nSnapshot Fuzzy EWS AUROC = {snap_auroc:.6f}")

    news2 = df["NEWS-2"].values.astype(np.float32)
    valid_news = np.isfinite(news2)
    news_auroc = roc_auc_score(label[valid_news], news2[valid_news])
    print(f"NEWS-2 AUROC (same sample) = {news_auroc:.6f}")
    print(f"Delta (Snapshot - NEWS-2)  = {snap_auroc - news_auroc:+.6f}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
