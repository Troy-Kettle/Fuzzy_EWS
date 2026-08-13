"""
Build target dataset by merging observations with admission-level flags.

Logic:
- Add DIED_FLAG, ICU_FLAG, DISCHARGING_SPECIALTY from admission summary
- For DIED_FLAG patients: mark obs within 24h of the last observation as positive
- For ICU_FLAG patients: detect major gaps in obs (gap > ICU_GAP_HOURS) to infer ICU
  admission times; mark obs within 24h before each gap as positive. If no gap found,
  treat last observation as ICU admission.
- EVENT_FLAG = 1 if either death or ICU within-24h condition is met

ICU_GAP_HOURS rationale: ICU patients average ~70 obs and a median inter-obs gap
of 4.2h, so overnight gaps routinely exceed 6-8h. A 24h threshold is the minimum
that reliably distinguishes an ICU transfer from a normal overnight gap.
"""

import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --- Config ---
OBS_PATH = REPO / "datasets" / "20250630_final_observations-sorted_V7_training.csv"
ADMISSION_PATH = REPO / "datasets" / "Admission summary all specialties.xlsx"
OUTPUT_PATH = REPO / "datasets" / "final_observations_with_targets.csv"
ICU_GAP_HOURS = 24.0   # gap threshold to infer ICU admission (hours)


def main():
    print("Loading admission summary...")
    adm = pd.read_excel(ADMISSION_PATH, usecols=[
        "ANON_ADMISSION_ID", "DIED_FLAG", "ICU_FLAG", "DISCHARGING_SPECIALTY"
    ])
    adm["ICU_FLAG"] = adm["ICU_FLAG"].fillna(0).astype(int)
    adm["DIED_FLAG"] = adm["DIED_FLAG"].astype(int)

    print(f"  Admissions loaded: {len(adm):,}")
    print(f"  DIED_FLAG=1: {adm['DIED_FLAG'].sum():,}")
    print(f"  ICU_FLAG=1:  {adm['ICU_FLAG'].sum():,}")

    print("\nLoading observations (this may take a moment)...")
    df = pd.read_csv(OBS_PATH, low_memory=False)
    print(f"  Rows: {len(df):,}")

    # Merge admission-level flags
    df = df.merge(
        adm[["ANON_ADMISSION_ID", "DIED_FLAG", "ICU_FLAG", "DISCHARGING_SPECIALTY"]],
        on="ANON_ADMISSION_ID",
        how="left"
    )
    df["ICU_FLAG"] = df["ICU_FLAG"].fillna(0).astype(int)
    df["DIED_FLAG"] = df["DIED_FLAG"].fillna(0).astype(int)

    # --- Build absolute timestamp (days + time) as fractional hours from admission ---
    print("\nBuilding absolute timestamps...")
    df["_abs_hours"] = (
        df["DAYS_SINCE_ADMISSION"] * 24
        + pd.to_timedelta(df["OBS_TIME"]).dt.total_seconds() / 3600
    )

    # --- Death within 24h flag ---
    print("Computing DEATH_WITHIN_24H flag...")
    # For each admission with death, find the last obs time
    death_ids = df.loc[df["DIED_FLAG"] == 1, "ANON_ADMISSION_ID"].unique()
    print(f"  Admissions with death: {len(death_ids):,}")

    last_obs = (
        df[df["ANON_ADMISSION_ID"].isin(death_ids)]
        .groupby("ANON_ADMISSION_ID")["_abs_hours"]
        .max()
        .rename("_death_time")
    )

    df = df.merge(last_obs, on="ANON_ADMISSION_ID", how="left")
    df["DEATH_WITHIN_24H"] = (
        (df["DIED_FLAG"] == 1) &
        (df["_death_time"] - df["_abs_hours"] <= 24) &
        (df["_death_time"] - df["_abs_hours"] >= 0)
    ).astype(int)
    df.drop(columns=["_death_time"], inplace=True)

    # --- ICU within 24h flag ---
    print("Computing ICU_WITHIN_24H flag...")
    icu_ids = df.loc[df["ICU_FLAG"] == 1, "ANON_ADMISSION_ID"].unique()
    print(f"  Admissions with ICU: {len(icu_ids):,}")

    icu_df = df[df["ANON_ADMISSION_ID"].isin(icu_ids)][
        ["ANON_ADMISSION_ID", "_abs_hours"]
    ].copy().sort_values(["ANON_ADMISSION_ID", "_abs_hours"])

    # Detect ICU event times per admission
    def get_icu_event_times(grp):
        times = grp["_abs_hours"].values
        if len(times) < 2:
            return [times[-1]]
        gaps = np.diff(times)
        gap_positions = np.where(gaps > ICU_GAP_HOURS)[0]
        if len(gap_positions) == 0:
            # No gap found — ICU admission at end
            return [times[-1]]
        # ICU event is the start of each gap (obs just before the gap)
        return [times[i] for i in gap_positions]

    print("  Finding ICU event times per admission...")
    icu_events = []
    for adm_id, grp in icu_df.groupby("ANON_ADMISSION_ID"):
        for t in get_icu_event_times(grp):
            icu_events.append((adm_id, t))

    icu_events_df = pd.DataFrame(icu_events, columns=["ANON_ADMISSION_ID", "_icu_event_time"])
    print(f"  Total ICU events found: {len(icu_events_df):,}")

    # For each obs in ICU admissions, check if any ICU event is 0-24h ahead
    icu_obs = df[df["ANON_ADMISSION_ID"].isin(icu_ids)][
        ["ANON_ADMISSION_ID", "_abs_hours"]
    ].copy()
    icu_obs = icu_obs.merge(icu_events_df, on="ANON_ADMISSION_ID", how="left")
    icu_obs["_within_24h"] = (
        (icu_obs["_icu_event_time"] - icu_obs["_abs_hours"] <= 24) &
        (icu_obs["_icu_event_time"] - icu_obs["_abs_hours"] >= 0)
    )
    # If any ICU event triggers the flag, mark it
    icu_flag_per_obs = (
        icu_obs.groupby(["ANON_ADMISSION_ID", "_abs_hours"])["_within_24h"]
        .any()
        .reset_index()
        .rename(columns={"_within_24h": "ICU_WITHIN_24H"})
    )
    icu_flag_per_obs["ICU_WITHIN_24H"] = icu_flag_per_obs["ICU_WITHIN_24H"].astype(int)

    df = df.merge(icu_flag_per_obs, on=["ANON_ADMISSION_ID", "_abs_hours"], how="left")
    df["ICU_WITHIN_24H"] = df["ICU_WITHIN_24H"].fillna(0).astype(int)

    # --- Combined event flag ---
    df["EVENT_FLAG"] = ((df["DEATH_WITHIN_24H"] == 1) | (df["ICU_WITHIN_24H"] == 1)).astype(int)

    # Drop helper column
    df.drop(columns=["_abs_hours"], inplace=True)

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"Rows: {len(df):,}")
    print(f"DEATH_WITHIN_24H=1: {df['DEATH_WITHIN_24H'].sum():,}")
    print(f"ICU_WITHIN_24H=1:   {df['ICU_WITHIN_24H'].sum():,}")
    print(f"EVENT_FLAG=1:       {df['EVENT_FLAG'].sum():,}")
    print(f"EVENT_FLAG=0:       {(df['EVENT_FLAG']==0).sum():,}")

    print(f"\nSaving to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
