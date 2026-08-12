"""Shared loading, scoring and lead-time machinery for the results/main pipeline.

Both ``main_grid_search.py`` and ``main_validation.py`` import from here so the two
stages agree exactly on preprocessing, cohort sampling and the temporal formula
(``engine_scoring.temporal_score`` — EWMA memory + sigmoid worsening-trend, the same
path app/streamlit_app.py uses).

The one genuinely new thing in this module is the lead-time definition (see
``lead_time_patient`` / ``lead_time_row``): lead is now measured **backward from the
true event time**, not forward from the start of the 24 h label window, and it uses
the most recent contiguous alert episode rather than the first crossing anywhere in
the stay. ``derive_event_times`` recovers the true event time by replaying the label
construction in engine/build_target_dataset.py.
"""

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "engine"))
import engine_scoring as es  # noqa: E402

DATA_PATH   = REPO / "datasets" / "final_observations_with_targets.csv"
MAIN_DIR    = REPO / "results" / "main"
PATIENT_DIR = MAIN_DIR / "patient level"
ROW_DIR     = MAIN_DIR / "row-level"

RANDOM_SEED = 42
NE_PATIENTS = 22_336      # non-event admissions kept for the grid-search cohort
ROW_NEG_CAP = 500_000     # non-event observations kept for row-level metrics
ICU_GAP_MIN = 24.0 * 60.0 # gap that infers an ICU transfer (build_target_dataset.py)

FIXED_PARAMS = (0.5, 5.0, 0.75)   # "Sherif's params"

ALPHA_VALS = np.round(np.arange(0.1, 1.05, 0.1), 2)
BETA_VALS  = np.round(np.arange(0.0, 4.55, 0.5), 1)
GAMMA_VALS = np.round(np.arange(0.1, 1.05, 0.1), 2)

# The EWMA + trend layer applies only to continuously-varying physiological vitals;
# inspired_oxygen and ACVPU are categorical step-signals whose transitions create
# artefactual excursions, so they stay at their snapshot value.
TEMPORAL_VITALS = es.TEMPORAL_VITALS_DEFAULT   # canonical set (incl. inspired_oxygen);
# defined once in engine_scoring so the app and the pipeline cannot drift apart again.

O2CAT_CONCERN = {"Low": 1.0, "Low-moderate": 1.5, "Moderate": 2.0,
                 "High": 2.5, "Very high": 3.0}

TARGETS = {"Death 24h": "DEATH_WITHIN_24H",
           "ICU 24h":   "ICU_WITHIN_24H",
           "Event 24h": "EVENT_FLAG"}
TARGET_SHORT = {"DEATH_WITHIN_24H": "death", "ICU_WITHIN_24H": "icu",
                "EVENT_FLAG": "event"}

SYSTEMS   = ["NEWS-2", "Snapshot Fuzzy", "Temporal Fuzzy"]
SYS_COLOR = {"NEWS-2": "#E74C3C", "Snapshot Fuzzy": "#3498DB", "Temporal Fuzzy": "#2ECC71"}

# Text columns held as pandas categoricals — without this the full 9.3M-row frame
# with every source column costs several GB of Python strings.
CAT_COLS = ["CHART_TYPE", "HR_CAT", "SBP_CAT", "RESP_CAT", "SATS_CAT", "TEMP_CAT",
            "INSP_O2_CAT", "AVPU_ACVPU", "DISCHARGING_SPECIALTY", "OBS_TIME"]


# ── Load ──────────────────────────────────────────────────────────────────────

def load(all_columns: bool = False) -> pd.DataFrame:
    """Load, filter and time-order the observation dataset.

    all_columns=True keeps every source column (needed to write scored_data_full.csv);
    False loads only what scoring and labelling require, which is much lighter.
    """
    core = ["ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
            "HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
            "INSPIRED_O2_TEXT", "INSP_O2_CAT", "AVPU_ACVPU", "TEMPERATURE",
            "COMPLETE_DATA", "NEWS-2", "ACVPU_SCORE", "DIED_FLAG", "ICU_FLAG",
            "DEATH_WITHIN_24H", "ICU_WITHIN_24H", "EVENT_FLAG"]

    print(f"Loading {DATA_PATH.name} ({'all columns' if all_columns else 'core columns'})…",
          flush=True)
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, usecols=None if all_columns else core, low_memory=False)
    print(f"  {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)

    for c in CAT_COLS:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype("category")

    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    print(f"  After COMPLETE_DATA filter: {len(df):,} rows")

    for c in ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
              "TEMPERATURE", "DAYS_SINCE_ADMISSION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE",
                      "SATS_SPO2", "TEMPERATURE"], inplace=True)
    print(f"  After core-vital completeness filter: {len(df):,} rows")

    df["INSPIRED_O2_TEXT"] = (pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce")
                              .fillna(21.0).clip(21.0, 100.0))
    df["NEWS-2"] = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0)
    df["ACVPU_SCORE"] = pd.to_numeric(df["ACVPU_SCORE"], errors="coerce").fillna(0.0)
    df["ACVPU_NUM"] = (df["AVPU_ACVPU"].astype(object).map(es.ACVPU_MAP)
                       .fillna(0.0).astype(np.float32))
    df["O2_CONCERN"] = (df["INSP_O2_CAT"].astype(object).map(O2CAT_CONCERN)
                        .fillna(0.0).astype(np.float32))
    for c in ["DIED_FLAG", "ICU_FLAG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)

    obs = pd.to_datetime(df["OBS_TIME"].astype(object), format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"] * 1440.0
                       + obs.dt.hour.fillna(0) * 60.0
                       + obs.dt.minute.fillna(0)
                       + obs.dt.second.fillna(0) / 60.0).astype(np.float64)
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)

    on_o2 = df["O2_CONCERN"].values > 0
    print(f"  O2 via INSP_O2_CAT: {on_o2.sum():,} rows ({100*on_o2.mean():.1f}%)")
    return df


def build_pv(df, luts, vitals):
    """Per-vital 0-3 concern scores; inspired_oxygen comes from the clinical
    delivery category (INSP_O2_CAT) rather than the unreliable INSPIRED_O2_TEXT."""
    pv = es.apply_luts(df, luts, vitals)
    pv["inspired_oxygen"] = df["O2_CONCERN"].values.astype(np.float32)
    return pv


def build_luts(vitals):
    luts = {v: es.build_lut(v) for v in vitals}
    luts["blood_pressure"] = es.build_lut("blood_pressure", sharper_sbp=True)
    return luts


def temporal(pv, ewma, slopes, vitals, beta, gamma):
    return es.temporal_score(pv, ewma, slopes, vitals, beta, gamma,
                             method="additive", temporal_vitals=TEMPORAL_VITALS)


# ── True event times ──────────────────────────────────────────────────────────

def derive_event_times(df, gs, ge):
    """Per-admission true event time in t_minutes, replaying the label construction
    in engine/build_target_dataset.py:

      death → the admission's last observation (that is what the labeller treats as
              the time of death)
      ICU   → the observation immediately preceding the first >24 h gap in the
              record; if the stay has no such gap, the last observation

    Returns (death_t, icu_t, event_t), each len(gs), NaN where the admission has no
    event of that kind. event_t is the earlier of the two when both apply — the first
    deterioration is what the lead-time table is asking about.
    """
    t   = df["t_minutes"].values
    ids = df["ANON_ADMISSION_ID"].values
    died = df["DIED_FLAG"].values[gs].astype(bool)
    icu  = df["ICU_FLAG"].values[gs].astype(bool)
    last_t = t[ge - 1]

    # first >24h intra-admission gap: index i means the gap sits between row i and i+1
    dt = np.diff(t)
    same = ids[1:] == ids[:-1]
    gap_idx = np.flatnonzero(same & (dt > ICU_GAP_MIN))

    k = np.searchsorted(gap_idx, gs)
    safe = np.minimum(k, max(len(gap_idx) - 1, 0))
    has_gap = (k < len(gap_idx)) & (gap_idx[safe] < ge - 1) if len(gap_idx) else np.zeros(len(gs), bool)
    icu_t_all = np.where(has_gap, t[gap_idx[safe]] if len(gap_idx) else last_t, last_t)

    death_t = np.where(died, last_t, np.nan)
    icu_t   = np.where(icu,  icu_t_all, np.nan)
    event_t = np.fmin(death_t, icu_t)     # fmin ignores NaN unless both are NaN
    return death_t, icu_t, event_t


# ── Lead time — measured backward from the true event ─────────────────────────

DET_TARGETS = [0.60, 0.70, 0.80, 0.90]
N_THRESHOLDS = 120


def _thresholds(score):
    return np.unique(np.round(np.quantile(score, np.linspace(0.50, 0.9995, N_THRESHOLDS)), 4))


def lead_time_patient(score, times, gs, ge, event_t, e_pat):
    """Patient-level lead time, measured backward from the true event time.

    For one event admission and one threshold T: look only at observations at or
    before the event, find the most recent one that is at/above T, then walk backward
    while the score stays at/above T. Lead = event time − the time that unbroken alert
    episode began. So it answers "when did the alert that was running into this
    patient's event start", not "when did this stay first ever touch T" — a single
    spike days earlier no longer sets the clock.

    Returns (thresholds, detection_rate, median_lead_h, false_alarm_rate); detection
    counts an event admission whose pre-event record alerts at all, so the sensitivity
    axis stays comparable with the previous table.
    """
    ev = np.flatnonzero((e_pat == 1) & np.isfinite(event_t))
    ne = np.flatnonzero(e_pat == 0)

    # Per event admission, cache the pre-event slice once (scores + times).
    pre = []
    for g in ev:
        s, e = gs[g], ge[g]
        tt = times[s:e]
        m = tt <= event_t[g]
        if not m.any():
            continue
        pre.append((score[s:e][m], tt[m], event_t[g]))

    peak = np.maximum.reduceat(score, gs)
    neg_peak = peak[ne]
    thr = _thresholds(score)

    det, lead, fa = [], [], []
    for T in thr:
        n_det = 0
        leads = []
        for sc, tt, et in pre:
            alert = sc >= T
            if not alert.any():
                continue
            n_det += 1
            j = len(alert) - 1 - int(np.argmax(alert[::-1]))    # last alerting index
            quiet = np.flatnonzero(~alert[:j + 1])
            r = quiet[-1] + 1 if quiet.size else 0             # start of that run
            leads.append((et - tt[r]) / 60.0)
        det.append(n_det / len(pre) if pre else np.nan)
        lead.append(float(np.median(leads)) if leads else np.nan)
        fa.append(float(np.mean(neg_peak >= T)))
    return thr, np.array(det), np.array(lead), np.array(fa)


def lead_time_row(score, times, gs, ge, event_t, e_row):
    """Row-level lead time, measured backward from the true event time.

    Unit of analysis is one observation, so this asks a per-reading question: of the
    observations sitting inside the labelled pre-event window, what fraction alert at
    threshold T (sensitivity), and how far ahead of the event does an alerting one
    land (median lead)? False-alarm rate is the fraction of non-event observations
    at/above T. Lead is bounded by the 24 h label window by construction, which is
    exactly the difference from the patient-level table.
    """
    n = len(score)
    et_row = np.repeat(event_t, ge - gs)
    pos = (e_row == 1) & np.isfinite(et_row) & (times <= et_row)
    neg = e_row == 0

    pos_score = score[pos]
    pos_lead = (et_row[pos] - times[pos]) / 60.0
    neg_score = score[neg]
    thr = _thresholds(score)

    det, lead, fa = [], [], []
    for T in thr:
        a = pos_score >= T
        det.append(float(a.mean()) if pos_score.size else np.nan)
        lead.append(float(np.median(pos_lead[a])) if a.any() else np.nan)
        fa.append(float((neg_score >= T).mean()) if neg_score.size else np.nan)
    return thr, np.array(det), np.array(lead), np.array(fa)


def lead_time_table(curves):
    """Interpolate each system's curve onto the shared sensitivity targets.

    ``curves``: {system: (thr, det, lead, fa)}. Rows past a system's maximum
    achievable detection are left blank rather than extrapolated, and the ceiling is
    reported so a big lead time bought by never reaching the other systems'
    sensitivity is visible rather than hidden.
    """
    rows = []
    for name, (thr, det, lead, fa) in curves.items():
        order = np.argsort(det)
        d, l, f = det[order], lead[order], fa[order]
        ok = np.isfinite(l)
        max_det = float(np.nanmax(det))
        for D in DET_TARGETS:
            if D > max_det or not ok.any():
                li, fi = np.nan, np.nan
            else:
                li = float(np.interp(D, d[ok], l[ok]))
                fi = float(np.interp(D, d, f))
            rows.append({
                "Sensitivity (% events detected pre-onset)": D,
                "System": name,
                "Median lead (hours)": round(li, 1) if np.isfinite(li) else "",
                "False-alarm rate (1 - specificity)": round(fi, 3) if np.isfinite(fi) else "",
                "Max detection reached": round(max_det, 3),
            })
    return pd.DataFrame(rows)
