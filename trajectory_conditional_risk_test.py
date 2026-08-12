"""
Trajectory-Aware Conditional Risk Model — standalone brief test.
================================================================

STANDALONE. Imports nothing from `engine/`, writes nothing into `results/current/`,
touches no FEWS artefact. Delete this one file to remove the experiment entirely.

Implements the study plan's measurement question: at each observation, estimate
P(event within h=24h | trajectory of each vital), and test whether the derivative
terms (velocity, acceleration) add predictive value over levels alone.

Pipeline
--------
  1. Load observation-level dataset, build hours-since-admission timeline.
  2. Cohort: adult ward admissions with >= MIN_OBS observation sets.
  3. Censoring: an observation is kept only if the full 24h horizon is observed
     (last obs >= t + 24h) or the event itself occurred.
  4. Per observation, per vital: locally weighted QUADRATIC over the preceding
     24h window, exponential decay weights, fitted by weighted least squares.
     Extract level b0, velocity b1 (per HOUR), acceleration b2 (per HOUR^2) and
     the standard errors SE(b1), SE(b2). Never finite differences.
  5. Discrete-time hazard model on the observation grid (logistic link,
     restricted cubic splines, 4 knots). No probability multiplication.
  6. Ablation ladder M0..M4 + per-vital velocity contributions.
  7. Evaluate on a PATIENT-disjoint test split: AUROC, AUPRC, calibration.

Usage
-----
    python trajectory_conditional_risk_test.py                  # default sample
    python trajectory_conditional_risk_test.py --admissions 0   # full cohort
    python trajectory_conditional_risk_test.py --horizon-note   # see caveat below

Note on the horizon: EVENT_FLAG in this dataset is already defined as
"event within 24 h of this observation" (see engine/build_target_dataset.py), so
h = 24 h is fixed by the label, not by this script. The h in {6,12,48} sensitivity
analyses from the plan would need the raw event times, which this file does not
reconstruct.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = Path("datasets/final_observations_with_targets.csv")
OUT_DIR = Path("results/trajectory_risk_test")

HORIZON_H = 24.0        # prediction horizon (hours) — fixed by the label
WINDOW_H = 24.0         # trajectory look-back window (hours)
KAPPA_H = 12.0          # exponential weight decay constant (hours)
MAX_LOOKBACK = 16       # cap on prior observations gathered per window
MIN_OBS = 4             # cohort: minimum observation sets per admission
N_KNOTS = 4             # restricted cubic spline knots
TEST_FRAC = 0.30        # patient-level hold-out fraction
SEED = 20260811
DEFAULT_ADMISSIONS = 100_000   # 0 == use every admission

VITALS = {
    "HR":   "HEART_RATE",
    "SBP":  "SYSTOLIC_BP",
    "RR":   "RESP_RATE",
    "SPO2": "SATS_SPO2",
    "TEMP": "TEMPERATURE",
}
VKEYS = list(VITALS)

# ── FEWS reference-scoring config (only used with --with-fews) ───────────────
# Mirrors auroc_target_comparison_patient.py. Read-only: nothing here writes to
# engine/ or results/current/.
ACVPU_MAP = {"Alert": 0.0, "Responds to voice": 1.0,
             "Responds to pain": 2.0, "Unresponsive": 3.0}
O2CAT_CONCERN = {"Low": 1.0, "Low-moderate": 1.5, "Moderate": 2.0,
                 "High": 2.5, "Very high": 3.0}
TEMPORAL_VITALS = es.TEMPORAL_VITALS_DEFAULT   # canonical set (incl. inspired_oxygen);
# defined once in engine_scoring so the app and the pipeline cannot drift apart again.
FEWS_EVENT_OPTIMAL = (0.1, 1.0, 1.0)    # α, β, γ — event-optimal leaf
FEWS_SHERIF = (0.5, 5.0, 0.75)          # α, β, γ — Sherif's fixed params


# ── Loading ──────────────────────────────────────────────────────────────────
def load(n_admissions: int) -> pd.DataFrame:
    print("Loading dataset…")
    t0 = time.time()
    cols = ["ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
            *VITALS.values(), "ACVPU_SCORE", "INSPIRED_O2_TEXT",
            "AVPU_ACVPU", "INSP_O2_CAT",
            "COMPLETE_DATA", "NEWS-2", "EVENT_FLAG"]
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    print(f"  raw rows                     {len(df):>12,}  ({time.time()-t0:.0f}s)")

    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1]
    for c in [*VITALS.values(), "DAYS_SINCE_ADMISSION", "NEWS-2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(VITALS.values())).copy()
    print(f"  complete + non-null vitals   {len(df):>12,}")

    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_hours"] = (df["DAYS_SINCE_ADMISSION"] * 24.0
                     + obs.dt.hour.fillna(0)
                     + obs.dt.minute.fillna(0) / 60.0
                     + obs.dt.second.fillna(0) / 3600.0).astype(np.float64)
    df["hour_of_day"] = obs.dt.hour.fillna(0).astype(np.int8)
    df["NEWS-2"] = df["NEWS-2"].fillna(0)
    df["pid"] = df["ANON_ADMISSION_ID"].astype("int64")

    # Non-trajectory vitals that NEWS-2 also uses. Included as static covariates so
    # M0 vs M1 compares functional form on the same inputs, not different inputs.
    df["ACVPU_SCORE"] = pd.to_numeric(df["ACVPU_SCORE"], errors="coerce").fillna(0.0)
    df["INSPIRED_O2_TEXT"] = (pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce")
                              .fillna(21.0).clip(21, 100))
    df["FIO2"] = df["INSPIRED_O2_TEXT"]
    # Inputs the FEWS engine needs (--with-fews). Mappings match
    # auroc_target_comparison_patient.py so the reference scores are the real ones.
    df["ACVPU_NUM"] = df["AVPU_ACVPU"].map(ACVPU_MAP).fillna(0.0)
    df["O2_CONCERN"] = df["INSP_O2_CAT"].map(O2CAT_CONCERN).fillna(0.0).astype(np.float32)

    if n_admissions:
        ids = df["pid"].unique()
        if len(ids) > n_admissions:
            rng = np.random.default_rng(SEED)
            keep = rng.choice(ids, size=n_admissions, replace=False)
            df = df[np.isin(df["pid"].values, keep)]
            print(f"  sampled {n_admissions:,} of {len(ids):,} admissions "
                  f"→ {len(df):,} rows")

    df = df.sort_values(["pid", "t_hours"], kind="mergesort").reset_index(drop=True)
    return df


def apply_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Cohort filter: adult ward admissions with >= MIN_OBS observation sets."""
    n_obs = df.groupby("pid", sort=False)["t_hours"].transform("size")
    n_before = df["pid"].nunique()
    df = df[n_obs >= MIN_OBS].copy()
    print(f"\nCohort  >= {MIN_OBS} observation sets")
    print(f"  admissions kept              {df['pid'].nunique():>12,}  "
          f"of {n_before:,}  ({100*df['pid'].nunique()/n_before:.1f}%)")
    print(f"  observations kept            {len(df):>12,}")
    return df.reset_index(drop=True)


def apply_censoring(df: pd.DataFrame, extra_cols=()) -> pd.DataFrame:
    """Horizon censoring at the last recorded observation."""
    # Censoring: discharge (== last recorded observation) is not "no event".
    last_t = df.groupby("pid", sort=False)["t_hours"].transform("max")
    followed = (last_t - df["t_hours"]) >= HORIZON_H
    keep = followed | (df["EVENT_FLAG"] == 1)
    n_drop = int((~keep).sum())
    print(f"\nCensoring at horizon h = {HORIZON_H:.0f}h")
    print(f"  rows censored (dropped)      {n_drop:>12,}  "
          f"({100*n_drop/len(df):.1f}%)")
    keep_cols = ["pid", "t_hours", "hour_of_day", "EVENT_FLAG", "NEWS-2",
                 "ACVPU_SCORE", "FIO2", *VITALS.values(), *extra_cols]
    df = df.loc[keep, keep_cols].reset_index(drop=True)
    print(f"  analysis rows                {len(df):>12,}")
    print(f"  event prevalence             {df['EVENT_FLAG'].mean():>12.4%}")
    print(f"  admissions with >=1 event    {df.groupby('pid')['EVENT_FLAG'].max().sum():>12,}")
    return df


# ── FEWS reference scores (optional, read-only import of engine/) ────────────
def score_fews(df: pd.DataFrame) -> list[str]:
    """
    Score the real FEWS Snapshot and Temporal systems on the COHORT frame, i.e.
    before horizon censoring, so EWMA and slope windows see intact patient
    sequences. Adds columns in place and returns their names.

    Imports engine/ read-only. Writes nothing to engine/ or results/current/.
    """
    import engine.engine_scoring as es

    print("\n" + "=" * 78)
    print("FEWS REFERENCE SCORES  (engine/engine_scoring.py, read-only)")
    print("=" * 78)
    t0 = time.time()

    # ACVPU is not a scored vital (flag only) — this stays the 6-vital set
    vitals_full = es.VITALS_BASE
    luts = {v: es.build_lut(v) for v in vitals_full}
    luts["blood_pressure"] = es.build_lut("blood_pressure", sharper_sbp=True)

    pv = es.apply_luts(df, luts, vitals_full)
    pv["inspired_oxygen"] = df["O2_CONCERN"].values.astype(np.float32)

    gs, ge = es.group_boundaries(df["pid"].values)
    times = (df["t_hours"].values * 60.0).astype(np.float64)   # engine wants minutes
    slopes = es.compute_slopes(times, pv, gs, ge, vitals_full)

    added = []
    for label, (alpha, beta, gamma) in (("opt", FEWS_EVENT_OPTIMAL),
                                        ("sherif", FEWS_SHERIF)):
        alphas = {v: float(alpha) for v in vitals_full}
        refs = {v: es.EWMA_REF_DEFAULT for v in vitals_full}
        ewma = es.compute_ewma(times, pv, gs, ge, vitals_full, alphas, refs)
        if label == "opt":
            df["FEWS_snapshot"] = es.snapshot_score(
                pv, vitals_full, method="additive", gamma=gamma).astype(np.float32)
            added.append("FEWS_snapshot")
        col = f"FEWS_temporal_{label}"
        df[col] = es.temporal_score(pv, ewma, slopes, vitals_full, beta, gamma,
                                    method="additive",
                                    temporal_vitals=TEMPORAL_VITALS).astype(np.float32)
        added.append(col)
        print(f"  scored temporal α={alpha} β={beta} γ={gamma}  → {col}")

    print(f"  7 vitals (incl. ACVPU), sharper SBP LUT   ({time.time()-t0:.0f}s)")
    return added


# ── §3.4 pre-flight quantisation / recording-artefact check ──────────────────
def preflight_checks(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("PRE-FLIGHT — quantisation and recording artefacts (plan §3.4, §7.2)")
    print("=" * 78)
    same_pid = df["pid"].values[1:] == df["pid"].values[:-1]
    rows = []
    for k, col in VITALS.items():
        v = df[col].values
        d = v[1:] - v[:-1]
        d = d[same_pid]
        rows.append({
            "vital": k,
            "pct_identical_consecutive": 100 * np.mean(d == 0),
            "pct_even_valued": 100 * np.mean(np.isclose(v % 2, 0)),
            "n_distinct": int(pd.Series(v).nunique()),
            "median_abs_delta": float(np.median(np.abs(d))),
        })
    q = pd.DataFrame(rows)
    print(q.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    for r in rows:
        if r["pct_identical_consecutive"] > 60:
            print(f"  ! {r['vital']}: {r['pct_identical_consecutive']:.0f}% of consecutive "
                  f"values are identical — velocity is substantially recording behaviour.")

    hod = df["hour_of_day"].value_counts(normalize=True).sort_index() * 100
    peak = hod.nlargest(4)
    print("\n  Observation clustering by hour of day (top 4 hours, % of all obs):")
    print("   " + "  ".join(f"{int(h):02d}:00={p:.1f}%" for h, p in peak.items())
          + f"   [uniform would be {100/24:.1f}%]")
    return q


# ── Weighted local polynomial fits (vectorised) ──────────────────────────────
def _inv3_sym(a00, a01, a02, a11, a12, a22):
    """Batched analytic inverse of a symmetric 3x3. Returns diag terms + det."""
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c11 = a00 * a22 - a02 * a02
    c22 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02
    return c00, c01, c02, c11, c22, det


def fit_trajectories(df: pd.DataFrame, chunk: int = 300_000) -> dict[str, np.ndarray]:
    """
    For every observation, per vital, fit

        x(tau) ~= b0 + b1*(tau-t) + 0.5*b2*(tau-t)^2

    by WLS over the preceding WINDOW_H hours with weights exp(-(t-tau)/KAPPA_H).
    Quadratic when >= 4 points in window, weighted linear when 2-3, constant when 1.
    Derivatives are returned per HOUR / per HOUR^2 (never per observation).
    """
    print("\n" + "=" * 78)
    print(f"TRAJECTORY FEATURES — weighted local quadratic, {WINDOW_H:.0f}h window, "
          f"kappa={KAPPA_H:.0f}h")
    print("=" * 78)
    t0 = time.time()

    n = len(df)
    t = df["t_hours"].values.astype(np.float64)
    pid = df["pid"].values
    V = np.column_stack([df[c].values.astype(np.float64) for c in VITALS.values()])

    # first row index of each patient block (df is sorted by pid, t)
    new_pat = np.empty(n, dtype=bool)
    new_pat[0] = True
    new_pat[1:] = pid[1:] != pid[:-1]
    pstart = np.maximum.accumulate(np.where(new_pat, np.arange(n), 0))

    nv = len(VKEYS)
    out = {
        "b0": np.empty((n, nv)), "b1": np.empty((n, nv)), "b2": np.empty((n, nv)),
        "se1": np.full((n, nv), np.nan), "se2": np.full((n, nv), np.nan),
    }
    n_pts = np.empty(n, dtype=np.int16)
    quad_flag = np.zeros(n, dtype=np.int8)
    trunc_flag = np.zeros(n, dtype=np.int8)

    offs = np.arange(MAX_LOOKBACK + 1)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        k = np.arange(lo, hi)

        idx = k[:, None] - offs[None, :]
        ok = idx >= pstart[k][:, None]
        idx = np.where(ok, idx, k[:, None])
        dt = t[k][:, None] - t[idx]                 # >= 0 hours back
        ok &= (dt <= WINDOW_H) & (dt >= 0)

        w = np.where(ok, np.exp(-dt / KAPPA_H), 0.0)
        u = np.where(ok, -dt / WINDOW_H, 0.0)       # scaled lag in [-1, 0]

        cnt = ok.sum(1)
        n_pts[lo:hi] = cnt
        trunc_flag[lo:hi] = (cnt == MAX_LOOKBACK + 1)

        wu = w * u
        wuu = wu * u
        S0 = w.sum(1); S1 = wu.sum(1); S2 = wuu.sum(1)
        S3 = (wuu * u).sum(1); S4 = (wuu * u * u).sum(1)

        # symmetric normal matrix for basis [1, u, u^2/2]
        a00, a01, a02 = S0, S1, 0.5 * S2
        a11, a12, a22 = S2, 0.5 * S3, 0.25 * S4
        c00, c01, c02, c11, c22, det = _inv3_sym(a00, a01, a02, a11, a12, a22)
        det_ok = np.abs(det) > 1e-12 * np.maximum(a00, 1e-12) ** 3

        d2 = S0 * S2 - S1 * S1                      # linear-basis determinant
        d2_ok = np.abs(d2) > 1e-12 * np.maximum(S0, 1e-12) ** 2

        use_q = (cnt >= 4) & det_ok
        use_l = (~use_q) & (cnt >= 2) & d2_ok
        quad_flag[lo:hi] = use_q
        det_s = np.where(det_ok, det, 1.0)      # safe divisor; masked out below
        d2_s = np.where(d2_ok, d2, 1.0)
        # dof == 0 means the fit interpolates its points: the SE is not estimable,
        # so leave it missing rather than reporting a spurious zero.
        se1_ok = np.where(use_q, cnt > 3, cnt > 2)
        c12 = a01 * a02 - a00 * a12

        Vw = V[idx]                                 # (m, L+1, nv)
        for j in range(nv):
            x = Vw[:, :, j]
            wx = w * x
            m0 = wx.sum(1)
            m1 = (wx * u).sum(1)
            m2 = 0.5 * (wx * u * u).sum(1)
            sxx = (wx * x).sum(1)

            # --- quadratic ---
            q0 = (c00 * m0 + c01 * m1 + c02 * m2) / det_s
            q1 = (c01 * m0 + c11 * m1 + c12 * m2) / det_s
            q2 = (c02 * m0 + c12 * m1 + c22 * m2) / det_s
            rss_q = np.maximum(sxx - (q0 * m0 + q1 * m1 + q2 * m2), 0.0)
            s2_q = rss_q / np.maximum(cnt - 3, 1)
            v1_q = np.maximum(s2_q * c11 / det_s, 0.0)
            v2_q = np.maximum(s2_q * c22 / det_s, 0.0)

            # --- weighted linear fallback ---
            l0 = (S2 * m0 - S1 * m1) / d2_s
            l1 = (S0 * m1 - S1 * m0) / d2_s
            rss_l = np.maximum(sxx - (l0 * m0 + l1 * m1), 0.0)
            v1_l = np.maximum(rss_l / np.maximum(cnt - 2, 1) * S0 / d2_s, 0.0)

            # --- constant fallback ---
            k0 = m0 / np.where(S0 > 0, S0, 1.0)

            b0 = np.where(use_q, q0, np.where(use_l, l0, k0))
            b1 = np.where(use_q, q1, np.where(use_l, l1, 0.0))
            b2 = np.where(use_q, q2, 0.0)
            s1 = np.where(use_q, np.sqrt(v1_q), np.where(use_l, np.sqrt(v1_l), np.nan))
            s1 = np.where(se1_ok & (use_q | use_l), s1, np.nan)
            s2 = np.where(use_q & (cnt > 3), np.sqrt(v2_q), np.nan)

            # rescale from u = -dt/WINDOW_H back to per-hour units
            out["b0"][lo:hi, j] = b0
            out["b1"][lo:hi, j] = b1 / WINDOW_H
            out["b2"][lo:hi, j] = b2 / (WINDOW_H ** 2)
            out["se1"][lo:hi, j] = s1 / WINDOW_H
            out["se2"][lo:hi, j] = s2 / (WINDOW_H ** 2)

        if (lo // chunk) % 4 == 0:
            print(f"  … {hi:,}/{n:,} rows  ({time.time()-t0:.0f}s)", flush=True)

    out["n_pts"] = n_pts.astype(np.float64)
    out["quad_flag"] = quad_flag.astype(np.float64)
    out["trunc_flag"] = trunc_flag.astype(np.float64)

    print(f"  done in {time.time()-t0:.0f}s")
    print(f"  points in window: median {np.median(n_pts):.0f}, "
          f"mean {n_pts.mean():.1f}, capped at {MAX_LOOKBACK+1} for "
          f"{100*trunc_flag.mean():.2f}% of rows")
    print(f"  quadratic fitted for {100*quad_flag.mean():.1f}% of rows "
          f"(rest fell back to weighted linear/constant)")
    for j, kname in enumerate(VKEYS):
        b1 = out["b1"][:, j]
        print(f"    {kname:5s} velocity  median |b1| = {np.median(np.abs(b1)):8.4f} /h"
              f"   IQR [{np.percentile(b1,25):+.3f}, {np.percentile(b1,75):+.3f}]")
    return out


# ── Restricted cubic splines ─────────────────────────────────────────────────
def rcs_knots(x: np.ndarray, n_knots: int = N_KNOTS) -> np.ndarray | None:
    qs = {3: [.10, .50, .90], 4: [.05, .35, .65, .95],
          5: [.05, .275, .50, .725, .95]}[n_knots]
    kn = np.unique(np.quantile(x, qs))
    return kn if len(kn) >= 3 else None


def rcs_basis(x: np.ndarray, kn: np.ndarray | None) -> np.ndarray:
    """Harrell restricted cubic spline: k knots -> k-1 columns (linear + k-2)."""
    if kn is None:
        return x[:, None]
    k = len(kn)
    tk, tk1 = kn[-1], kn[-2]
    denom = (tk - kn[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        tj = kn[j]
        term = (np.maximum(x - tj, 0) ** 3
                - np.maximum(x - tk1, 0) ** 3 * (tk - tj) / (tk - tk1)
                + np.maximum(x - tk, 0) ** 3 * (tk1 - tj) / (tk - tk1))
        cols.append(term / denom)
    return np.column_stack(cols)


class SplineBlock:
    """Winsorise on train, then expand into an RCS basis, then standardise."""

    def __init__(self, name: str, spline: bool = True):
        self.name, self.spline = name, spline

    def fit(self, x: np.ndarray):
        self.lo, self.hi = np.percentile(x, [0.1, 99.9])
        if self.hi <= self.lo:
            self.lo, self.hi = x.min(), max(x.max(), x.min() + 1e-9)
        xc = np.clip(x, self.lo, self.hi)
        self.kn = rcs_knots(xc) if self.spline else None
        B = rcs_basis(xc, self.kn)
        self.mu, self.sd = B.mean(0), B.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        B = rcs_basis(np.clip(x, self.lo, self.hi), self.kn)
        return (B - self.mu) / self.sd

    @property
    def ncol(self) -> int:
        return 1 if self.kn is None else len(self.kn) - 1


# ── Feature assembly ─────────────────────────────────────────────────────────
def build_feature_dict(df: pd.DataFrame, tr: dict) -> dict[str, tuple[np.ndarray, bool]]:
    """name -> (raw 1-D column, use_spline). Keys drive the ablation ladder."""
    f: dict[str, tuple[np.ndarray, bool]] = {}
    for j, v in enumerate(VKEYS):
        f[f"lvl_{v}"] = (tr["b0"][:, j], True)
        f[f"vel_{v}"] = (tr["b1"][:, j], True)
        f[f"acc_{v}"] = (tr["b2"][:, j], True)

    # observation-precision terms (SE of slope / curvature), log-scaled
    med = {}
    for j, v in enumerate(VKEYS):
        for nm, key, scale in (("se1", "se1", 1.0), ("se2", "se2", 1.0)):
            s = tr[key][:, j] * scale
            m = np.nanmedian(s)
            med[f"{nm}_{v}"] = m
            f[f"{nm}_{v}"] = (np.log1p(np.where(np.isnan(s), m, s)), True)

    dt_prev = df.groupby("pid", sort=False)["t_hours"].diff().values
    dt_prev = np.where(np.isnan(dt_prev), np.nanmedian(dt_prev), dt_prev)
    f["log_dt_prev"] = (np.log1p(np.clip(dt_prev, 0, 72)), True)
    f["n_pts"] = (tr["n_pts"], True)
    f["quad_flag"] = (tr["quad_flag"], False)
    f["trunc_flag"] = (tr["trunc_flag"], False)

    # baseline hazard in time since admission
    f["t_adm"] = (np.log1p(np.clip(df["t_hours"].values, 0, None)), True)

    # static covariates (no trajectory) — the two remaining NEWS-2 inputs
    f["acvpu"] = (df["ACVPU_SCORE"].values.astype(np.float64), False)
    f["fio2"] = (df["FIO2"].values.astype(np.float64), True)
    return f


LEVEL_KEYS = [f"lvl_{v}" for v in VKEYS] + ["t_adm", "acvpu", "fio2"]
VEL_KEYS = [f"vel_{v}" for v in VKEYS]
ACC_KEYS = [f"acc_{v}" for v in VKEYS]
PREC_KEYS = ([f"se1_{v}" for v in VKEYS] + [f"se2_{v}" for v in VKEYS]
             + ["log_dt_prev", "n_pts", "quad_flag", "trunc_flag"])

LADDER = {
    "M0  NEWS-2 only":                       None,   # raw score, no fit
    "M1  Levels":                            LEVEL_KEYS,
    "M2  Levels + velocity":                 LEVEL_KEYS + VEL_KEYS,
    "M3  Levels + velocity + acceleration":  LEVEL_KEYS + VEL_KEYS + ACC_KEYS,
    "M4  M3 + precision & interval":         LEVEL_KEYS + VEL_KEYS + ACC_KEYS + PREC_KEYS,
}


# ── Evaluation ───────────────────────────────────────────────────────────────
def calibration_stats(y: np.ndarray, p: np.ndarray) -> dict:
    eps = 1e-9
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
    m.fit(lp[:, None], y)
    slope = float(m.coef_[0, 0])
    icpt = float(m.intercept_[0])
    # ICI: mean |p - smoothed(p)| using a spline recalibration curve
    blk = SplineBlock("lp").fit(lp)
    ms = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    ms.fit(blk.transform(lp), y)
    smooth = ms.predict_proba(blk.transform(lp))[:, 1]
    return {"cal_slope": slope, "cal_intercept": icpt,
            "ICI": float(np.mean(np.abs(p - smooth))),
            "brier": float(brier_score_loss(y, p))}


def alerts_per_1000_at_sens(y: np.ndarray, s: np.ndarray, sens: float = 0.80) -> float:
    """Rows flagged per 1000 observations at the threshold giving `sens` recall."""
    thr = np.quantile(s[y == 1], 1 - sens)
    return float(1000 * np.mean(s >= thr))


def evaluate(name: str, y: np.ndarray, p: np.ndarray, fitted: bool) -> dict:
    row = {"model": name,
           "AUROC": float(roc_auc_score(y, p)),
           "AUPRC": float(average_precision_score(y, p)),
           "alerts_per_1000_obs@80%sens": alerts_per_1000_at_sens(y, p)}
    if fitted:
        row.update(calibration_stats(y, p))
    return row


def fit_and_score(keys, feats, blocks, tr_idx, te_idx, y_tr, y_te):
    # float32 keeps the full-cohort design matrices in memory; sklearn preserves it.
    Xtr = np.column_stack([blocks[k].transform(feats[k][0][tr_idx])
                           for k in keys]).astype(np.float32)
    model = LogisticRegression(penalty="l2", C=10.0, solver="lbfgs",
                               max_iter=2000, n_jobs=-1)
    model.fit(Xtr, y_tr)
    ncol = Xtr.shape[1]
    del Xtr
    Xte = np.column_stack([blocks[k].transform(feats[k][0][te_idx])
                           for k in keys]).astype(np.float32)
    p = model.predict_proba(Xte)[:, 1]
    del Xte
    return p, ncol


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--admissions", type=int, default=DEFAULT_ADMISSIONS,
                    help="sample this many admissions (0 = all)")
    ap.add_argument("--with-fews", action="store_true",
                    help="also score FEWS Snapshot/Temporal on the identical test "
                         "rows (read-only import of engine/)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 78)
    print("TRAJECTORY-AWARE CONDITIONAL RISK MODEL — standalone test")
    print(f"horizon h = {HORIZON_H:.0f}h   window = {WINDOW_H:.0f}h   "
          f"kappa = {KAPPA_H:.0f}h   knots = {N_KNOTS}")
    print("=" * 78)

    df = load(args.admissions)
    df = apply_cohort(df)
    fews_cols = score_fews(df) if args.with_fews else []
    df = apply_censoring(df, extra_cols=fews_cols)
    qcheck = preflight_checks(df)
    qcheck.to_csv(OUT_DIR / "preflight_quantisation.csv", index=False)

    tr = fit_trajectories(df)
    feats = build_feature_dict(df, tr)

    # ── patient-disjoint split ───────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    ids = df["pid"].unique()
    te_ids = rng.choice(ids, size=int(TEST_FRAC * len(ids)), replace=False)
    is_te = np.isin(df["pid"].values, te_ids)
    tr_idx, te_idx = np.where(~is_te)[0], np.where(is_te)[0]
    y = df["EVENT_FLAG"].values.astype(np.int8)
    y_tr, y_te = y[tr_idx], y[te_idx]

    print("\n" + "=" * 78)
    print("SPLIT — by patient, never by observation")
    print("=" * 78)
    print(f"  train  {len(tr_idx):>10,} obs / {len(ids)-len(te_ids):>8,} admissions"
          f"   prevalence {y_tr.mean():.4%}")
    print(f"  test   {len(te_idx):>10,} obs / {len(te_ids):>8,} admissions"
          f"   prevalence {y_te.mean():.4%}")
    print("  no resampling — fitted at natural prevalence")

    # fit spline blocks on TRAIN only
    blocks = {k: SplineBlock(k, spline=sp).fit(v[tr_idx])
              for k, (v, sp) in feats.items()}

    # ── ablation ladder ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("ABLATION LADDER  (test set, patient-disjoint)")
    print("=" * 78)
    rows = []
    # Existing systems, scored on exactly these test rows — unfitted ranking scores,
    # so no train/test leakage concern and no calibration stats.
    fews_ref = {"R1  FEWS Snapshot (α=.1 β=1 γ=1)": "FEWS_snapshot",
                "R2  FEWS Temporal (α=.1 β=1 γ=1)": "FEWS_temporal_opt",
                "R3  FEWS Temporal (Sherif α=.5 β=5 γ=.75)": "FEWS_temporal_sherif"}
    for name, col in fews_ref.items():
        if col not in df.columns:
            continue
        r = evaluate(name, y_te, df[col].values[te_idx].astype(float), fitted=False)
        r["n_terms"] = 0
        rows.append(r)
        print(f"  {name:<40s} AUROC {r['AUROC']:.4f}   AUPRC {r['AUPRC']:.4f}")

    for name, keys in LADDER.items():
        t0 = time.time()
        if keys is None:
            p = df["NEWS-2"].values[te_idx].astype(float)
            p = p / max(p.max(), 1.0)          # rank-preserving; not a probability
            rows.append(evaluate(name, y_te, p, fitted=False))
            ncol = 1
        else:
            p, ncol = fit_and_score(keys, feats, blocks, tr_idx, te_idx, y_tr, y_te)
            rows.append(evaluate(name, y_te, p, fitted=True))
        rows[-1]["n_terms"] = ncol
        print(f"  {name:<40s} AUROC {rows[-1]['AUROC']:.4f}   "
              f"AUPRC {rows[-1]['AUPRC']:.4f}   ({time.time()-t0:.0f}s)", flush=True)

    ladder = pd.DataFrame(rows)
    ladder.to_csv(OUT_DIR / "ablation_ladder.csv", index=False)

    # ── per-vital velocity contribution (the headline claim) ─────────────────
    print("\n" + "=" * 78)
    print("PER-VITAL VELOCITY CONTRIBUTION  (M1 + velocity of one vital only)")
    print("=" * 78)
    base = ladder.loc[ladder["model"].str.startswith("M1")].iloc[0]
    pv = []
    for v in VKEYS:
        p, _ = fit_and_score(LEVEL_KEYS + [f"vel_{v}"], feats, blocks,
                             tr_idx, te_idx, y_tr, y_te)
        r = evaluate(f"M1 + vel_{v}", y_te, p, fitted=False)
        r["dAUROC_vs_M1"] = r["AUROC"] - base["AUROC"]
        r["dAUPRC_vs_M1"] = r["AUPRC"] - base["AUPRC"]
        pv.append(r)
        print(f"  {v:<6s} AUROC {r['AUROC']:.4f} ({r['dAUROC_vs_M1']:+.4f})   "
              f"AUPRC {r['AUPRC']:.4f} ({r['dAUPRC_vs_M1']:+.4f})", flush=True)
    pvdf = pd.DataFrame(pv)
    pvdf.to_csv(OUT_DIR / "per_vital_velocity.csv", index=False)

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    show = ["model", "AUROC", "AUPRC", "alerts_per_1000_obs@80%sens",
            "cal_slope", "cal_intercept", "ICI", "brier", "n_terms"]
    print(ladder[show].to_string(index=False,
                                 float_format=lambda x: f"{x:10.4f}"))
    m1 = ladder.loc[ladder.model.str.startswith("M1")].iloc[0]
    m2 = ladder.loc[ladder.model.str.startswith("M2")].iloc[0]
    m3 = ladder.loc[ladder.model.str.startswith("M3")].iloc[0]
    print(f"\n  M1 -> M2 (velocity added):     "
          f"dAUROC {m2.AUROC-m1.AUROC:+.4f}   dAUPRC {m2.AUPRC-m1.AUPRC:+.4f}")
    print(f"  M2 -> M3 (acceleration added): "
          f"dAUROC {m3.AUROC-m2.AUROC:+.4f}   dAUPRC {m3.AUPRC-m2.AUPRC:+.4f}")
    print(f"\n  Test-set event prevalence (AUPRC floor): {y_te.mean():.4%}")
    print(f"  Outputs → {OUT_DIR}/")
    print(f"  Total runtime {time.time()-t_start:.0f}s")
    print("\n  Caveats not addressed by this brief test: treatment paradox "
          "(plan §7.1),\n  cluster-robust inference, lead-time analysis, and "
          "h in {6,12,48} sensitivity.")


if __name__ == "__main__":
    main()
