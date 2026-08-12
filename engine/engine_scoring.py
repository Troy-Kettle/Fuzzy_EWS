"""
Shared fuzzy-EWS scoring module — single source of truth for the improvement
experiments (realises plan items A1, A2, A3, B1, C1, D1, D2, D3).

This deliberately reproduces the CANONICAL engine logic in
fuzzy_system/streamlit_app.py (defuzz centroid + aggregate_total + five-set SBP),
but vectorised via per-vital lookup tables so it is fast over millions of rows.

Single source of truth (A3): ``grid_search_excess_patient.py``,
``auroc_target_comparison_patient.py`` and ``trajectory_auroc_patient.py`` all now
call ``temporal_score`` / ``snapshot_score`` here rather than carrying their own
copies (earlier they each inlined the temporal maths — that divergence is removed,
and ``tests/test_engine_scoring.py`` locks the engine against regression). The
categorical-exclusion rule lives here via the ``temporal_vitals`` argument.

Key corrections vs the earlier analysis scripts:
  A1  defuzz returns EXACTLY 0 when only "No concern" fires (streamlit_app.py:454-458),
      so a perfectly normal vital scores 0 (was 0.2467 → normal patient ~1.48). Depends
      on the MIN_FIRING gate — see the comment there for what turning it off costs.
  A2  ACVPU text→value uses the canonical engine map (AVPU_FUZZY_SCORE, streamlit_app.py:55):
      {Alert:0, Voice:1, Confused:2, Pain:3, Unresponsive:3}.
  C1  systolic BP uses five sets, not seven: above-mild and above-moderate are absorbed
      into No concern, leaving above-severe as the only above-normal set. This is the
      ONLY SBP model — see ``_merge_sbp_no_concern``.

Aggregation methods (B1, engine aggregate_total):
  additive       Σ vᵢ                                   (total burden)
  multiplicative (1 − Π(1 − vᵢ/3)) · 3 · n              (noisy-OR; any vital concerning)
  nonlinear      (mean(vᵢ/3)^p)^(1/p) · 3 · n           (power-mean; emphasise worst)
γ then mixes the chosen base with a worst-vital term (unchanged from before).

Temporal adjustment (``temporal_score``) is the same two-step EWMA + sigmoid
worsening-trend formula as app/streamlit_app.py (the interactive "main system"):
EWMA memory (clamped up to raw) is pushed further toward 3 by a sigmoid factor of
the OLS slope of raw scores over a look-back window (``compute_slopes``), scaled by
β. This replaced an earlier excess-EWMA (`raw + β·max(0, raw−EWMA)`) design that
diverged from the app; the two are now unified so validation exercises the same
logic the app presents to users. Temporal α/β/γ still supports per-vital α (D1) and
per-vital EWMA reference (D2) via ``compute_ewma``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO        = Path(__file__).resolve().parent.parent
SIGMOID_DIR = REPO / "membership_functions" / "sigmoid"

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (trapz deprecated).
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ── Vitals ────────────────────────────────────────────────────────────────────
VITALS_BASE = ["heart_rate", "blood_pressure", "temperature",
               "respiratory_rate", "oxygen_saturation", "inspired_oxygen"]
ACVPU = "acvpu"
# Not a member of VITALS_BASE: supplementary O2 flow is an ALTERNATIVE unit for the
# inspired_oxygen vital, not a seventh vital. Datasets that record oxygen in mixed
# units (a value plus a "%" / "litres" unit column) score each row through the
# membership function for its OWN unit — see ``inspired_oxygen_concern``.
SUPP_O2_LMIN = "supplementary_oxygen_lmin"

VITAL_COL = {"heart_rate": "HEART_RATE", "blood_pressure": "SYSTOLIC_BP",
             "temperature": "TEMPERATURE", "respiratory_rate": "RESP_RATE",
             "oxygen_saturation": "SATS_SPO2", "inspired_oxygen": "INSPIRED_O2_TEXT",
             "supplementary_oxygen_lmin": "SUPP_O2_LMIN", "acvpu": "ACVPU_NUM"}
MF_FILE = {"heart_rate": "heart_rate_membership_functions.csv",
           "blood_pressure": "systolic_blood_pressure_membership_functions.csv",
           "temperature": "temperature_membership_functions.csv",
           "respiratory_rate": "respiratory_rate_membership_functions.csv",
           "oxygen_saturation": "oxygen_saturation_membership_functions.csv",
           "inspired_oxygen": "inspired_oxygen_concentration_membership_functions.csv",
           "supplementary_oxygen_lmin": "supplementary_oxygen_lmin_membership_functions.csv",
           "acvpu": "avpu_acvpu_membership_functions.csv"}
VITAL_TYPE = {"heart_rate": "7var", "blood_pressure": "7var", "temperature": "7var",
              "respiratory_rate": "7var", "oxygen_saturation": "3var_down",
              "inspired_oxygen": "3var_up", "supplementary_oxygen_lmin": "3var_up",
              "acvpu": "3var_up"}

# A2: canonical engine ACVPU mapping (Voice=1, Confused=2)
ACVPU_MAP = {"Alert": 0.0, "Responds to voice": 1.0,
             "Newly confused / agitated": 2.0, "Responds to pain": 3.0,
             "Unresponsive": 3.0}

# ACVPU contributes NOTHING to the fuzzy score (decision of 2026-08-12). It is not a
# scored vital and there is no bonus: any reading other than Alert makes the whole row a
# positive flag of deterioration on its own, the way a non-Alert ACVPU triggers
# escalation in practice regardless of the aggregated number. Use
# ``acvpu_deterioration_flag`` — that flag IS the ACVPU output.
#
# Kept at 0.0 rather than deleted so that any caller still passing ``acvpu_raw=`` to
# snapshot_score/temporal_score becomes a no-op instead of silently reintroducing the
# old +3. This applies to EVERY system including the NEWS-2 baseline, whose consciousness
# sub-score is also 0 here — see ``news2_consciousness_score``.
ACVPU_BONUS = 0.0


def acvpu_deterioration_flag(acvpu_num) -> np.ndarray:
    """True wherever ACVPU indicates anything other than Alert (ACVPU_NUM > 0).
    Independent of the aggregated fuzzy/temporal score — a separate automatic
    flag of deterioration, not a substitute for it."""
    return np.asarray(acvpu_num) > 0.0


def apply_acvpu_bonus(total: np.ndarray, acvpu_num, bonus: float = ACVPU_BONUS) -> np.ndarray:
    """DISABLED — ``ACVPU_BONUS`` is 0.0, so this returns ``total`` unchanged.

    ACVPU adds nothing to the fuzzy score; ``acvpu_deterioration_flag`` is the whole of
    its contribution. Retained only so existing ``acvpu_raw=`` call sites stay harmless.
    """
    flag = acvpu_deterioration_flag(acvpu_num).astype(total.dtype)
    return total + bonus * flag


# ── Inspired oxygen recorded in mixed units (value + "%" / "litres") ─────────
# Datasets such as "Annotated dataset_training_anonymised_V5_Troy" record oxygen as
# INSPIRED_O2 plus INSPIRED_O2_UNITS, which is either "%" (FiO2) or "litres"
# (supplementary flow, L/min). The two are NOT interconverted: flow is kept in
# L/min and scored through its own membership function
# (supplementary_oxygen_lmin_membership_functions.csv). An earlier version of the
# annotated pipelines replaced every flow reading with a pseudo-FiO2
# (21 + 4·L/min) and fed it through the FiO2 membership function; that fabricated
# concentrations the source never recorded (e.g. 0.5 L/min → "23%") and is no
# longer used.
_UNIT_PCT   = {"%", "percent", "pct", "fio2"}
_UNIT_LITRE = {"litres", "liters", "litre", "liter", "l/min", "lpm", "l"}


def split_inspired_oxygen(value, units):
    """Split a mixed-unit oxygen column into (fio2_pct, flow_lmin), each NaN where
    that unit does not apply — the raw source values, with no interconversion.

    Rows whose unit is missing/unrecognised are treated as room air: FiO2 21%,
    flow NaN."""
    value = pd.to_numeric(pd.Series(value).reset_index(drop=True), errors="coerce")
    u = pd.Series(units).reset_index(drop=True).astype(str).str.strip().str.lower()
    is_pct, is_flow = u.isin(_UNIT_PCT), u.isin(_UNIT_LITRE)

    fio2 = pd.Series(np.nan, index=value.index, dtype="float64")
    flow = pd.Series(np.nan, index=value.index, dtype="float64")
    fio2[is_pct] = value[is_pct]
    flow[is_flow] = value[is_flow]
    # unknown unit, or a "%"/"litres" row with no number: room air
    fio2[(~is_pct & ~is_flow) | value.isna()] = 21.0
    return fio2.to_numpy(np.float64), flow.to_numpy(np.float64)


def on_supplemental_oxygen(fio2_pct, flow_lmin) -> np.ndarray:
    """NEWS-2's binary "on supplemental oxygen" condition, evaluated in each row's
    own units: FiO2 above room air (>21%) OR any positive supplementary flow.
    Rows with neither recorded are room air (False)."""
    fio2 = np.asarray(fio2_pct, dtype=np.float64)
    flow = np.asarray(flow_lmin, dtype=np.float64)
    return ((np.nan_to_num(fio2, nan=21.0) > 21.0)
            | (np.nan_to_num(flow, nan=0.0) > 0.0))


def inspired_oxygen_concern(fio2_pct, flow_lmin, fio2_lut, lmin_lut) -> np.ndarray:
    """Per-row inspired-oxygen concern (0-3) from whichever unit the row records,
    scored through that unit's own LUT: FiO2% → ``fio2_lut``, flow L/min →
    ``lmin_lut`` (both from ``build_lut``). Flow takes precedence where a row somehow
    carries both. Flow ≤ 0 is room air and is scored as 21% on ``fio2_lut``, so it
    agrees with rows that record room air as "21%" (with MIN_FIRING disabled that is
    ~0.27, not 0 — hence routing rather than hard-coding a zero)."""
    fio2 = np.asarray(fio2_pct, dtype=np.float64)
    flow = np.asarray(flow_lmin, dtype=np.float64)
    fx, fy = fio2_lut

    # default = room air; overwritten below by whichever unit the row actually records
    out = np.full(fio2.shape, np.interp(21.0, fx, fy), dtype=np.float32)

    has_pct = ~np.isnan(fio2)
    if has_pct.any():
        out[has_pct] = np.interp(np.clip(fio2[has_pct], fx[0], fx[-1]), fx, fy)
    has_flow = ~np.isnan(flow) & (flow > 0.0)
    if has_flow.any():
        lx, ly = lmin_lut
        out[has_flow] = np.interp(np.clip(flow[has_flow], lx[0], lx[-1]), lx, ly)
    return out


# ── NEWS-2 (chronic-aware Scale 1 / Scale 2 SpO2 + consciousness) ────────────
def news2_spo2_score(spo2, chronic_resp, on_oxygen) -> np.ndarray:
    """Vectorised NEWS-2 SpO2 sub-score, mirroring app/streamlit_app.py's
    ``calculate_news2``. Scale 1 (default, target 96-98%) is used unless
    ``chronic_resp`` is truthy for that row, in which case Scale 2 applies
    (target 88-92%, for known chronic hypercapnic respiratory failure e.g.
    COPD) — above the target range Scale 2 additionally depends on whether the
    patient is on supplemental oxygen."""
    spo2 = np.asarray(spo2, dtype=np.float64)
    chronic = np.asarray(chronic_resp).astype(bool)
    on_o2 = np.asarray(on_oxygen).astype(bool)

    scale1 = np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95], [3, 2, 1], default=0)

    s2_low = np.select([spo2 <= 83, spo2 <= 85, spo2 <= 87, spo2 <= 92], [3, 2, 1, 0], default=0)
    s2_high_air = np.zeros_like(spo2)                       # 93-100% on air: 0
    s2_high_o2 = np.select([spo2 <= 94, spo2 <= 96], [1, 2], default=3)
    scale2 = np.where(spo2 <= 92, s2_low, np.where(on_o2, s2_high_o2, s2_high_air))

    return np.where(chronic, scale2, scale1).astype(np.int64)


def news2_consciousness_score(acvpu_num) -> np.ndarray:
    """Always 0 — NEWS-2 gets no consciousness sub-score here either (decision of
    2026-08-12).

    ACVPU is out-of-band for EVERY system, not just the fuzzy one: a non-Alert reading is
    handled as a positive deterioration flag during validation
    (``acvpu_deterioration_flag``) rather than as points in any total. Real NEWS-2 awards
    3 here, but leaving it in would give the NEWS-2 baseline a consciousness input the
    fuzzy scores do not have and make the comparison unmatched.

    Returns zeros rather than being deleted so every existing call site stays valid and
    inert. Note ``NEWS_2_Score_Source`` in the annotated spreadsheet is the dataset's own
    precomputed NEWS-2 and DOES still include consciousness — it is source data, not
    something this function can strip.
    """
    return np.zeros(np.shape(np.asarray(acvpu_num)), dtype=np.int64)


# Per-vital physiological EWMA defaults (D1/D2): slow-drifting vitals keep more
# memory (lower α / longer ref); fast vitals adapt quickly. Used by the "physio"
# temporal profile; the "global" profile ignores these and uses one α/ref.
PHYSIO_ALPHA = {"heart_rate": 0.3, "blood_pressure": 0.15, "temperature": 0.1,
                "respiratory_rate": 0.3, "oxygen_saturation": 0.3,
                "inspired_oxygen": 0.3, "acvpu": 0.5}
PHYSIO_REF   = {"heart_rate": 360.0, "blood_pressure": 720.0, "temperature": 720.0,
                "respiratory_rate": 360.0, "oxygen_saturation": 360.0,
                "inspired_oxygen": 360.0, "acvpu": 360.0}
# Matches app/streamlit_app.py TemporalConfig defaults (ewma_ref_minutes, window_hours):
# the "main system" is the canonical temporal formula (see temporal_score below), so
# the engine's default reference spacing tracks it rather than the old 360min value.
EWMA_REF_DEFAULT     = 60.0
WINDOW_HOURS_DEFAULT = 24.0

# Vitals that receive the EWMA + worsening-trend adjustment in temporal_score. Defined
# ONCE here: it used to be copied into nine analysis scripts, which is how the app and
# the pipeline silently drifted apart on inspired oxygen.
#
# inspired_oxygen IS included (decision of 2026-08-12): it is a single vital measured in
# one of two units, and the per-row score is already built with the membership function
# for that row's unit, so the score series a patient's EWMA/trend runs over follows the
# unit automatically — there is never more than one oxygen signal in play at a time.
# It was previously excluded as a "categorical step-signal", which described the old
# 5-level INSP_O2_CAT ramp rather than the continuous per-unit membership functions now
# used. app/streamlit_app.py has always included it; this makes the pipeline agree.
#
# acvpu does not appear here because it is not a scored vital at all — see ACVPU_BONUS.
TEMPORAL_VITALS_DEFAULT = {"heart_rate", "blood_pressure", "temperature",
                           "respiratory_rate", "oxygen_saturation", "inspired_oxygen"}

LABELS_7      = ["Below normal - severe concern", "Below normal - moderate concern",
                 "Below normal - mild concern", "No concern",
                 "Above normal - mild concern", "Above normal - moderate concern",
                 "Above normal - severe concern"]
LABELS_3_DOWN = ["Below normal - severe concern", "Below normal - moderate concern",
                 "Below normal - mild concern", "No concern"]
LABELS_3_UP   = ["No concern", "Above normal - mild concern",
                 "Above normal - moderate concern", "Above normal - severe concern"]

OUTPUT_MF = {"No concern": (-0.5, 0, 0, 0.75), "Mild concern": (0.25, 1, 1, 1.75),
             "Moderate concern": (1.25, 2, 2, 2.75), "Severe concern": (2.25, 3, 3, 3.5)}
_OX = np.arange(0, 3.01, 0.01)
_OGRID = {lbl: np.array([(1.0 if b <= x <= c else
                          (0.0 if x <= a or x >= d else
                           (x-a)/(b-a) if a < x < b else (d-x)/(d-c))) for x in _OX])
          for lbl, (a, b, c, d) in OUTPUT_MF.items()}
# Minimum firing strength: memberships below this are pruned before defuzzification, so
# the overlapping edges of the sigmoid sets don't give a normal input a small non-zero
# centroid. This is what makes the A1 exact-zero rule below fire for normal readings.
# Briefly set to 0.0 on 2026-08-12 and restored the same day — with the gate off, every
# vital's most-normal value floors at ~0.25-0.28 and a fully normal patient totals ~1.96
# instead of ~0.48, which is the pre-A1 behaviour the zero rule exists to prevent.
# Consequence to be aware of: pruning is why the FiO2 score is exactly 0 across 21-24%
# (mild concern is only 0.003 at 21% and 0.040 at 24%) and then steps to 0.43 at 25%.
# Must stay in sync with app/streamlit_app.py and app/grid_search_auroc.py.
MIN_FIRING = 0.05


# ── A1: defuzz centroid with the canonical exact-zero rule ───────────────────
def defuzz_centroid(concern: dict) -> float:
    c = {k: (v if v >= MIN_FIRING else 0.0) for k, v in concern.items()}
    # exact zero when only "No concern" remains active
    if c.get("No concern", 0.0) > 0 and all(
            (lvl == "No concern") or (f == 0.0) for lvl, f in c.items()):
        return 0.0
    agg = np.zeros(len(_OX))
    for lvl, f in c.items():
        if f > 0:
            np.maximum(agg, np.minimum(f, _OGRID[lvl]), out=agg)
    s = agg.sum()
    return 0.0 if s == 0 else float(np.dot(_OX, agg) / s)


def _map_concern(memb: dict) -> dict:
    """Collapse linguistic memberships → 4 concern levels (max within group)."""
    con = {"No concern": 0.0, "Mild concern": 0.0,
           "Moderate concern": 0.0, "Severe concern": 0.0}
    for k, v in memb.items():
        kl = k.lower()
        if   "severe"   in kl: con["Severe concern"]   = max(con["Severe concern"],   v)
        elif "moderate" in kl: con["Moderate concern"] = max(con["Moderate concern"], v)
        elif "mild"     in kl: con["Mild concern"]     = max(con["Mild concern"],     v)
        else:                  con["No concern"]        = max(con["No concern"],       v)
    return con


def _merge_sbp_no_concern(memb: dict) -> dict:
    """The ONLY systolic-BP model, aligned with NEWS-2 and clinical judgement: fold
    No concern + Above-mild + Above-moderate into one wider No concern set (sum,
    partition of unity preserved), leaving Above-severe untouched so it overlaps the
    widened No concern set directly instead of sitting behind a mild/moderate buffer.

    SBP therefore has five sets, not seven:
        Below normal - severe / moderate / mild concern, No concern,
        Above normal - severe concern

    Mirrors app/streamlit_app.py's custom_mf_sbp_merged. The earlier "sharper" variant
    (hypertension collapsed into a capped Mild bucket) was removed on 2026-08-12 — the
    merged set is the one in use and there is no longer an alternative to select."""
    m = dict(memb)
    no_con = m.get("No concern", 0.0)
    a_mild = m.pop("Above normal - mild concern", 0.0)
    a_mod  = m.pop("Above normal - moderate concern", 0.0)
    m["No concern"] = min(1.0, no_con + a_mild + a_mod)
    return m


# LUT sampling step for vitals whose membership CSV grid is coarser than the values
# that actually occur in the data. The LUT defuzzifies at each sampled point and
# interpolates the SCORE between them, but defuzzification is non-linear, so a coarse
# grid disagrees with true inference (interpolate memberships, then defuzzify — what
# app/streamlit_app.py does) at any input between grid points. The supplementary-O2
# CSV is on a 1 L/min grid while the data records 0.5 L/min, which put the engine
# 0.016 below the app there; sampling finely closes it. Vitals not listed keep their
# CSV grid, where every observed value already lands on a grid point.
LUT_GRID_STEP = {SUPP_O2_LMIN: 0.1}


def build_lut(vital: str):
    """Input value → defuzzified 0-3 score LUT (bakes in the A1 zero-rule).

    blood_pressure always goes through the five-set merge (``_merge_sbp_no_concern``).
    There is no variant to choose: callers used to pass ``sharper_sbp=True`` for an
    asymmetric alternative, which no longer exists.
    """
    df = pd.read_csv(SIGMOID_DIR / MF_FILE[vital])
    x  = df["Value"].values.astype(float)
    labels = {"7var": LABELS_7, "3var_down": LABELS_3_DOWN, "3var_up": LABELS_3_UP}[VITAL_TYPE[vital]]
    step = LUT_GRID_STEP.get(vital)
    xs = np.round(np.arange(x[0], x[-1] + step / 2.0, step), 6) if step else x
    scores = []
    for v in xs:
        memb = {lab: float(np.interp(v, x, df[lab].values)) for lab in labels}
        if vital == "blood_pressure":
            memb = _merge_sbp_no_concern(memb)
        scores.append(defuzz_centroid(_map_concern(memb)))
    return xs, np.array(scores)


def apply_luts(df: pd.DataFrame, luts: dict, vitals) -> dict:
    out = {}
    for v in vitals:
        col = df[VITAL_COL[v]].values.astype(np.float64)
        x, y = luts[v]
        out[v] = np.interp(np.clip(col, x[0], x[-1]), x, y).astype(np.float32)
    return out


# ── Time-decay EWMA (D1 per-vital α, D2 per-vital ref) ───────────────────────
def _ewma_group(times, raw, alpha, ref, seed=None):
    """seed (D-/issue #6): EWMA initial value. ``None`` keeps the original ew[0]=raw[0]
    (so the first observation has zero excess); a number (e.g. 0.0, population-normal
    under the A1 zero-rule) makes a patient's very first deranged reading already show
    excess — fixing the ≤2–3-obs admissions where the EWMA never builds a baseline."""
    ew = np.empty_like(raw, dtype=np.float64)
    if seed is None:
        ew[0] = raw[0]
        start = 1
    else:
        # treat the seed as the running baseline 'before' the first observation, then
        # let the first observation update it with a full-step weight (dt→ref).
        prev = float(seed)
        a0 = 1.0 - (1.0 - alpha)              # one ref-step of decay onto the seed
        ew[0] = a0 * raw[0] + (1.0 - a0) * prev
        start = 1
    for i in range(start, len(raw)):
        dt = max(float(times[i] - times[i-1]), 0.0)
        a  = 1.0 - (1.0 - alpha) ** (dt / ref)
        ew[i] = a * raw[i] + (1.0 - a) * ew[i-1]
    return ew


def compute_ewma(times, pv, gs, ge, vitals, alphas: dict, refs: dict, seed=None) -> dict:
    """seed: scalar EWMA initialiser shared by all vitals/groups (see _ewma_group).
    Default ``None`` reproduces the original raw[0] seeding exactly."""
    out = {}
    for v in vitals:
        raw = pv[v].astype(np.float64)
        ew  = np.empty(len(raw), np.float64)
        a, r = alphas[v], refs[v]
        for g in range(len(gs)):
            s, e = gs[g], ge[g]
            ew[s:e] = _ewma_group(times[s:e], raw[s:e], a, r, seed)
        out[v] = ew
    return out


# ── B1: aggregation methods ───────────────────────────────────────────────────
def aggregate(adj_stack: np.ndarray, method: str, gamma: float, power: float = 2.0):
    """adj_stack: (n_rows, n_vitals) of temporal-adjusted 0-3 scores."""
    n = adj_stack.shape[1]
    if method == "multiplicative":
        norm = np.clip(adj_stack / 3.0, 0.0, 1.0)
        base = (1.0 - np.prod(1.0 - norm, axis=1)) * 3.0 * n
    elif method == "nonlinear":
        norm = np.clip(adj_stack / 3.0, 0.0, 1.0)
        base = (np.mean(norm ** power, axis=1) ** (1.0 / power)) * 3.0 * n
    else:  # additive
        base = adj_stack.sum(axis=1)
    if gamma >= 1.0:
        return base.astype(np.float32)
    max_v = adj_stack.max(axis=1)
    return ((1.0 - gamma) * (n * max_v) + gamma * base).astype(np.float32)


def _ols_slope_group(times: np.ndarray, raw: np.ndarray, window_minutes: float) -> np.ndarray:
    """OLS slope of raw scores within a time-based look-back window, for every
    observation in one patient's timeline (streamlit_app.py ``_linear_slope``,
    vectorised into a sliding window via a two-pointer scan)."""
    n = len(times)
    slopes = np.zeros(n, dtype=np.float64)
    if n < 2:
        return slopes
    left = 0
    for right in range(n):
        while left < right and (times[right] - times[left]) > window_minutes:
            left += 1
        if right - left + 1 < 2:
            continue
        t_slice = times[left:right + 1]
        s_slice = raw[left:right + 1]
        t_h = (t_slice - t_slice[0]) / 60.0
        mean_t = t_h.mean(); mean_s = s_slice.mean()
        dt = t_h - mean_t
        ss_tt = (dt * dt).sum()
        if ss_tt == 0:
            continue
        slopes[right] = (dt * (s_slice - mean_s)).sum() / ss_tt
    return slopes


def compute_slopes(times, pv, gs, ge, vitals, window_hours: float = WINDOW_HOURS_DEFAULT) -> dict:
    """OLS trend slope of RAW (non-EWMA) per-vital scores within a look-back window,
    for every observation — streamlit_app.py Step 2 input. Parameter-independent of
    α/β/γ, so callers should compute this once and reuse it across a grid search."""
    window_minutes = window_hours * 60.0
    out = {}
    for v in vitals:
        raw = pv[v].astype(np.float64)
        slopes = np.empty(len(raw), np.float64)
        for g in range(len(gs)):
            s, e = gs[g], ge[g]
            slopes[s:e] = _ols_slope_group(times[s:e], raw[s:e], window_minutes)
        out[v] = slopes
    return out


# Dead zone + persistence gate on the worsening-trend factor. Without a floor, a
# single noisy uptick (slope > 0 from one blip) already saturates trend_factor at
# high β and pushes even normal-looking patients toward the ceiling — this is why
# raising β made discrimination worse instead of better (it just made the trigger
# fire faster on noise, not more selective about real deterioration). MIN_SLOPE is
# a placeholder threshold (score-points/hour), not yet tuned against data — a
# starting point for future grid search. REQUIRE_CONSECUTIVE additionally requires
# the slope to clear the dead zone on two consecutive observations, not just one,
# before the trend counts as "worsening".
TREND_MIN_SLOPE_DEFAULT = 0.05
TREND_REQUIRE_CONSECUTIVE_DEFAULT = True


def sustained_slope_mask(slopes: np.ndarray, gs: np.ndarray, ge: np.ndarray,
                         min_slope: float = TREND_MIN_SLOPE_DEFAULT) -> np.ndarray:
    """True at row i iff slope[i] > min_slope AND the immediately preceding
    observation in the SAME patient group also cleared min_slope — requires the
    rise to persist across two consecutive readings, not just one noisy blip,
    before trend_factor is allowed to fire. A group's first observation can never
    be True (no prior reading within the stay to confirm against)."""
    pos = np.asarray(slopes) > min_slope
    mask = np.zeros(len(slopes), dtype=bool)
    for s, e in zip(gs, ge):
        if e - s < 2:
            continue
        seg = pos[s:e]
        mask[s + 1:e] = seg[1:] & seg[:-1]
    return mask


def trend_factor(slopes: np.ndarray, beta: float, min_slope: float = 0.0,
                 sustained: np.ndarray = None) -> np.ndarray:
    """Sigmoid worsening-trend factor (streamlit_app.py): 0 when slope<=min_slope,
    else 2/(1+e^(-β·(slope-min_slope))) − 1, saturating to 1 as the slope steepens
    past the dead zone. ``min_slope=0.0`` (the default here) reproduces the
    original formula exactly; ``temporal_score`` applies the non-zero default via
    ``TREND_MIN_SLOPE_DEFAULT``. ``sustained``, if given (see
    ``sustained_slope_mask``), additionally zeroes any row where the trend hasn't
    persisted across two consecutive observations."""
    tf = np.zeros_like(slopes, dtype=np.float64)
    pos = slopes > min_slope
    if sustained is not None:
        pos = pos & sustained
    if pos.any() and beta > 0:
        ex = np.exp(np.clip(-beta * (slopes[pos] - min_slope), -700, 700))
        tf[pos] = 2.0 / (1.0 + ex) - 1.0
    return tf


def temporal_score(pv, ewma, slopes, vitals, beta, gamma, method="additive", power=2.0,
                   temporal_vitals=None, acvpu_raw=None, gs=None, ge=None,
                   min_slope=TREND_MIN_SLOPE_DEFAULT,
                   require_consecutive=TREND_REQUIRE_CONSECUTIVE_DEFAULT,
                   return_components=False, sustained_masks=None):
    """Two-step per-vital temporal adjustment — EWMA memory + worsening-trend factor —
    reproducing app/streamlit_app.py's canonical formula exactly (single source of
    truth for both the interactive app and the validation pipeline):

      Step 1  base = max(EWMA(raw), raw)          EWMA smoothing must not lower concern.
      Step 2  adjusted = base + f·(3 − base)       f = trend_factor(OLS slope of raw
                                                    over the look-back window, β),
                                                    gated by a dead zone (min_slope)
                                                    and, if gs/ge are given, a
                                                    two-consecutive-reading
                                                    persistence check.

    adjusted is structurally in [base, 3] ⊇ [raw, 3] (f ∈ [0,1)), so the aggregated
    total is always ≥ the snapshot total — no separate floor is needed.

    ``slopes``: {vital: array} from ``compute_slopes`` (parameter-independent, so
    callers should precompute it once outside a grid search).

    temporal_vitals: optional collection of vital names that receive the EWMA +
    trend adjustment. Vitals NOT in this set stay at their snapshot (raw) value —
    used to exclude categorical step-signals (inspired_oxygen, acvpu) whose
    transitions create artefactual EWMA/trend excursions. ``None`` adjusts every
    vital in ``vitals``.

    acvpu_raw: optional raw ACVPU_NUM ordinal array (0-3, Alert=0). When given,
    the score is aggregated as normal, then ``ACVPU_BONUS`` is added on top for
    every row where ACVPU is non-Alert (see ``apply_acvpu_bonus``).

    gs/ge: optional patient group boundaries (from ``group_boundaries``). Required
    for the ``require_consecutive`` persistence gate (see ``sustained_slope_mask``);
    without them the dead zone (``min_slope``) still applies, but the trend can
    fire off a single observation rather than needing two consecutive ones.
    min_slope=0.0 and require_consecutive=False reproduce the original (pre-gate)
    formula exactly.

    return_components: if True, also return the per-vital adjusted scores (the
    ``adj_stack`` columns fed into ``aggregate``, i.e. each vital's own temporal
    score before combination and before the ACVPU bonus) as {vital: array},
    alongside the aggregated total.

    sustained_masks: optional {vital: bool array} precomputed by
    ``sustained_slope_mask``. The mask depends only on ``slopes`` and
    ``min_slope`` — not on α/β/γ — so a grid search calling ``temporal_score``
    hundreds of times should compute it once per vital and pass it here rather
    than paying the persistence-gate's per-patient Python loop on every call.
    Ignored when ``require_consecutive`` is False or ``gs``/``ge`` are absent.
    """
    cols = []
    for v in vitals:
        raw = pv[v].astype(np.float64)
        if temporal_vitals is not None and v not in temporal_vitals:
            cols.append(raw.astype(np.float32))          # categorical: snapshot only
            continue
        base = np.maximum(ewma[v], raw)
        sustained = None
        if require_consecutive and gs is not None and ge is not None:
            sustained = (sustained_masks[v] if sustained_masks is not None
                        else sustained_slope_mask(slopes[v], gs, ge, min_slope))
        f = trend_factor(slopes[v], beta, min_slope=min_slope, sustained=sustained)
        adjusted = base + f * (3.0 - base)
        cols.append(np.clip(adjusted, 0.0, 3.0).astype(np.float32))
    adj_stack = np.column_stack(cols)
    total = aggregate(adj_stack, method, gamma, power)
    if acvpu_raw is not None:
        total = apply_acvpu_bonus(total, acvpu_raw)
    if return_components:
        components = {v: cols[i] for i, v in enumerate(vitals)}
        return total, components
    return total


def snapshot_score(pv, vitals, method="additive", gamma=1.0, power=2.0, acvpu_raw=None):
    """acvpu_raw: optional raw ACVPU_NUM ordinal array (0-3, Alert=0) — score is
    calculated as normal, then ``ACVPU_BONUS`` added on top for non-Alert rows."""
    adj_stack = np.column_stack([pv[v] for v in vitals]).astype(np.float64)
    total = aggregate(adj_stack, method, gamma, power)
    if acvpu_raw is not None:
        total = apply_acvpu_bonus(total, acvpu_raw)
    return total


# ── Grouping + AUROC helpers ──────────────────────────────────────────────────
def group_boundaries(ids):
    ch = np.empty(len(ids), bool); ch[0] = True; ch[1:] = ids[1:] != ids[:-1]
    st = np.where(ch)[0]
    return st, np.append(st[1:], len(ids))


def pools(d, i, e, target):
    if target == "death": return d == 1, (d == 0) & (i == 0)
    if target == "icu":   return i == 1, (i == 0) & (d == 0)
    return e == 1, e == 0


def auroc(d, i, e, score, target):
    pm, nm = pools(d, i, e, target)
    keep = pm | nm
    y = pm[keep].astype(np.int8); s = score[keep]; m = np.isfinite(s)
    if y[m].sum() == 0 or y[m].sum() == m.sum():
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def patient_peak(score, gs):
    return np.maximum.reduceat(score, gs)


def patient_aggregate(score, gs, ge, method="peak", times=None,
                      threshold=None, lead_minutes=240.0):
    """Reduce a row-level score to one value per patient (issue #3).

    ``peak`` (default) reproduces patient_peak and discards the trajectory — the very
    thing the temporal score adds. The other methods are trajectory-sensitive, so a
    patient who *climbs* to a given level ranks differently from one who is statically
    there, which is what lets temporal decouple from snapshot:

      peak               max over the stay (baseline).
      mean               mean over the stay.
      area               time-integrated burden ∫score dt (needs ``times``); falls
                         back to the row-sum when times are absent.
      pre_peak_slope     rise rate to the peak: (peak − first) / (t_peak − t_first).
      time_above         minutes spent at/above ``threshold`` (needs times+threshold).
      score_at_lead      score at the last observation ≥ ``lead_minutes`` before the
                         peak — the early-warning value rather than the peak itself.
    """
    n = len(gs)
    if method == "peak":
        return np.maximum.reduceat(score, gs)
    out = np.empty(n, np.float64)
    for g in range(n):
        s, e = gs[g], ge[g]
        sc = score[s:e].astype(np.float64)
        tt = None if times is None else times[s:e].astype(np.float64)
        if method == "mean":
            out[g] = sc.mean()
        elif method == "area":
            out[g] = (_trapz(sc, tt) if tt is not None and len(sc) > 1 else sc.sum())
        elif method == "pre_peak_slope":
            k = int(np.argmax(sc))
            if tt is None or k == 0 or (tt[k] - tt[0]) <= 0:
                out[g] = 0.0
            else:
                out[g] = (sc[k] - sc[0]) / (tt[k] - tt[0])
        elif method == "time_above":
            if tt is None or threshold is None or len(sc) < 2:
                out[g] = float(np.sum(sc >= (threshold or np.inf)))
            else:
                above = (sc >= threshold).astype(np.float64)
                # minutes above threshold via piecewise-constant (left) integration
                out[g] = float(np.sum(above[:-1] * np.diff(tt)))
        elif method == "score_at_lead":
            k = int(np.argmax(sc))
            if tt is None:
                out[g] = sc[max(0, k - 1)]
            else:
                cutoff = tt[k] - lead_minutes
                pre = np.where(tt[: k + 1] <= cutoff)[0]
                out[g] = sc[pre[-1]] if len(pre) else sc[0]
        else:
            raise ValueError(f"unknown patient_aggregate method: {method}")
    return out
