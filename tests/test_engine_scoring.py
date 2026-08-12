"""Test suite for the Fuzzy EWS engine.

Locks the behaviour the engine must preserve:

  * A1 zero-rule: a perfectly normal vital defuzzes to exactly 0.
  * temporal_vitals exclusion: categorical vitals get no EWMA/trend adjustment.
  * EWMA recovers raw when alpha=1 and is bounded by the running raw extremes.
  * OLS trend slope: sign/magnitude matches a plain per-patient regression.
  * sigmoid trend factor: matches app/streamlit_app.py's formula exactly, and the
    resulting temporal total is structurally >= the snapshot total (no explicit
    floor needed, unlike the old excess-EWMA design).
  * REGRESSION: engine.temporal_score reproduces app/streamlit_app.py's per-patient
    EWMA + sigmoid worsening-trend maths bit-for-bit (this is the point of unifying
    the engine onto the app's formula — see engine_scoring.py module docstring).

Run:  pytest tests/ -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))
import engine_scoring as es

VITALS = es.VITALS_BASE                      # 6 vitals
TV = {"heart_rate", "blood_pressure", "temperature",
      "respiratory_rate", "oxygen_saturation"}   # boosted (categorical excluded)


# ── helpers ──────────────────────────────────────────────────────────────────
def _synthetic(n_pat=40, seed=0):
    """Build a small synthetic pv / ewma / slopes / grouping set resembling the pipeline."""
    rng = np.random.default_rng(seed)
    rows, gs = [], [0]
    times = []
    for p in range(n_pat):
        k = int(rng.integers(1, 12))
        t = 0.0
        for j in range(k):
            rows.append(p)
            t += float(rng.uniform(30, 600))   # strictly increasing, as the sorted pipeline produces
            times.append(t)
        gs.append(gs[-1] + k)
    n = len(rows)
    pv = {v: rng.uniform(0.0, 3.0, n).astype(np.float32) for v in VITALS}
    gs_arr = np.array(gs[:-1]); ge_arr = np.append(gs_arr[1:], n)
    t = np.array(times, np.float64)
    alphas = {v: 0.3 for v in VITALS}; refs = {v: es.EWMA_REF_DEFAULT for v in VITALS}
    ewma = es.compute_ewma(t, pv, gs_arr, ge_arr, VITALS, alphas, refs)
    slopes = es.compute_slopes(t, pv, gs_arr, ge_arr, VITALS)
    return pv, ewma, slopes, gs_arr, ge_arr, t


def _reference_ewma(values, times_min, alpha, ref_minutes):
    """Verbatim port of app/streamlit_app.py's ``_ewma`` (one patient, one vital)."""
    result = [values[0]]
    for i in range(1, len(values)):
        dt = max(float(times_min[i] - times_min[i - 1]), 0.0)
        alpha_eff = alpha if dt <= 0 else 1.0 - (1.0 - alpha) ** (dt / ref_minutes)
        result.append(alpha_eff * values[i] + (1.0 - alpha_eff) * result[-1])
    return result


def _reference_slope(times_hours, values):
    """Verbatim port of app/streamlit_app.py's ``_linear_slope``."""
    n = len(times_hours)
    if n < 2:
        return 0.0
    mean_t = sum(times_hours) / n; mean_v = sum(values) / n
    ss_tt = sum((t - mean_t) ** 2 for t in times_hours)
    if ss_tt == 0:
        return 0.0
    ss_tv = sum((t - mean_t) * (v - mean_v) for t, v in zip(times_hours, values))
    return ss_tv / ss_tt


def _reference_slope_ending_at(times_min, raw, end_idx, window_min):
    """Verbatim port of app/streamlit_app.py's ``_slope_ending_at``."""
    end_t = times_min[end_idx]
    window_raw, window_times = [], []
    for t, s in zip(times_min[:end_idx + 1], raw[:end_idx + 1]):
        if end_t - t <= window_min:
            window_raw.append(s); window_times.append(t)
    if len(window_raw) < 2:
        return 0.0
    t0 = window_times[0]
    window_times_h = [(t - t0) / 60.0 for t in window_times]
    return _reference_slope(window_times_h, window_raw)


def _reference_adjusted(raw, times_min, alpha, beta, ref_minutes, window_hours,
                        min_slope=0.0, require_consecutive=False):
    """Verbatim port of app/streamlit_app.py's ``_compute_temporal_adjusted_scores``
    two-step formula (incl. the dead-zone + persistence gate) for a single vital's
    timeline, returning the adjusted score at the LAST observation only (what the
    app displays as "current"). Defaults reproduce the original (pre-gate) formula
    exactly."""
    ewma_scores = _reference_ewma(raw, times_min, alpha, ref_minutes)
    clamped = [max(e, r) for e, r in zip(ewma_scores, raw)]
    ewma_current = clamped[-1]
    window_min = window_hours * 60.0
    n = len(raw)
    slope = _reference_slope_ending_at(times_min, raw, n - 1, window_min)

    fires = slope > min_slope
    if fires and require_consecutive:
        if n < 2:
            fires = False
        else:
            prev_slope = _reference_slope_ending_at(times_min, raw, n - 2, window_min)
            fires = prev_slope > min_slope

    trend_factor = (2.0 / (1.0 + math.exp(-beta * (slope - min_slope))) - 1.0) if fires else 0.0
    adjusted = ewma_current + trend_factor * (3.0 - ewma_current)
    return max(0.0, min(3.0, adjusted))


# ── A1 zero-rule ─────────────────────────────────────────────────────────────
def test_normal_vital_defuzzes_to_zero():
    z = es.defuzz_centroid({"No concern": 1.0, "Mild concern": 0.0,
                            "Moderate concern": 0.0, "Severe concern": 0.0})
    assert z == 0.0


# HR / SBP / temperature / SpO2 / inspired_oxygen have a value where ONLY "No concern"
# fires once MIN_FIRING prunes the overlapping edges, so the A1 exact-zero rule triggers.
# respiratory_rate (0.37) keeps an adjacent mild membership >= MIN_FIRING even at its
# most-normal value (17/min), so it floors above 0 — real engine behaviour worth locking.
def test_lut_reaches_exact_zero_where_expected():
    for v in ["heart_rate", "blood_pressure", "temperature", "oxygen_saturation",
              "inspired_oxygen"]:
        _, y = es.build_lut(v)
        assert y.min() == pytest.approx(0.0, abs=1e-9), f"{v} should reach exactly 0"


def test_lut_normal_region_is_low():
    for v in VITALS:
        _, y = es.build_lut(v)
        assert 0.0 <= y.min() < 0.5, f"{v} normal floor {y.min():.3f} out of expected range"


# The FiO2 LUT is exactly 0 across 21-24%: MIN_FIRING (0.05) prunes the mild-concern
# membership there (0.003 at 21% rising to only 0.040 at 24%), leaving "No concern"
# alone active. That reproduces the expert-drawn crisp set in
# membership_functions/original/ (No concern spans 21-24, mild starts at 25) — so it
# is by design, not a rounding artefact. Locked here because it looks like a bug, and
# because it is the behaviour that returns whenever MIN_FIRING is restored.
def test_fio2_lut_zero_across_no_concern_set():
    x, y = es.build_lut("inspired_oxygen")
    for v in (21, 22, 23, 24):
        assert np.interp(v, x, y) == pytest.approx(0.0, abs=1e-12), f"FiO2 {v}% should be 0"
    assert np.interp(25, x, y) > 0.4          # first value where mild clears MIN_FIRING


# ── inspired oxygen recorded in mixed units ──────────────────────────────────
def test_split_inspired_oxygen_does_not_interconvert():
    fio2, flow = es.split_inspired_oxygen([21, 40, 0.5, 4, None],
                                          ["%", "%", "litres", "litres", None])
    assert np.array_equal(fio2[:2], [21.0, 40.0])
    assert np.isnan(fio2[2]) and np.isnan(fio2[3])      # flow rows carry no FiO2
    assert np.array_equal(flow[2:4], [0.5, 4.0])        # flow kept in L/min
    assert np.isnan(flow[0]) and np.isnan(flow[1])
    assert fio2[4] == 21.0 and np.isnan(flow[4])        # unknown unit → room air


def test_inspired_oxygen_concern_uses_the_matching_membership_function():
    fio2_lut, lmin_lut = es.build_lut("inspired_oxygen"), es.build_lut(es.SUPP_O2_LMIN)
    fio2, flow = es.split_inspired_oxygen([21, 40, 0.5, 4], ["%", "%", "litres", "litres"])
    got = es.inspired_oxygen_concern(fio2, flow, fio2_lut, lmin_lut)
    assert got[0] == pytest.approx(np.interp(21, *fio2_lut), abs=1e-9)   # room air
    assert got[1] == pytest.approx(np.interp(40, *fio2_lut), rel=1e-6)
    assert got[2] == pytest.approx(np.interp(0.5, *lmin_lut), rel=1e-6)
    assert got[3] == pytest.approx(np.interp(4.0, *lmin_lut), rel=1e-6)
    # NOT the old pseudo-FiO2 route: 4 L/min must not score as 21+4*4 = 37%
    assert got[3] != pytest.approx(np.interp(37.0, *fio2_lut), rel=1e-3)


def test_inspired_oxygen_concern_flow_is_monotone_and_bounded():
    fio2_lut, lmin_lut = es.build_lut("inspired_oxygen"), es.build_lut(es.SUPP_O2_LMIN)
    flow = np.arange(0.5, 20.5, 0.5)
    got = es.inspired_oxygen_concern(np.full(flow.shape, np.nan), flow, fio2_lut, lmin_lut)
    assert np.all(np.diff(got) >= -1e-6), "concern must not fall as flow rises"
    assert got.min() > 0.0 and got.max() <= 3.0


def test_on_supplemental_oxygen_reads_both_units():
    fio2, flow = es.split_inspired_oxygen([21, 24, 0.5, None], ["%", "%", "litres", None])
    assert list(es.on_supplemental_oxygen(fio2, flow)) == [False, True, True, False]


# ── EWMA ─────────────────────────────────────────────────────────────────────
def test_ewma_alpha_one_recovers_raw():
    pv, _, _, gs, ge, t = _synthetic()
    alphas = {v: 1.0 for v in VITALS}; refs = {v: es.EWMA_REF_DEFAULT for v in VITALS}
    ew = es.compute_ewma(t, pv, gs, ge, VITALS, alphas, refs)
    for v in VITALS:
        assert np.allclose(ew[v], pv[v], atol=1e-6)


def test_ewma_bounded_by_running_extremes():
    pv, ewma, _, gs, ge, t = _synthetic()
    for v in VITALS:
        for g in range(len(gs)):
            s, e = gs[g], ge[g]
            seg_raw, seg_ew = pv[v][s:e], ewma[v][s:e]
            run_min = np.minimum.accumulate(seg_raw)
            run_max = np.maximum.accumulate(seg_raw)
            assert np.all(seg_ew >= run_min - 1e-5)
            assert np.all(seg_ew <= run_max + 1e-5)


def test_seed_zero_gives_first_obs_excess():
    pv, _, _, gs, ge, t = _synthetic()
    alphas = {v: 0.3 for v in VITALS}; refs = {v: es.EWMA_REF_DEFAULT for v in VITALS}
    ew_raw0 = es.compute_ewma(t, pv, gs, ge, VITALS, alphas, refs, seed=None)
    ew_seed = es.compute_ewma(t, pv, gs, ge, VITALS, alphas, refs, seed=0.0)
    v = "heart_rate"
    first = gs                                  # first-row index of each patient
    # seed=None: first-obs excess is exactly 0; seed=0: strictly positive where raw>0
    assert np.allclose((pv[v][first] - ew_raw0[v][first]), 0.0, atol=1e-6)
    raw_first = pv[v][first].astype(np.float64)
    exc_seed = raw_first - ew_seed[v][first]
    assert np.all(exc_seed[raw_first > 0.01] > 0.0)


# ── OLS trend slope ────────────────────────────────────────────────────────────
def test_ols_slope_positive_for_rising_patient():
    times = np.array([0.0, 60.0, 120.0, 180.0])
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    slopes = es._ols_slope_group(times, raw, window_minutes=24 * 60.0)
    # last point's slope should be positive (score/hour), matching a hand regression
    expected = _reference_slope(list((times - times[0]) / 60.0), list(raw))
    assert slopes[-1] == pytest.approx(expected, abs=1e-9)
    assert slopes[-1] > 0


def test_ols_slope_zero_for_flat_patient():
    times = np.array([0.0, 60.0, 120.0, 180.0])
    raw = np.array([2.0, 2.0, 2.0, 2.0])
    slopes = es._ols_slope_group(times, raw, window_minutes=24 * 60.0)
    assert np.allclose(slopes[1:], 0.0, atol=1e-9)


def test_compute_slopes_matches_reference_regression():
    pv, _, slopes, gs, ge, t = _synthetic(seed=3)
    for v in VITALS:
        for g in range(len(gs)):
            s, e = gs[g], ge[g]
            tt, raw = t[s:e], pv[v][s:e].astype(np.float64)
            window_min = es.WINDOW_HOURS_DEFAULT * 60.0
            # reference: full-window regression ending at the LAST point of this segment
            left = 0
            for right in range(len(tt)):
                while left < right and (tt[right] - tt[left]) > window_min:
                    left += 1
            expected = _reference_slope(list((tt[left:] - tt[left]) / 60.0), list(raw[left:])) \
                if len(tt) - left >= 2 else 0.0
            assert slopes[v][e - 1] == pytest.approx(expected, abs=1e-6)


# ── sigmoid trend factor ───────────────────────────────────────────────────────
def test_trend_factor_zero_for_nonpositive_slope():
    slopes = np.array([-1.0, 0.0, -0.001])
    tf = es.trend_factor(slopes, beta=2.0)
    assert np.allclose(tf, 0.0)


def test_trend_factor_zero_when_beta_zero():
    slopes = np.array([0.5, 1.0, 5.0])
    tf = es.trend_factor(slopes, beta=0.0)
    assert np.allclose(tf, 0.0)


def test_trend_factor_matches_streamlit_formula():
    slopes = np.array([0.1, 0.5, 1.0, 2.0])
    beta = 1.5
    tf = es.trend_factor(slopes, beta)
    expected = np.array([2.0 / (1.0 + math.exp(-beta * s)) - 1.0 for s in slopes])
    assert np.allclose(tf, expected, atol=1e-12)
    assert np.all((tf >= 0.0) & (tf < 1.0))


# ── temporal_score ───────────────────────────────────────────────────────────
def test_temporal_at_least_snapshot():
    pv, ewma, slopes, gs, ge, t = _synthetic()
    snap = es.snapshot_score(pv, VITALS)
    temp = es.temporal_score(pv, ewma, slopes, VITALS, beta=2.0, gamma=1.0,
                             temporal_vitals=TV)
    assert np.all(temp >= snap - 1e-5)


def test_temporal_vitals_exclusion():
    pv, ewma, slopes, gs, ge, t = _synthetic()
    # force a big change on a categorical vital; it must NOT get EWMA/trend adjustment
    pv2 = {k: v.copy() for k, v in pv.items()}
    base = es.temporal_score(pv, ewma, slopes, VITALS, 3.0, 1.0, temporal_vitals=TV)
    pv2["inspired_oxygen"] = np.clip(pv2["inspired_oxygen"] + 2.0, 0, 3).astype(np.float32)
    ew2 = {**ewma, "inspired_oxygen": np.zeros_like(ewma["inspired_oxygen"])}
    sl2 = {**slopes, "inspired_oxygen": np.full_like(slopes["inspired_oxygen"], 5.0)}
    bumped = es.temporal_score(pv2, ew2, sl2, VITALS, 3.0, 1.0, temporal_vitals=TV)
    # difference comes only from the snapshot contribution of inspired_oxygen
    expected = base + (pv2["inspired_oxygen"] - pv["inspired_oxygen"])
    assert np.allclose(bumped, expected, atol=1e-4)


def test_beta_zero_equals_ewma_clamped_total():
    pv, ewma, slopes, gs, ge, t = _synthetic()
    temp = es.temporal_score(pv, ewma, slopes, VITALS, beta=0.0, gamma=1.0,
                             temporal_vitals=TV)
    expected = sum(np.maximum(ewma[v], pv[v]) for v in TV) + sum(
        pv[v] for v in VITALS if v not in TV)
    assert np.allclose(temp, expected, atol=1e-4)


def test_beta_zero_equals_snapshot_when_ewma_flat():
    pv, _, slopes, gs, ge, t = _synthetic()
    flat = {v: pv[v].astype(np.float64) for v in VITALS}   # ewma == raw
    snap = es.snapshot_score(pv, VITALS)
    temp = es.temporal_score(pv, flat, slopes, VITALS, beta=0.0, gamma=1.0,
                             temporal_vitals=TV)
    assert np.allclose(temp, snap, atol=1e-4)


# ── REGRESSION: engine reproduces app/streamlit_app.py's per-patient formula ────
@pytest.mark.parametrize("alpha,beta", [(0.7, 2.0), (0.3, 0.5), (1.0, 4.0), (0.5, 0.0)])
def test_engine_matches_streamlit_formula_no_gate(alpha, beta):
    """min_slope=0, require_consecutive=False reproduces the original (pre-gate)
    formula exactly — the backward-compatible path."""
    rng = np.random.default_rng(11)
    ref_minutes = 60.0
    window_hours = 24.0
    for _ in range(5):   # a handful of synthetic single-patient timelines
        n = int(rng.integers(2, 15))
        raw = list(rng.uniform(0.0, 3.0, n))
        times_min = np.cumsum(rng.uniform(20, 400, n)).tolist()

        expected = _reference_adjusted(raw, times_min, alpha, beta, ref_minutes, window_hours,
                                       min_slope=0.0, require_consecutive=False)

        pv = {"heart_rate": np.array(raw, dtype=np.float32)}
        gs = np.array([0]); ge = np.array([n])
        t = np.array(times_min, dtype=np.float64)
        ewma = es.compute_ewma(t, pv, gs, ge, ["heart_rate"],
                               {"heart_rate": alpha}, {"heart_rate": ref_minutes})
        slopes = es.compute_slopes(t, pv, gs, ge, ["heart_rate"], window_hours=window_hours)
        got = es.temporal_score(pv, ewma, slopes, ["heart_rate"], beta, gamma=1.0,
                                min_slope=0.0, require_consecutive=False)
        assert got[-1] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("alpha,beta", [(0.7, 2.0), (0.3, 0.5), (1.0, 4.0), (0.5, 0.0)])
def test_engine_matches_streamlit_formula_default_gate(alpha, beta):
    """The new default (dead zone + 2-consecutive-reading persistence gate)
    matches between engine and app when both are given the same gs/ge."""
    rng = np.random.default_rng(11)
    ref_minutes = 60.0
    window_hours = 24.0
    min_slope = es.TREND_MIN_SLOPE_DEFAULT
    for _ in range(5):
        n = int(rng.integers(2, 15))
        raw = list(rng.uniform(0.0, 3.0, n))
        times_min = np.cumsum(rng.uniform(20, 400, n)).tolist()

        expected = _reference_adjusted(raw, times_min, alpha, beta, ref_minutes, window_hours,
                                       min_slope=min_slope, require_consecutive=True)

        pv = {"heart_rate": np.array(raw, dtype=np.float32)}
        gs = np.array([0]); ge = np.array([n])
        t = np.array(times_min, dtype=np.float64)
        ewma = es.compute_ewma(t, pv, gs, ge, ["heart_rate"],
                               {"heart_rate": alpha}, {"heart_rate": ref_minutes})
        slopes = es.compute_slopes(t, pv, gs, ge, ["heart_rate"], window_hours=window_hours)
        got = es.temporal_score(pv, ewma, slopes, ["heart_rate"], beta, gamma=1.0, gs=gs, ge=ge)
        assert got[-1] == pytest.approx(expected, abs=1e-4)


# ── dead zone + persistence gate ────────────────────────────────────────────
def test_trend_factor_dead_zone_blocks_small_slope():
    slopes = np.array([0.01, 0.04, 0.05, 0.06, 0.2])
    tf = es.trend_factor(slopes, beta=2.0, min_slope=0.05)
    assert np.allclose(tf[:3], 0.0)          # <= min_slope: blocked
    assert np.all(tf[3:] > 0.0)              # > min_slope: fires


def test_trend_factor_dead_zone_continuous_at_threshold():
    """tf starts at exactly 0 right at the threshold (no jump), then ramps up."""
    beta = 2.0
    min_slope = 0.05
    tf_at = es.trend_factor(np.array([min_slope]), beta, min_slope=min_slope)[0]
    assert tf_at == pytest.approx(0.0, abs=1e-12)


def test_sustained_slope_mask_requires_two_consecutive():
    # one patient group of 5 obs; slope pattern: below, above, below, above, above
    slopes = np.array([0.0, 0.2, 0.0, 0.2, 0.2])
    gs = np.array([0]); ge = np.array([5])
    mask = es.sustained_slope_mask(slopes, gs, ge, min_slope=0.05)
    # index 0: no prior obs -> False. index 1: prior(0) below -> False.
    # index 2: below threshold itself -> False. index 3: prior(2) below -> False.
    # index 4: both idx3 and idx4 above -> True.
    assert list(mask) == [False, False, False, False, True]


def test_sustained_slope_mask_respects_group_boundaries():
    # two patients; last obs of patient A and first obs of patient B are adjacent
    # in the array but must NOT be treated as consecutive.
    slopes = np.array([0.2, 0.2, 0.2])
    gs = np.array([0, 2]); ge = np.array([2, 3])
    mask = es.sustained_slope_mask(slopes, gs, ge, min_slope=0.05)
    assert list(mask) == [False, True, False]  # idx 2 is patient B's first obs


def test_temporal_score_require_consecutive_blocks_single_blip():
    """A single noisy positive-slope reading must NOT push the score up when
    require_consecutive=True and gs/ge are supplied."""
    pv = {"heart_rate": np.array([0.0, 0.0, 1.5], dtype=np.float32)}
    gs = np.array([0]); ge = np.array([3])
    t = np.array([0.0, 120.0, 240.0])
    ewma = es.compute_ewma(t, pv, gs, ge, ["heart_rate"], {"heart_rate": 1.0}, {"heart_rate": 60.0})
    slopes = es.compute_slopes(t, pv, gs, ge, ["heart_rate"], window_hours=24.0)
    gated = es.temporal_score(pv, ewma, slopes, ["heart_rate"], beta=5.0, gamma=1.0,
                              gs=gs, ge=ge, min_slope=0.05, require_consecutive=True)
    ungated = es.temporal_score(pv, ewma, slopes, ["heart_rate"], beta=5.0, gamma=1.0,
                                min_slope=0.0, require_consecutive=False)
    # the ungated (old-formula) score should be pushed noticeably higher by the
    # single blip than the gated score, which should stay near the EWMA-clamped base
    assert gated[-1] < ungated[-1]


# ── trajectory aggregates ────────────────────────────────────────────────────
def test_patient_aggregate_peak_matches_patient_peak():
    pv, ewma, slopes, gs, ge, t = _synthetic()
    sc = es.snapshot_score(pv, VITALS)
    assert np.array_equal(es.patient_aggregate(sc, gs, ge, "peak"),
                          es.patient_peak(sc, gs))


@pytest.mark.parametrize("method", ["mean", "area", "pre_peak_slope",
                                    "time_above", "score_at_lead"])
def test_patient_aggregate_methods_run(method):
    pv, ewma, slopes, gs, ge, t = _synthetic()
    sc = es.snapshot_score(pv, VITALS)
    out = es.patient_aggregate(sc, gs, ge, method, times=t, threshold=5.0)
    assert out.shape == (len(gs),)
    assert np.all(np.isfinite(out))


def test_pre_peak_slope_positive_for_rising_patient():
    # one patient, strictly rising score over increasing time → positive slope
    score = np.array([1.0, 2.0, 3.0, 4.0])
    times = np.array([0.0, 60.0, 120.0, 180.0])
    gs = np.array([0]); ge = np.array([4])
    out = es.patient_aggregate(score, gs, ge, "pre_peak_slope", times=times)
    assert out[0] > 0
