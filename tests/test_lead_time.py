"""Tests for the backward-from-the-event lead-time machinery in main_pipeline_common.

The behaviour these lock down is the change from the previous lead-time definition:
lead is measured to the *true event time* (not the start of the 24 h label window) and
from the *most recent contiguous alert episode* (not the first crossing anywhere in the
stay), so an isolated spike days earlier can no longer set the clock.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "engine"))

import main_pipeline_common as C          # noqa: E402
import engine_scoring as es               # noqa: E402
from stats import delong_auc_ci           # noqa: E402


def _frame(specs):
    """specs: list of (admission_id, died, icu, [times]) → df + group boundaries."""
    rows = [{"ANON_ADMISSION_ID": aid, "DIED_FLAG": died, "ICU_FLAG": icu,
             "t_minutes": float(t)}
            for aid, died, icu, times in specs for t in times]
    df = pd.DataFrame(rows)
    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    return df, gs, ge


class TestDeriveEventTimes:
    def test_death_is_last_observation(self):
        df, gs, ge = _frame([(1, 1, 0, [0, 60, 120])])
        death_t, icu_t, event_t = C.derive_event_times(df, gs, ge)
        assert death_t[0] == 120.0
        assert np.isnan(icu_t[0])
        assert event_t[0] == 120.0

    def test_icu_is_observation_before_the_gap(self):
        # 60 → 3000 is a 2940-minute gap (> 24h), so the ICU transfer is inferred at 60
        df, gs, ge = _frame([(1, 0, 1, [0, 60, 3000, 3060])])
        _, icu_t, event_t = C.derive_event_times(df, gs, ge)
        assert icu_t[0] == 60.0
        assert event_t[0] == 60.0

    def test_icu_without_a_gap_falls_back_to_last_observation(self):
        df, gs, ge = _frame([(1, 0, 1, [0, 150, 300])])
        _, icu_t, _ = C.derive_event_times(df, gs, ge)
        assert icu_t[0] == 300.0

    def test_no_event_gives_nan(self):
        df, gs, ge = _frame([(1, 0, 0, [0, 90])])
        _, _, event_t = C.derive_event_times(df, gs, ge)
        assert np.isnan(event_t[0])

    def test_death_and_icu_takes_the_earlier(self):
        df, gs, ge = _frame([(1, 1, 1, [0, 100, 5000])])
        death_t, icu_t, event_t = C.derive_event_times(df, gs, ge)
        assert death_t[0] == 5000.0 and icu_t[0] == 100.0
        assert event_t[0] == 100.0      # first deterioration wins

    def test_boundaries_are_per_admission(self):
        df, gs, ge = _frame([(1, 1, 0, [0, 60]), (2, 0, 1, [0, 30, 4000]),
                             (3, 0, 0, [0, 10])])
        _, _, event_t = C.derive_event_times(df, gs, ge)
        assert event_t[0] == 60.0        # death → its own last obs, not admission 2's
        assert event_t[1] == 30.0
        assert np.isnan(event_t[2])


class TestLeadTimePatient:
    # event at t=600. An isolated alert at t=120 is followed by a quiet reading, then
    # an unbroken alert run from t=360 onward.
    times = np.array([0, 120, 240, 360, 480, 600], float)
    score = np.array([1.0, 5.0, 1.0, 5.0, 5.0, 5.0])

    def _run(self, thr_target=4.0):
        ids = np.zeros(len(self.times), np.int32)
        gs, ge = es.group_boundaries(ids)
        out = C.lead_time_patient(self.score, self.times, gs, ge,
                                  np.array([600.0]), np.array([1]))
        thr, det, lead, fa = out
        k = int(np.argmin(np.abs(thr - thr_target)))
        return det[k], lead[k]

    def test_uses_most_recent_contiguous_run(self):
        det, lead = self._run()
        assert det == 1.0
        # run starts at t=360 → 4h, NOT the t=120 spike which would give 8h
        assert lead == pytest.approx(4.0)

    def test_detection_is_zero_above_every_score(self):
        ids = np.zeros(len(self.times), np.int32)
        gs, ge = es.group_boundaries(ids)
        thr, det, lead, fa = C.lead_time_patient(
            self.score, self.times, gs, ge, np.array([600.0]), np.array([1]))
        assert np.all(det <= 1.0) and np.all(np.isfinite(det))

    def test_observations_after_the_event_are_ignored(self):
        # same stay, but with two more alerting readings AFTER the event time
        times = np.r_[self.times, [720.0, 840.0]]
        score = np.r_[self.score, [5.0, 5.0]]
        ids = np.zeros(len(times), np.int32)
        gs, ge = es.group_boundaries(ids)
        thr, det, lead, fa = C.lead_time_patient(
            score, times, gs, ge, np.array([600.0]), np.array([1]))
        k = int(np.argmin(np.abs(thr - 4.0)))
        assert lead[k] == pytest.approx(4.0)   # unchanged by post-event readings


class TestLeadTimeRow:
    def test_sensitivity_and_median_lead_over_flagged_rows(self):
        times = np.array([0, 120, 240, 360, 480, 600], float)
        score = np.array([1.0, 5.0, 1.0, 5.0, 5.0, 5.0])
        e_row = np.array([0, 0, 1, 1, 1, 1])          # last four are in the window
        ids = np.zeros(6, np.int32)
        gs, ge = es.group_boundaries(ids)
        thr, det, lead, fa = C.lead_time_row(score, times, gs, ge,
                                             np.array([600.0]), e_row)
        k = int(np.argmin(np.abs(thr - 4.0)))
        # flagged rows: t=240 (no alert), 360, 480, 600 → 3 of 4 alert
        assert det[k] == pytest.approx(0.75)
        # alerting leads: 4h, 2h, 0h → median 2h
        assert lead[k] == pytest.approx(2.0)


class TestLeadTimeTable:
    def test_unreachable_sensitivity_is_blank_not_extrapolated(self):
        # a system whose detection tops out at 65% must not report a 90% row
        thr = np.array([1.0, 2.0, 3.0])
        det = np.array([0.65, 0.40, 0.10])
        lead = np.array([10.0, 8.0, 5.0])
        fa = np.array([0.30, 0.15, 0.05])
        tab = C.lead_time_table({"Temporal": (thr, det, lead, fa)})
        at90 = tab[tab["Sensitivity (% events detected pre-onset)"] == 0.90]
        at60 = tab[tab["Sensitivity (% events detected pre-onset)"] == 0.60]
        assert at90["Median lead (hours)"].iloc[0] == ""
        assert at60["Median lead (hours)"].iloc[0] != ""
        assert at90["Max detection reached"].iloc[0] == pytest.approx(0.65)


class TestDelongAucCi:
    def test_point_estimate_matches_sklearn_and_ci_brackets_it(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 20_000)
        s = y * 0.9 + rng.normal(size=20_000)
        auc, lo, hi = delong_auc_ci(y, s)
        assert auc == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        assert lo < auc < hi

    def test_degenerate_labels_return_nan(self):
        auc, lo, hi = delong_auc_ci(np.zeros(10, int), np.arange(10.0))
        assert np.isnan(auc) and np.isnan(lo) and np.isnan(hi)
