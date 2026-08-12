"""Tests for engine.stats (DeLong) and engine.diagnostics (coupling)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))
import stats as st
import diagnostics as dg


def _data(n=600, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    # signal-bearing score: positives shifted up
    a = rng.normal(0, 1, n) + 0.8 * y
    b = a + rng.normal(0, 0.4, n)          # correlated, slightly noisier
    return y, a, b


def test_delong_auc_matches_sklearn():
    y, a, b = _data()
    r = st.delong_roc_test(y, a, b)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, a), abs=1e-9)
    assert r["auc_b"] == pytest.approx(roc_auc_score(y, b), abs=1e-9)


def test_delong_self_comparison_is_null():
    y, a, _ = _data()
    r = st.delong_roc_test(y, a, a.copy())
    assert r["delta"] == pytest.approx(0.0, abs=1e-12)
    assert r["se"] == pytest.approx(0.0, abs=1e-9)
    assert r["p"] == pytest.approx(1.0, abs=1e-9)


def test_delong_detects_real_difference():
    # a clearly separates classes, b is noise → significant difference expected
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.4).astype(int)
    a = rng.normal(0, 1, n) + 1.5 * y
    b = rng.normal(0, 1, n)
    r = st.delong_roc_test(y, a, b)
    assert r["delta"] > 0.2
    assert r["p"] < 1e-6


def test_delong_handles_nan_pairwise():
    y, a, b = _data()
    a[5] = np.nan; b[9] = np.nan
    r = st.delong_roc_test(y, a, b)        # should not raise
    assert np.isfinite(r["p"])


def test_coupling_identical_scores():
    rng = np.random.default_rng(2)
    s = rng.random(500)
    c = dg.coupling_stats(s, s.copy())
    assert c["spearman"] == pytest.approx(1.0, abs=1e-12)
    assert c["pct_identical"] == pytest.approx(100.0)
    assert c["mean_boost"] == pytest.approx(0.0, abs=1e-12)


def test_stratified_auroc_shapes():
    rng = np.random.default_rng(3)
    n = 1000
    y = (rng.random(n) < 0.3).astype(int)
    sc = {"x": rng.random(n) + 0.5 * y, "y": rng.random(n)}
    stay = rng.integers(1, 30, n)
    df = dg.stratified_auroc(y, sc, stay)
    assert set(df["system"]) == {"x", "y"}
    assert df["auroc"].notna().any()
