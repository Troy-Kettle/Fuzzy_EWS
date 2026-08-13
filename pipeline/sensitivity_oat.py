"""
ROW-LEVEL (per-observation) one-at-a-time sensitivity for α, β, γ.

Each parameter is varied over its full grid while the other two are held at the
baseline configuration (the best event point from the patient-level grid search
in results/main_dataset/patient level/grid_search/grid_results.csv):

    α = 0.1   β = 1.0   γ = 1.0

Unlike pipeline/grid_search_main.py, metrics are computed on the raw
observation scores — no per-patient peak reduction — so each row of the dataset
is one sample. Data loading, LUTs, O2 override, patient sampling and the
EWMA + sigmoid-trend scoring path (matching app/streamlit_app.py) are identical
to that script.

Primary metric is AUPRC (average precision); AUROC is reported alongside since
row-level positives are rare and AUPRC is prevalence-dependent.

Output → results/sensitivity_one_at_a_time/sensitivity_one_at_a_time.csv
"""

import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_scoring as es

# Everything shared lives in pipeline/common.py — the grid, the cohort constants, the
# loader and the per-vital scoring. (Previously imported from grid_search_excess_patient,
# which was one of five overlapping grid-search scripts and has been removed.)
from common import (
    ALPHA_VALS, BETA_VALS, GAMMA_VALS, TEMPORAL_VITALS,
    NE_PATIENTS, RANDOM_SEED, load, build_pv,
)

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

OUT_DIR = REPO / "results" / "sensitivity_one_at_a_time"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "sensitivity_one_at_a_time.csv"

# Baseline (best event/death config from the patient-level grid search)
BASE_ALPHA = 0.1
BASE_BETA = 1.0
BASE_GAMMA = 1.0

TARGETS = ["death", "icu", "event"]


def _metrics(d, i, e, score, target):
    """AUPRC + AUROC + prevalence on the target's positive/negative pools."""
    pm, nm = es.pools(d, i, e, target)
    keep = pm | nm
    y = pm[keep].astype(np.int8)
    s = np.asarray(score)[keep]
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan"), float("nan")
    return (float(average_precision_score(y, s)),
            float(roc_auc_score(y, s)),
            float(y.mean()))


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    t_total = time.time()

    df = load()
    vitals = es.VITALS_BASE

    print("Building fuzzy LUTs (engine, five-set SBP)…")
    luts = {v: es.build_lut(v) for v in vitals}

    # Same patient sample as the patient-level grid search
    event_ids = set(df.loc[df["EVENT_FLAG"] == 1, "ANON_ADMISSION_ID"].unique())
    ne_ids = list(set(df["ANON_ADMISSION_ID"].unique()) - event_ids)
    ne_sample = rng.choice(ne_ids, size=min(NE_PATIENTS, len(ne_ids)), replace=False)
    df = df[df["ANON_ADMISSION_ID"].isin(set(event_ids) | set(ne_sample))].copy()
    print(f"  Computation dataset: {len(df):,} rows")

    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    t_arr = df["t_minutes"].values.astype(np.float64)

    pv = build_pv(df, luts, vitals)
    snapshot = sum(pv[v] for v in vitals).astype(np.float32)

    # ROW-LEVEL labels (no per-patient reduction)
    d_rows = df["DEATH_WITHIN_24H"].values
    i_rows = df["ICU_WITHIN_24H"].values
    e_rows = df["EVENT_FLAG"].values
    print(f"  Row-level positives: death={int(d_rows.sum()):,}  "
          f"icu={int(i_rows.sum()):,}  event={int(e_rows.sum()):,}  of {len(df):,}")

    def row_metrics(score, target):
        return _metrics(d_rows, i_rows, e_rows, score, target)

    baselines = {}
    print("\nBaselines (row-level):")
    for t in TARGETS:
        n_ap, n_au, prev = row_metrics(df["NEWS-2"].values.astype(np.float64), t)
        s_ap, s_au, _ = row_metrics(snapshot.astype(np.float64), t)
        baselines[f"news2_auprc_{t}"] = n_ap
        baselines[f"snap_auprc_{t}"] = s_ap
        baselines[f"prevalence_{t}"] = prev
        print(f"  {t:6s}  prevalence={prev:.5f}  NEWS-2 AUPRC={n_ap:.5f} "
              f"(AUROC={n_au:.5f})  Snapshot AUPRC={s_ap:.5f} (AUROC={s_au:.5f})")

    # EWMA depends only on α — cache the α values we actually need
    alphas_needed = sorted({float(BASE_ALPHA)} | {float(a) for a in ALPHA_VALS})
    print(f"\nPrecomputing EWMA for {len(alphas_needed)} α values…")
    ewma_cache = {}
    for a in alphas_needed:
        t0 = time.time()
        if np.isclose(a, 1.0):
            ewma_cache[a] = {v: pv[v].astype(np.float64) for v in vitals}
        else:
            ewma_cache[a] = es.compute_ewma(
                t_arr, pv, gs, ge, vitals,
                {v: a for v in vitals},
                {v: es.EWMA_REF_DEFAULT for v in vitals},
            )
        print(f"  α={a:.1f}  {time.time()-t0:.0f}s", flush=True)

    tv_set = set(TEMPORAL_VITALS)
    print("\nPrecomputing OLS trend slopes…")
    t0sl = time.time()
    slopes = es.compute_slopes(t_arr, pv, gs, ge, vitals)
    print(f"  done in {time.time()-t0sl:.0f}s")

    def score_for(a, b, g):
        final = es.temporal_score(pv, ewma_cache[float(a)], slopes, vitals, b, g,
                                  method="additive", temporal_vitals=tv_set)
        return final.astype(np.float64)

    # Baseline point, used for the Δ columns
    base_scores = score_for(BASE_ALPHA, BASE_BETA, BASE_GAMMA)
    base_auprc = {t: row_metrics(base_scores, t)[0] for t in TARGETS}
    print(f"\nBaseline (α={BASE_ALPHA}, β={BASE_BETA}, γ={BASE_GAMMA}) AUPRC:  "
          + "  ".join(f"{t}={base_auprc[t]:.5f}" for t in TARGETS))

    sweeps = [
        ("alpha", ALPHA_VALS, lambda v: (v, BASE_BETA, BASE_GAMMA)),
        ("beta",  BETA_VALS,  lambda v: (BASE_ALPHA, v, BASE_GAMMA)),
        ("gamma", GAMMA_VALS, lambda v: (BASE_ALPHA, BASE_BETA, v)),
    ]

    rows = []
    print("\nOne-at-a-time sweeps (row-level)…")
    for param, values, combo in sweeps:
        print(f"\n── {param} ──")
        for v in values:
            a, b, g = combo(float(v))
            score = score_for(a, b, g)
            row = {
                "parameter": param,
                "value": float(v),
                "alpha": float(a), "beta": float(b), "gamma": float(g),
                "is_baseline": bool(np.isclose(a, BASE_ALPHA)
                                    and np.isclose(b, BASE_BETA)
                                    and np.isclose(g, BASE_GAMMA)),
            }
            for t in TARGETS:
                ap, au, prev = row_metrics(score, t)
                row[f"auprc_{t}"] = ap
                row[f"auroc_{t}"] = au
                row[f"prevalence_{t}"] = prev
                row[f"delta_auprc_{t}_vs_baseline"] = ap - base_auprc[t]
                row[f"delta_auprc_{t}_vs_news2"] = ap - baselines[f"news2_auprc_{t}"]
                row[f"delta_auprc_{t}_vs_snapshot"] = ap - baselines[f"snap_auprc_{t}"]
            rows.append(row)
            print(f"  {param}={float(v):.2f}  " +
                  "  ".join(f"{t} AUPRC={row[f'auprc_{t}']:.5f}" for t in TARGETS),
                  flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV.relative_to(REPO)}  ({len(res)} rows)")

    print("\n═══ ROW-LEVEL SENSITIVITY RANGE (event AUPRC) ═══")
    for param, _, _ in sweeps:
        sub = res[res["parameter"] == param]
        best = sub.loc[sub["auprc_event"].idxmax()]
        print(f"  {param:6s}  range {sub['auprc_event'].min():.5f} → "
              f"{sub['auprc_event'].max():.5f}  "
              f"(spread {sub['auprc_event'].max()-sub['auprc_event'].min():.5f})  "
              f"best {param}={best['value']:.2f}")
    print(f"\nTotal: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
