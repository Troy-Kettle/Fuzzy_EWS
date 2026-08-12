"""
ROW-LEVEL (per-observation) grid search scored by AUPRC (average precision).

Same grid, sampling, LUTs and EWMA + sigmoid-trend scoring path (engine_scoring.py,
matching app/streamlit_app.py) as grid_search_excess_patient.py, with two differences:

  • metrics are computed on raw observation scores — no per-patient peak
    reduction, so each row is one sample
  • the ranked metric is AUPRC, not AUROC (AUROC still recorded alongside)

Grid:  α 0.1→1.0 (×10)  ×  β 0.0→4.5 (×10)  ×  γ 0.1→1.0 (×10)

Outputs → results/current/grid_search_rowlevel_auprc/
"""

import sys, time, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "engine"))
import engine_scoring as es

from grid_search_excess_patient import (
    ALPHA_VALS, BETA_VALS, GAMMA_VALS, TEMPORAL_VITALS,
    NE_PATIENTS, RANDOM_SEED, TARGET_LABEL, load, build_pv,
)

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

OUT_DIR = REPO / "results" / "24thJuly" / "grid_search_rowlevel_auprc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["death", "icu", "event"]


def _pool(d, i, e, score, target):
    pm, nm = es.pools(d, i, e, target)
    keep = pm | nm
    y = pm[keep].astype(np.int8)
    s = np.asarray(score)[keep]
    m = np.isfinite(s)
    return y[m], s[m]


def auprc(d, i, e, score, target):
    y, s = _pool(d, i, e, score, target)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def auroc(d, i, e, score, target):
    y, s = _pool(d, i, e, score, target)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def make_heatmaps(res, target, baselines, prevalence, out_path):
    """Pairwise AUPRC heatmaps, third parameter fixed at its best value."""
    col = f"auprc_{target}"
    best = res.loc[res[col].idxmax()]
    ba, bb, bg = best["alpha"], best["beta"], best["gamma"]

    slices = [
        ("alpha", "beta",  "gamma", bg, ALPHA_VALS, BETA_VALS,  "α (EWMA memory)", "β (trend steepness)"),
        ("alpha", "gamma", "beta",  bb, ALPHA_VALS, GAMMA_VALS, "α (EWMA memory)", "γ (aggregation)"),
        ("beta",  "gamma", "alpha", ba, BETA_VALS,  GAMMA_VALS, "β (trend steepness)", "γ (aggregation)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    vmin, vmax = res[col].quantile(0.05), res[col].max()
    for ax, (xp, yp, fp, fv, xv, yv, xl, yl) in zip(axes, slices):
        sub = res[np.isclose(res[fp], fv)]
        piv = sub.pivot_table(index=yp, columns=xp, values=col)
        piv = piv.reindex(index=sorted(piv.index), columns=sorted(piv.columns))
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="RdYlGn",
                       vmin=vmin, vmax=vmax,
                       extent=[xv.min()-0.025, xv.max()+0.025,
                               yv.min()-0.025, yv.max()+0.025])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("AUPRC", fontsize=10)
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.plot(best[xp], best[yp], "w*", ms=16, markeredgecolor="k",
                markeredgewidth=0.8, zorder=10,
                label=f"Best: {xp}={best[xp]:.1f}, {yp}={best[yp]:.1f}")
        ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
        ax.set_title(f"{xl.split('(')[0].strip()} × {yl.split('(')[0].strip()}\n"
                     f"({fp}={fv:.1f} fixed)", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Row-level AUPRC Grid Search — {TARGET_LABEL[target]}\n"
        f"Best: α={ba:.1f}, β={bb:.1f}, γ={bg:.1f}  →  AUPRC={best[col]:.5f}   "
        f"│  NEWS-2={baselines['news2_'+target]:.5f}  "
        f"│  Snapshot={baselines['snap_'+target]:.5f}  "
        f"│  Prevalence={prevalence[target]:.5f}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out_path.name}")


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    t_total = time.time()

    df = load()
    vitals = es.VITALS_BASE

    print("Building fuzzy LUTs (engine, five-set SBP)…")
    luts = {v: es.build_lut(v) for v in vitals}

    event_ids = set(df.loc[df["EVENT_FLAG"] == 1, "ANON_ADMISSION_ID"].unique())
    ne_ids = list(set(df["ANON_ADMISSION_ID"].unique()) - event_ids)
    ne_sample = rng.choice(ne_ids, size=min(NE_PATIENTS, len(ne_ids)), replace=False)
    df = df[df["ANON_ADMISSION_ID"].isin(set(event_ids) | set(ne_sample))].copy()
    print(f"  Computation dataset: {len(df):,} rows")

    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    t_arr = df["t_minutes"].values.astype(np.float64)

    pv = build_pv(df, luts, vitals)
    snapshot = sum(pv[v] for v in vitals).astype(np.float64)

    # Row-level labels
    d_rows = df["DEATH_WITHIN_24H"].values
    i_rows = df["ICU_WITHIN_24H"].values
    e_rows = df["EVENT_FLAG"].values

    prevalence = {}
    baselines = {}
    news2 = df["NEWS-2"].values.astype(np.float64)
    print("\nBaselines (row-level AUPRC):")
    for t in TARGETS:
        y, _ = _pool(d_rows, i_rows, e_rows, snapshot, t)
        prevalence[t] = float(y.mean())
        baselines[f"news2_{t}"] = auprc(d_rows, i_rows, e_rows, news2, t)
        baselines[f"snap_{t}"] = auprc(d_rows, i_rows, e_rows, snapshot, t)
        print(f"  {t:6s}  prevalence={prevalence[t]:.5f}  "
              f"NEWS-2={baselines[f'news2_{t}']:.5f}  "
              f"Snapshot={baselines[f'snap_{t}']:.5f}")

    print("\nPrecomputing EWMA for each α…")
    ewma_cache = {}
    for a in ALPHA_VALS:
        t0 = time.time()
        if np.isclose(a, 1.0):
            ewma_cache[a] = {v: pv[v].astype(np.float64) for v in vitals}
        else:
            ewma_cache[a] = es.compute_ewma(
                t_arr, pv, gs, ge, vitals,
                {v: float(a) for v in vitals},
                {v: es.EWMA_REF_DEFAULT for v in vitals},
            )
        print(f"  α={a:.1f}  {time.time()-t0:.0f}s", flush=True)

    tv_set = set(TEMPORAL_VITALS)
    print("\nPrecomputing OLS trend slopes…")
    t0sl = time.time()
    slopes = es.compute_slopes(t_arr, pv, gs, ge, vitals)
    print(f"  done in {time.time()-t0sl:.0f}s")

    n_combos = len(ALPHA_VALS) * len(BETA_VALS) * len(GAMMA_VALS)
    print(f"\nGrid search: {n_combos} combos (row-level AUPRC)")
    results = []; combo = 0; t0g = time.time()

    for a in ALPHA_VALS:
        ew = ewma_cache[a]
        for b in BETA_VALS:
            for g in GAMMA_VALS:
                final = es.temporal_score(pv, ew, slopes, vitals, b, g,
                                          method="additive",
                                          temporal_vitals=tv_set).astype(np.float64)
                row = {"alpha": float(a), "beta": float(b), "gamma": float(g)}
                for t in TARGETS:
                    row[f"auprc_{t}"] = auprc(d_rows, i_rows, e_rows, final, t)
                    row[f"auroc_{t}"] = auroc(d_rows, i_rows, e_rows, final, t)
                results.append(row); combo += 1
                if combo % 100 == 0:
                    print(f"  [{combo:>4d}/{n_combos}]  "
                          f"α={a} β={b} γ={g}  event AUPRC={row['auprc_event']:.5f} "
                          f"({time.time()-t0g:.0f}s)", flush=True)

    res = pd.DataFrame(results)
    for t in TARGETS:
        res[f"prevalence_{t}"] = prevalence[t]
        res[f"delta_auprc_{t}_vs_news2"] = res[f"auprc_{t}"] - baselines[f"news2_{t}"]
        res[f"delta_auprc_{t}_vs_snapshot"] = res[f"auprc_{t}"] - baselines[f"snap_{t}"]
    res.to_csv(OUT_DIR / "grid_results_rowlevel_auprc.csv", index=False)
    print(f"\nSaved grid_results_rowlevel_auprc.csv  ({len(res)} rows)  "
          f"Total: {time.time()-t_total:.0f}s")

    print("\n═══ BEST CONFIGURATIONS (row-level AUPRC) ═══")
    for t in TARGETS:
        b = res.loc[res[f"auprc_{t}"].idxmax()]
        print(f"  {t:6s}  α={b['alpha']:.1f} β={b['beta']:.1f} γ={b['gamma']:.1f} "
              f"AUPRC={b[f'auprc_{t}']:.5f}"
              f"  vs Snapshot {b[f'auprc_{t}']-baselines[f'snap_{t}']:+.5f}"
              f"  vs NEWS-2 {b[f'auprc_{t}']-baselines[f'news2_{t}']:+.5f}")
    print(f"\n  Grid AUPRC range (event): "
          f"{res['auprc_event'].min():.5f} → {res['auprc_event'].max():.5f}")

    print("\nGenerating heatmaps…")
    for t in TARGETS:
        make_heatmaps(res, t, baselines, prevalence, OUT_DIR / f"heatmap_auprc_{t}.png")
    print(f"\nDone. → {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
