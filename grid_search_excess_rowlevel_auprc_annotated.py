"""
ROW-LEVEL (per-observation) grid search scored by AUPRC (average precision) —
ANNOTATED dataset variant.

Same grid, LUTs, and EWMA + sigmoid-trend scoring path (engine_scoring.py,
matching app/streamlit_app.py) as grid_search_excess_rowlevel_auprc.py, but run
against the smaller clinically-annotated dataset
(datasets/Annotated dataset_training_anonymised_V5_Troy_with_death.xlsx)
instead of the full derived-label dataset. Differences from the main dataset
required adapting the loader:

  • labels: the annotated dataset carries a single row-level flag,
    4_HOURS_FROM_ANNOTATED_EVENT, rather than separate DEATH_WITHIN_24H /
    ICU_WITHIN_24H / EVENT_FLAG columns, and no ICU signal at all — so only
    the "event" target is scored here (death/icu are not derivable).
  • inspired oxygen: recorded as raw INSPIRED_O2 + INSPIRED_O2_UNITS ('%' or
    'litres') rather than the clinical delivery category (INSP_O2_CAT) the
    main dataset provides. The two units are kept SEPARATE — no interconversion
    — and each row is scored through the membership function for its own unit
    ('%' via the inspired-oxygen concentration MF, 'litres' via the
    supplementary-oxygen L/min MF). See engine_scoring.split_inspired_oxygen /
    inspired_oxygen_concern. Earlier revisions converted flow to a pseudo-FiO2
    (20 + 4×L/min) and scored it on the concentration MF; that fabricated
    concentrations the source never recorded and is no longer used.
  • the whole dataset (829 admissions) is used; no non-event patient
    downsampling (NE_PATIENTS) is needed since the annotated set is already
    small and curated, unlike the multi-million-row main dataset.

ACVPU: unlike the main dataset's pipeline this one has clean per-row ACVPU
text, and it is a strong signal here (event rate 5.9% Alert vs 49.6%
Unresponsive). Pass --acvpu to score it as a 7th vital; snapshot AUPRC rises
0.2975 → 0.3103 (AUROC 0.8117 → 0.8184), a bigger gain than the whole temporal
layer contributes. The default 6-vital run stays comparable to
grid_search_excess_rowlevel_auprc.py on the main dataset.

Outputs → results/annotated/grid_search_rowlevel_auprc[_acvpu]/
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
    ALPHA_VALS, BETA_VALS, GAMMA_VALS, TEMPORAL_VITALS, TARGET_LABEL,
)

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

DATA_PATH = REPO / "datasets" / "Annotated dataset_training_anonymised_V5_Troy_with_death.xlsx"

# --acvpu used to add ACVPU as a 7th scored vital. ACVPU is no longer scored at all
# (any non-Alert reading is a deterioration flag and contributes nothing to the total),
# so the flag is accepted but ignored, and the run is always 6-vital.
if "--acvpu" in sys.argv:
    print("note: --acvpu is obsolete and ignored — ACVPU is a flag, not a scored vital.")
USE_ACVPU = False
OUT_DIR = REPO / "results" / "annotated" / "grid_search_rowlevel_auprc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["event"]  # death/icu not derivable from this dataset — see module docstring

# This dataset spells newly-confused as plain "Confused"; the engine's canonical
# map uses the full "Newly confused / agitated" label. Same ordinal value (2).
ACVPU_MAP_ANNOTATED = dict(es.ACVPU_MAP, **{"Confused": 2.0})


# ── Load ─────────────────────────────────────────────────────────────────────
def load():
    print("Loading annotated dataset…"); t0 = time.time()
    df = pd.read_excel(DATA_PATH)
    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()

    df = df.rename(columns={
        "ADMISSION_ID": "ANON_ADMISSION_ID",
        "OBS_DAYS_SINCE_ADMISSION": "DAYS_SINCE_ADMISSION",
        "O2_SATS": "SATS_SPO2",
        "NEWS_2_SCORE": "NEWS-2",
        "4_HOURS_FROM_ANNOTATED_EVENT": "EVENT_FLAG",
    })

    for c in ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
              "TEMPERATURE", "DAYS_SINCE_ADMISSION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2", "TEMPERATURE"],
              inplace=True)

    # Inspired O2: raw value + unit, each kept in its own units and scored on its
    # own membership function — no interconversion (see module docstring).
    df["FIO2_PCT"], df["SUPP_O2_LMIN"] = es.split_inspired_oxygen(
        df["INSPIRED_O2"], df["INSPIRED_O2_UNITS"])

    df["ACVPU_NUM"] = df["ACVPU"].map(ACVPU_MAP_ANNOTATED).fillna(0.0)

    df["NEWS-2"] = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0)
    df["EVENT_FLAG"] = pd.to_numeric(df["EVENT_FLAG"], errors="coerce").fillna(0).astype(np.int8)

    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"] * 1440.0
                       + obs.dt.hour.fillna(0) * 60.0
                       + obs.dt.minute.fillna(0)
                       + obs.dt.second.fillna(0) / 60.0).astype(np.float32)
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df):,} rows, {df['ANON_ADMISSION_ID'].nunique():,} admissions "
          f"in {time.time()-t0:.0f}s (event rows: {df['EVENT_FLAG'].sum():,} = "
          f"{100*df['EVENT_FLAG'].mean():.1f}%)")
    return df


def _pool(e, score, target):
    """Every row is a sample here — unlike the main dataset there is no
    death/icu pool split (es.pools), since only the event label exists."""
    y = (e == 1).astype(np.int8)
    s = np.asarray(score)
    m = np.isfinite(s)
    return y[m], s[m]


def auprc(e, score, target):
    y, s = _pool(e, score, target)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def auroc(e, score, target):
    y, s = _pool(e, score, target)
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
        f"Row-level AUPRC Grid Search (annotated dataset, "
        f"{'7 vitals incl. ACVPU' if USE_ACVPU else '6 vitals'}) — {TARGET_LABEL[target]}\n"
        f"Best: α={ba:.1f}, β={bb:.1f}, γ={bg:.1f}  →  AUPRC={best[col]:.5f}   "
        f"│  NEWS-2={baselines['news2_'+target]:.5f}  "
        f"│  Snapshot={baselines['snap_'+target]:.5f}  "
        f"│  Prevalence={prevalence[target]:.5f}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out_path.name}")


def main():
    t_total = time.time()

    df = load()
    vitals = es.VITALS_BASE + [es.ACVPU] if USE_ACVPU else es.VITALS_BASE
    print(f"Vitals: {len(vitals)} ({'incl. ACVPU' if USE_ACVPU else 'no ACVPU'})")

    print("Building fuzzy LUTs (engine, five-set SBP, + supplementary-O2 L/min)…")
    luts = {v: es.build_lut(v) for v in vitals + [es.SUPP_O2_LMIN]}

    print(f"  Computation dataset: {len(df):,} rows")

    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    t_arr = df["t_minutes"].values.astype(np.float64)

    # inspired oxygen is scored per row in its own recorded unit, so it bypasses
    # the single-column apply_luts path
    pv = es.apply_luts(df, luts, [v for v in vitals if v != "inspired_oxygen"])
    pv["inspired_oxygen"] = es.inspired_oxygen_concern(
        df["FIO2_PCT"].values, df["SUPP_O2_LMIN"].values,
        luts["inspired_oxygen"], luts[es.SUPP_O2_LMIN])
    snapshot = sum(pv[v] for v in vitals).astype(np.float64)

    e_rows = df["EVENT_FLAG"].values

    prevalence = {}
    baselines = {}
    news2 = df["NEWS-2"].values.astype(np.float64)
    print("\nBaselines (row-level AUPRC):")
    for t in TARGETS:
        y, _ = _pool(e_rows, snapshot, t)
        prevalence[t] = float(y.mean())
        baselines[f"news2_{t}"] = auprc(e_rows, news2, t)
        baselines[f"snap_{t}"] = auprc(e_rows, snapshot, t)
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
    print(f"\nGrid search: {n_combos} combos (row-level AUPRC, annotated dataset)")
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
                    row[f"auprc_{t}"] = auprc(e_rows, final, t)
                    row[f"auroc_{t}"] = auroc(e_rows, final, t)
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

    print("\n═══ BEST CONFIGURATIONS (row-level AUPRC, annotated dataset) ═══")
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
