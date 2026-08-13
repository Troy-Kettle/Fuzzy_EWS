"""Grid search over α / β / γ for the NEW system, main dataset, all three targets.

"New system" = every fix that has landed since results/main was last generated,
all applied together:

  • five-set SBP membership: No concern absorbs above-mild and above-moderate, so the
    only above-normal set left is severe (engine_scoring._merge_sbp_no_concern)
  • fixed inspired-oxygen signal (INSP_O2_CAT clinical delivery category, not the
    unreliable INSPIRED_O2_TEXT LUT)
  • chronic-respiratory-aware NEWS-2 baseline (Scale 2 SpO2 sub-score wherever
    Chronic_Resp=1, joined in from the annotated dataset on
    (admission, day, obs time); every other row defaults to Scale 1 — this dataset
    does not carry the field itself, see the coverage line printed at run time)
  • ACVPU is NOT scored: any non-Alert reading flags the row as deterioration and
    contributes nothing to the fuzzy total (NEWS-2 keeps its own +3, as the real tool does)
  • new β/trend rules: worsening-trend factor gated by a dead zone
    (TREND_MIN_SLOPE_DEFAULT) AND a two-consecutive-reading persistence check
    (sustained_slope_mask), via gs/ge passed into engine_scoring.temporal_score

Scored at BOTH patient level (peak per admission, every admission, no sampling)
and row level (every one of the ~9.3M observations, no sampling — this is the
"use all the data" run; ~3h of metric evaluation alone per the row-level cap this
replaces).

Grid: α 0.1→1.0 (×10) × β 0.0→4.5 (×10) × γ 0.1→1.0 (×10) = 1,000 combos.

Outputs → results/main_dataset/{patient level, row-level}/grid_search/
"""

import sys, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_scoring as es
import common as C

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

# Shared definitions come from pipeline/common.py so the grid, the cohort and the oxygen
# category ramp cannot drift between the grid search, the validation and the sensitivity
# sweep. This script keeps its own loader only because it additionally joins in
# chronic-respiratory status (see load_chronic_lookup / attach_chronic_resp).
DATA_PATH    = C.DATA_PATH
CHRONIC_PATH = REPO / "datasets" / "Annotated dataset_training_anonymised_V5_Troy (1).xlsx"

PATIENT_DIR = C.PATIENT_DIR / "grid_search"
ROW_DIR     = C.ROW_DIR / "grid_search"
for d in (PATIENT_DIR, ROW_DIR):
    d.mkdir(parents=True, exist_ok=True)
OUT_DIR = C.MAIN_DIR
LEVELS = {"patient": PATIENT_DIR, "row": ROW_DIR}

ALPHA_VALS, BETA_VALS, GAMMA_VALS = C.ALPHA_VALS, C.BETA_VALS, C.GAMMA_VALS
VITALS = es.VITALS_BASE   # 6 scored vitals; ACVPU is not one (flag only, no score)
TEMPORAL_VITALS = C.TEMPORAL_VITALS
O2CAT_CONCERN = C.O2CAT_CONCERN

TARGETS = ["death", "icu", "event"]
TARGET_LABEL = {"death": "Death within 24h", "icu": "ICU within 24h",
                "event": "Event within 24h"}
TARGET_COLOR = {"death": "#E74C3C", "icu": "#3498DB", "event": "#27AE60"}
METRICS = ["auroc", "auprc"]
METRIC_LABEL = {"auroc": "AUROC", "auprc": "AUPRC"}


# ── NEWS-2 sub-scores (chronic-aware Scale 1/2 via engine_scoring) ────────────
def score_resp(x):
    return np.select([x <= 8, x <= 11, x <= 20, x <= 24], [3, 1, 0, 2], default=3)


def score_temp(x):
    return np.select([x <= 35.0, x <= 36.0, x <= 38.0, x <= 39.0], [3, 1, 0, 1], default=2)


def score_bp(x):
    return np.select([x <= 90, x <= 100, x <= 110, x <= 219], [3, 2, 1, 0], default=3)


def score_hr(x):
    return np.select([x <= 40, x <= 50, x <= 90, x <= 110, x <= 130], [3, 1, 0, 1, 2], default=3)


# ── Load ─────────────────────────────────────────────────────────────────────
def load_chronic_lookup():
    df = pd.read_excel(CHRONIC_PATH, sheet_name="Annotated training",
                       usecols=["ADMISSION_ID", "OBS_DAYS_SINCE_ADMISSION",
                                "OBS_TIME", "CHRONIC_RESP_OBS_SET"])
    df = df.drop_duplicates(subset=["ADMISSION_ID", "OBS_DAYS_SINCE_ADMISSION", "OBS_TIME"])
    return df.set_index(["ADMISSION_ID", "OBS_DAYS_SINCE_ADMISSION", "OBS_TIME"])["CHRONIC_RESP_OBS_SET"]


def load():
    print("Loading dataset…", flush=True); t0 = time.time()
    cols = ["ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
            "HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
            "INSPIRED_O2_TEXT", "INSP_O2_CAT", "AVPU_ACVPU", "TEMPERATURE",
            "COMPLETE_DATA", "DIED_FLAG", "ICU_FLAG",
            "DEATH_WITHIN_24H", "ICU_WITHIN_24H", "EVENT_FLAG"]
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    for c in ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
              "TEMPERATURE", "DAYS_SINCE_ADMISSION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2", "TEMPERATURE"],
              inplace=True)

    df["INSPIRED_O2_TEXT"] = (pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce")
                              .fillna(21.0).clip(21, 100))
    df["O2_CONCERN"] = df["INSP_O2_CAT"].map(O2CAT_CONCERN).fillna(0.0).astype(np.float32)
    df["ACVPU_NUM"] = df["AVPU_ACVPU"].map(es.ACVPU_MAP).fillna(0.0).astype(np.float32)
    for c in ["DIED_FLAG", "ICU_FLAG", "DEATH_WITHIN_24H", "ICU_WITHIN_24H", "EVENT_FLAG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)

    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"] * 1440.0
                       + obs.dt.hour.fillna(0) * 60.0
                       + obs.dt.minute.fillna(0)
                       + obs.dt.second.fillna(0) / 60.0).astype(np.float64)
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df):,} complete rows in {time.time()-t0:.0f}s, "
          f"{df['ANON_ADMISSION_ID'].nunique():,} patients", flush=True)
    return df


def attach_chronic_resp(df):
    lookup = load_chronic_lookup()
    key = list(zip(df["ANON_ADMISSION_ID"].values, df["DAYS_SINCE_ADMISSION"].astype("int64").values,
                    df["OBS_TIME"].values))
    matched = pd.Series(key, index=df.index).map(lookup)
    n_matched = matched.notna().sum()
    df["CHRONIC_RESP"] = matched.fillna(0).astype(np.int8)
    print(f"  Chronic-resp coverage: {n_matched:,}/{len(df):,} rows matched to the annotated "
          f"dataset; all other rows default to 0 = Scale 1.", flush=True)
    return df


def build_pv(df, luts, vitals):
    pv = es.apply_luts(df, luts, vitals)
    pv["inspired_oxygen"] = df["O2_CONCERN"].values.astype(np.float32)
    return pv


def score_both(d, i, e, score, target):
    pm, nm = es.pools(d, i, e, target)
    keep = pm | nm
    y = pm[keep].astype(np.int8)
    s = np.asarray(score)[keep]
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


# ── Figures (same layout as results/main) ──────────────────────────────────────
def make_heatmaps(res, level, target, metric, baselines, out_path):
    col = f"{metric}_{target}"
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
        cb.set_label(METRIC_LABEL[metric], fontsize=10)
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.plot(best[xp], best[yp], "w*", ms=16, markeredgecolor="k",
                markeredgewidth=0.8, zorder=10,
                label=f"Best: {xp}={best[xp]:.1f}, {yp}={best[yp]:.1f}")
        ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
        ax.set_title(f"{xl.split('(')[0].strip()} × {yl.split('(')[0].strip()}\n"
                     f"({fp}={fv:.1f} fixed)", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"{level.capitalize()}-level {METRIC_LABEL[metric]} grid search — {TARGET_LABEL[target]} (new system)\n"
        f"Best: α={ba:.1f}, β={bb:.1f}, γ={bg:.1f}  →  {METRIC_LABEL[metric]}={best[col]:.5f}   "
        f"│  NEWS-2={baselines[f'news2_{metric}_{target}']:.5f}  "
        f"│  Snapshot={baselines[f'snap_{metric}_{target}']:.5f}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {out_path.name}", flush=True)


def make_top_table(res, level, metric, baselines, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, tgt in zip(axes, TARGETS):
        col = f"{metric}_{tgt}"
        top = res.nlargest(10, col)[["alpha", "beta", "gamma", col]].reset_index(drop=True)
        top.columns = ["α", "β", "γ", METRIC_LABEL[metric]]
        top[METRIC_LABEL[metric]] = top[METRIC_LABEL[metric]].round(5)
        ax.axis("off")
        tbl = ax.table(cellText=top.values, colLabels=top.columns, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.1, 1.55)
        for j in range(4):
            tbl[0, j].set_facecolor("#1A252F"); tbl[0, j].set_text_props(color="white", weight="bold")
        for i in range(1, len(top) + 1):
            fc = "#EBF5FB" if i % 2 == 0 else "white"
            for j in range(4):
                tbl[i, j].set_facecolor(fc)
        for j in range(4):
            tbl[1, j].set_facecolor("#D5F5E3"); tbl[1, j].set_text_props(weight="bold")
        ax.set_title(f"Top 10 — {TARGET_LABEL[tgt]}\n"
                     f"NEWS-2: {baselines[f'news2_{metric}_{tgt}']:.4f}  │  "
                     f"Snapshot: {baselines[f'snap_{metric}_{tgt}']:.4f}", fontsize=11, pad=14)
    fig.suptitle(f"Top configurations (new system) — {level} level, {METRIC_LABEL[metric]}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {out_path.name}", flush=True)


def main():
    t_total = time.time()
    df = load()
    df = attach_chronic_resp(df)

    print("Building fuzzy LUTs (five-set SBP)…", flush=True)
    luts = {v: es.build_lut(v) for v in VITALS}

    pv = build_pv(df, luts, VITALS)
    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    t_arr = df["t_minutes"].values.astype(np.float64)
    acvpu_raw = df["ACVPU_NUM"].values

    d_row = df["DEATH_WITHIN_24H"].values
    i_row = df["ICU_WITHIN_24H"].values
    e_row = df["EVENT_FLAG"].values
    row_idx = np.arange(len(df))   # no sampling: every observation

    labels = {"row": (d_row, i_row, e_row),
              "patient": (np.maximum.reduceat(d_row, gs),
                          np.maximum.reduceat(i_row, gs),
                          np.maximum.reduceat(e_row, gs))}
    e_pat = labels["patient"][2]
    print(f"  Patients (all, no sampling): {len(gs):,}  "
          f"(event pos={int(e_pat.sum()):,} = {100*e_pat.mean():.2f}% prevalence)")
    print(f"  Rows (all, no sampling): {len(row_idx):,}  "
          f"(event pos={int(e_row.sum()):,} = {100*e_row.mean():.2f}% prevalence)")

    # Chronic-aware NEWS-2 baseline (Scale 1 unless Chronic_Resp=1)
    on_oxygen = df["INSPIRED_O2_TEXT"].values > 21
    news2 = (score_resp(df["RESP_RATE"].values)
             + es.news2_spo2_score(df["SATS_SPO2"].values, df["CHRONIC_RESP"].values, on_oxygen)
             + score_temp(df["TEMPERATURE"].values)
             + score_bp(df["SYSTOLIC_BP"].values)
             + score_hr(df["HEART_RATE"].values)
             + np.where(on_oxygen, 2, 0)
             + es.news2_consciousness_score(acvpu_raw)).astype(np.float64)

    snapshot = es.snapshot_score(pv, VITALS, method="additive", gamma=1.0,
                                 )

    baselines = {lv: {} for lv in LEVELS}
    print("\nBaselines:")
    for lv in LEVELS:
        d, i, e = labels[lv]
        n2 = news2[row_idx] if lv == "row" else np.maximum.reduceat(news2, gs)
        sn = snapshot[row_idx] if lv == "row" else np.maximum.reduceat(snapshot, gs)
        for t in TARGETS:
            for tag, sc in (("news2", n2), ("snap", sn)):
                ro, pr = score_both(d, i, e, sc, t)
                baselines[lv][f"{tag}_auroc_{t}"] = ro
                baselines[lv][f"{tag}_auprc_{t}"] = pr
        print(f"  {lv:8s} " + "  ".join(
            f"{t}: NEWS-2 {baselines[lv][f'news2_auroc_{t}']:.4f}/"
            f"{baselines[lv][f'news2_auprc_{t}']:.4f}  "
            f"Snap {baselines[lv][f'snap_auroc_{t}']:.4f}/"
            f"{baselines[lv][f'snap_auprc_{t}']:.4f}" for t in TARGETS))

    print("\nPrecomputing EWMA for each α…", flush=True)
    ewma_cache = {}
    for a in ALPHA_VALS:
        t0 = time.time()
        if np.isclose(a, 1.0):
            ewma_cache[a] = {v: pv[v].astype(np.float64) for v in VITALS}
        else:
            ewma_cache[a] = es.compute_ewma(t_arr, pv, gs, ge, VITALS,
                                            {v: float(a) for v in VITALS},
                                            {v: es.EWMA_REF_DEFAULT for v in VITALS})
        print(f"  α={a:.1f}  {time.time()-t0:.0f}s", flush=True)

    print("\nPrecomputing OLS trend slopes (α/β/γ-independent)…", flush=True)
    t0 = time.time()
    slopes = es.compute_slopes(t_arr, pv, gs, ge, VITALS)
    print(f"  done in {time.time()-t0:.0f}s")

    print("Precomputing sustained-slope persistence masks (new β/trend rules, "
          "α/β/γ-independent — computed once, not per grid combo)…", flush=True)
    t0 = time.time()
    sustained_masks = {v: es.sustained_slope_mask(slopes[v], gs, ge, es.TREND_MIN_SLOPE_DEFAULT)
                       for v in TEMPORAL_VITALS}
    print(f"  done in {time.time()-t0:.0f}s")

    n_combos = len(ALPHA_VALS) * len(BETA_VALS) * len(GAMMA_VALS)
    print(f"\nGrid search: {n_combos} combos × 2 levels × 3 targets × 2 metrics "
          f"(all {len(row_idx):,} rows, no cap)", flush=True)
    rows = {lv: [] for lv in LEVELS}
    combo = 0; t0g = time.time()
    for a in ALPHA_VALS:
        ew = ewma_cache[a]
        for b in BETA_VALS:
            for g in GAMMA_VALS:
                final = es.temporal_score(
                    pv, ew, slopes, VITALS, b, g, method="additive",
                    temporal_vitals=TEMPORAL_VITALS,
                    gs=gs, ge=ge, sustained_masks=sustained_masks,
                ).astype(np.float64)
                peak = np.maximum.reduceat(final, gs)
                for lv, sc in (("row", final[row_idx]), ("patient", peak)):
                    d, i, e = labels[lv]
                    r = {"alpha": float(a), "beta": float(b), "gamma": float(g)}
                    for t in TARGETS:
                        r[f"auroc_{t}"], r[f"auprc_{t}"] = score_both(d, i, e, sc, t)
                    rows[lv].append(r)
                combo += 1
                if combo % 20 == 0:
                    el = time.time() - t0g
                    print(f"  [{combo:>4d}/{n_combos}]  α={a} β={b} γ={g}  "
                          f"{el:.0f}s elapsed, ~{el/combo*(n_combos-combo)/60:.0f} min left",
                          flush=True)

    best_rows = []
    for lv, out_dir in LEVELS.items():
        res = pd.DataFrame(rows[lv])
        for t in TARGETS:
            for m in METRICS:
                res[f"delta_{m}_{t}_vs_news2"] = res[f"{m}_{t}"] - baselines[lv][f"news2_{m}_{t}"]
                res[f"delta_{m}_{t}_vs_snapshot"] = res[f"{m}_{t}"] - baselines[lv][f"snap_{m}_{t}"]
        res.to_csv(out_dir / "grid_results.csv", index=False)
        print(f"\n═══ BEST CONFIGURATIONS — {lv} level (new system) ═══")
        for m in METRICS:
            for t in TARGETS:
                col = f"{m}_{t}"
                bst = res.loc[res[col].idxmax()]
                print(f"  {m.upper()} {t:6s}  α={bst['alpha']:.1f} β={bst['beta']:.1f} "
                      f"γ={bst['gamma']:.1f}  {bst[col]:.5f}"
                      f"  vs Snapshot {bst[col]-baselines[lv][f'snap_{m}_{t}']:+.5f}"
                      f"  vs NEWS-2 {bst[col]-baselines[lv][f'news2_{m}_{t}']:+.5f}")
                best_rows.append({"level": lv, "metric": m.upper(), "target": t,
                                  "alpha": bst["alpha"], "beta": bst["beta"],
                                  "gamma": bst["gamma"], "value": round(bst[col], 5),
                                  "snapshot_baseline": round(baselines[lv][f"snap_{m}_{t}"], 5),
                                  "news2_baseline": round(baselines[lv][f"news2_{m}_{t}"], 5)})

        print(f"  generating figures for {lv} level…")
        for m in METRICS:
            for t in TARGETS:
                make_heatmaps(res, lv, t, m, baselines[lv], out_dir / f"heatmap_{m}_{t}.png")
            make_top_table(res, lv, m, baselines[lv], out_dir / f"top_configs_{m}.png")
        pd.DataFrame(baselines[lv], index=[0]).to_csv(out_dir / "baselines.csv", index=False)

    best = pd.DataFrame(best_rows)
    for out_dir in LEVELS.values():
        best.to_csv(out_dir / "best_configs.csv", index=False)
    print(f"\nDone in {time.time()-t_total:.0f}s. → {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
