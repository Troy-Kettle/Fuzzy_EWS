"""Grid search over the temporal parameters α / β / γ, scored at BOTH levels.

Previously the patient-level grid search ranked by AUROC (grid_search_excess_patient.py)
and a separate row-level one ranked by AUPRC (grid_search_excess_rowlevel_auprc.py) —
two scripts, two cohorts, two metrics. This runs one grid over one cohort and records
AUROC *and* AUPRC at patient level *and* row level for all three targets, so every
comparison in results/main comes off the same 1,000 evaluations.

  grid:   α 0.1→1.0 (×10)  ×  β 0.0→4.5 (×10)  ×  γ 0.1→1.0 (×10)
  vitals: the 6-vital set (ACVPU is a validation variant, not a tuning knob)
  cohort: patient level uses EVERY admission — no sampling. The older grid searches
          enriched the cohort to all event admissions + 22,336 non-event ones, which
          is harmless for AUROC (rank-based, prevalence-invariant) but not for AUPRC,
          whose absolute value tracks prevalence: enrichment lifted patient-level
          event prevalence from ~3.5% to ~38% and inflated every AUPRC accordingly.
          Since AUPRC is now a ranked metric, the full cohort is used instead.
          Row level still ranks on the capped sample (all event observations +
          500,000 non-event, seed 42) purely for runtime — 1,000 combos over all
          9.3M rows is ~3 h of metric evaluation alone. That cap enriches row-level
          prevalence too (~1.2% → ~18%), so treat the row-level grid AUPRC column as
          a *ranking* signal only. Absolute row-level AUPRC in the validation leaves
          is computed on all 9.3M observations with no sampling at all.

Outputs → results/main/patient level/grid_search/  and  results/main/row-level/grid_search/
The patient-level event winners are what main_validation.py then builds its leaves on.
"""

import time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import main_pipeline_common as C
import engine_scoring as es

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

TARGETS = ["death", "icu", "event"]
TARGET_LABEL = {"death": "Death within 24h", "icu": "ICU within 24h",
                "event": "Event within 24h"}
TARGET_COLOR = {"death": "#E74C3C", "icu": "#3498DB", "event": "#27AE60"}
METRICS = ["auroc", "auprc"]
METRIC_LABEL = {"auroc": "AUROC", "auprc": "AUPRC"}

LEVELS = {"patient": C.PATIENT_DIR / "grid_search",
          "row":     C.ROW_DIR / "grid_search"}


def _pool(d, i, e, score, target):
    pm, nm = es.pools(d, i, e, target)
    keep = pm | nm
    y = pm[keep].astype(np.int8)
    s = np.asarray(score)[keep]
    m = np.isfinite(s)
    return y[m], s[m]


def score_both(d, i, e, score, target):
    y, s = _pool(d, i, e, score, target)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


# ── Figures ───────────────────────────────────────────────────────────────────

def make_heatmaps(res, level, target, metric, baselines, out_path):
    """Pairwise heatmaps of one metric, third parameter pinned at its best value."""
    col = f"{metric}_{target}"
    best = res.loc[res[col].idxmax()]
    ba, bb, bg = best["alpha"], best["beta"], best["gamma"]
    slices = [
        ("alpha", "beta",  "gamma", bg, C.ALPHA_VALS, C.BETA_VALS,  "α (EWMA memory)", "β (trend steepness)"),
        ("alpha", "gamma", "beta",  bb, C.ALPHA_VALS, C.GAMMA_VALS, "α (EWMA memory)", "γ (aggregation)"),
        ("beta",  "gamma", "alpha", ba, C.BETA_VALS,  C.GAMMA_VALS, "β (trend steepness)", "γ (aggregation)"),
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
        f"{level.capitalize()}-level {METRIC_LABEL[metric]} grid search — {TARGET_LABEL[target]}\n"
        f"Best: α={ba:.1f}, β={bb:.1f}, γ={bg:.1f}  →  {METRIC_LABEL[metric]}={best[col]:.5f}   "
        f"│  NEWS-2={baselines[f'news2_{metric}_{target}']:.5f}  "
        f"│  Snapshot={baselines[f'snap_{metric}_{target}']:.5f}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {out_path.name}")


def make_sensitivity_lines(res, level, metric, baselines, out_path):
    """One-at-a-time sweeps through the event-optimal point."""
    best = res.loc[res[f"{metric}_event"].idxmax()]
    ba, bb, bg = best["alpha"], best["beta"], best["gamma"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    info = [("alpha", ba, "α (EWMA memory)",     f"β={bb:.1f}, γ={bg:.1f} fixed"),
            ("beta",  bb, "β (trend steepness)", f"α={ba:.1f}, γ={bg:.1f} fixed"),
            ("gamma", bg, "γ (aggregation)",     f"α={ba:.1f}, β={bb:.1f} fixed")]
    for ax, (param, bv, xlabel, subtitle) in zip(axes, info):
        if param == "alpha":
            sub = res[np.isclose(res["beta"], bb) & np.isclose(res["gamma"], bg)]
        elif param == "beta":
            sub = res[np.isclose(res["alpha"], ba) & np.isclose(res["gamma"], bg)]
        else:
            sub = res[np.isclose(res["alpha"], ba) & np.isclose(res["beta"], bb)]
        sub = sub.sort_values(param)
        for tgt, col in TARGET_COLOR.items():
            ax.plot(sub[param], sub[f"{metric}_{tgt}"], "o-", color=col, lw=2, ms=5,
                    label=TARGET_LABEL[tgt])
            ax.axhline(baselines[f"news2_{metric}_{tgt}"], color=col, ls=":", lw=1.3, alpha=0.7)
            ax.axhline(baselines[f"snap_{metric}_{tgt}"],  color=col, ls="--", lw=1.0, alpha=0.5)
        ax.axvline(bv, color="black", ls="--", lw=1.2, alpha=0.6, label=f"Best {param}={bv:.1f}")
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(METRIC_LABEL[metric], fontsize=11)
        ax.set_title(f"Sensitivity to {param}\n({subtitle})", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(True, alpha=0.3)
        if param == "alpha":
            from matplotlib.lines import Line2D
            h, _ = ax.get_legend_handles_labels()
            h += [Line2D([0], [0], color="grey", ls=":",  lw=1.3, label="NEWS-2"),
                  Line2D([0], [0], color="grey", ls="--", lw=1.0, label="Snapshot")]
            ax.legend(handles=h, fontsize=8)
    fig.suptitle(f"{level.capitalize()}-level {METRIC_LABEL[metric]} sensitivity at the "
                 f"event-optimal point\nDotted = NEWS-2  │  Dashed = Snapshot", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {out_path.name}")


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
    fig.suptitle(f"Top configurations — {level} level, {METRIC_LABEL[metric]}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    rng = np.random.default_rng(C.RANDOM_SEED)
    for d in LEVELS.values():
        d.mkdir(parents=True, exist_ok=True)

    df = C.load(all_columns=False)
    vitals = es.VITALS_BASE

    print("\nBuilding fuzzy LUTs (engine, five-set SBP)…")
    luts = C.build_luts(vitals)

    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    t_arr = df["t_minutes"].values

    pv = C.build_pv(df, luts, vitals)
    snapshot = sum(pv[v] for v in vitals).astype(np.float64)
    news2 = df["NEWS-2"].values.astype(np.float64)

    d_row = df["DEATH_WITHIN_24H"].values
    i_row = df["ICU_WITHIN_24H"].values
    e_row = df["EVENT_FLAG"].values

    # Row-level ranking sample: all event observations + a capped random draw of
    # non-event ones (runtime only — see module docstring).
    pos_idx = np.flatnonzero(e_row == 1)
    neg_idx = np.flatnonzero(e_row == 0)
    neg_samp = rng.choice(neg_idx, min(C.ROW_NEG_CAP, len(neg_idx)), replace=False)
    row_idx = np.sort(np.concatenate([pos_idx, neg_samp]))

    labels = {"row": (d_row[row_idx], i_row[row_idx], e_row[row_idx]),
              "patient": (np.maximum.reduceat(d_row, gs),
                          np.maximum.reduceat(i_row, gs),
                          np.maximum.reduceat(e_row, gs))}
    e_pat = labels["patient"][2]
    print(f"  Patients (all, no sampling): {len(gs):,}  "
          f"(event pos={int(e_pat.sum()):,} = {100*e_pat.mean():.2f}% prevalence)")
    print(f"  Row-level ranking sample: {len(row_idx):,} rows "
          f"({len(pos_idx):,} event + {len(neg_samp):,} non-event; "
          f"true row prevalence {100*e_row.mean():.2f}%)")

    # Baselines
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

    print("\nPrecomputing EWMA for each α…")
    ewma_cache = {}
    for a in C.ALPHA_VALS:
        t0 = time.time()
        if np.isclose(a, 1.0):
            # α=1 keeps no memory: the EWMA is the raw score itself.
            ewma_cache[a] = {v: pv[v].astype(np.float64) for v in vitals}
        else:
            ewma_cache[a] = es.compute_ewma(t_arr, pv, gs, ge, vitals,
                                            {v: float(a) for v in vitals},
                                            {v: es.EWMA_REF_DEFAULT for v in vitals})
        print(f"  α={a:.1f}  {time.time()-t0:.0f}s", flush=True)

    print("\nPrecomputing OLS trend slopes (α/β/γ-independent, computed once)…")
    t0 = time.time()
    slopes = es.compute_slopes(t_arr, pv, gs, ge, vitals)
    print(f"  done in {time.time()-t0:.0f}s")

    n_combos = len(C.ALPHA_VALS) * len(C.BETA_VALS) * len(C.GAMMA_VALS)
    print(f"\nGrid search: {n_combos} combos × 2 levels × 3 targets × 2 metrics")
    rows = {lv: [] for lv in LEVELS}
    combo = 0; t0g = time.time()
    for a in C.ALPHA_VALS:
        ew = ewma_cache[a]
        for b in C.BETA_VALS:
            for g in C.GAMMA_VALS:
                final = C.temporal(pv, ew, slopes, vitals, b, g).astype(np.float64)
                peak = np.maximum.reduceat(final, gs)
                for lv, sc in (("row", final[row_idx]), ("patient", peak)):
                    d, i, e = labels[lv]
                    r = {"alpha": float(a), "beta": float(b), "gamma": float(g)}
                    for t in TARGETS:
                        r[f"auroc_{t}"], r[f"auprc_{t}"] = score_both(d, i, e, sc, t)
                    rows[lv].append(r)
                combo += 1
                if combo % 50 == 0:
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
        print(f"\n═══ BEST CONFIGURATIONS — {lv} level ═══")
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
                make_heatmaps(res, lv, t, m, baselines[lv],
                              out_dir / f"heatmap_{m}_{t}.png")
            make_sensitivity_lines(res, lv, m, baselines[lv],
                                   out_dir / f"sensitivity_{m}.png")
            make_top_table(res, lv, m, baselines[lv], out_dir / f"top_configs_{m}.png")
        pd.DataFrame(baselines[lv], index=[0]).to_csv(out_dir / "baselines.csv", index=False)

    best = pd.DataFrame(best_rows)
    for out_dir in LEVELS.values():
        best.to_csv(out_dir / "best_configs.csv", index=False)
    print(f"\nDone in {time.time()-t_total:.0f}s.  "
          f"main_validation.py reads the patient-level event winners from best_configs.csv")


if __name__ == "__main__":
    main()
