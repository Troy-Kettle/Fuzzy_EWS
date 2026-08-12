"""
PATIENT-LEVEL grid search — improved system.

Same structure as the original but delegates all scoring to engine_scoring.py,
which implements the full set of correctness fixes:

  A1  defuzz exact-zero rule: a perfectly normal vital scores 0 (was ~0.25)
  A2  canonical ACVPU map: Voice=1, Confused=2 (was swapped)
  C1  five-set SBP: above-mild and above-moderate are absorbed into No concern, leaving
      above-severe as the only above-normal set
  O2  inspired-oxygen scored from INSP_O2_CAT (recorded clinical category) instead
      of the broken INSPIRED_O2_TEXT field that mixes L/min flow with FiO2%

Temporal formula is the same EWMA + sigmoid worsening-trend adjustment as
app/streamlit_app.py (the interactive main system) — see engine_scoring.temporal_score.
EWMA reference spacing = 60min, trend look-back window = 24h (both engine defaults,
matching the app). β controls the sigmoid steepness on the OLS trend slope, not an
excess-amplification factor.

Grid:  α 0.1→1.0 (×10)  ×  β 0.0→4.5 (×10)  ×  γ 0.1→1.0 (×10)  =  1,000 combos
Aggregation: additive + worst-vital mix (γ), EWMA + sigmoid-trend.
Outputs → patient_level_results/results/grid_search_excess/
"""

import sys, time, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "engine"))
import engine_scoring as es

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

DATA_PATH = REPO / "datasets" / "final_observations_with_targets.csv"
OUT_DIR   = REPO / "results" / "24thJuly" / "grid_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_VALS   = np.round(np.arange(0.1, 1.05, 0.1), 2)
BETA_VALS    = np.round(np.arange(0.0, 4.55, 0.5), 1)
GAMMA_VALS   = np.round(np.arange(0.1, 1.05, 0.1), 2)
NE_PATIENTS  = 22_336
RANDOM_SEED  = 42

# Temporal layer applies ONLY to continuously-varying physiological vitals.
# inspired_oxygen is a categorical clinical intervention (INSP_O2_CAT): its step
# changes create artefactual EWMA/trend excursions that swamp the real deterioration
# signal, so it contributes to the snapshot total but is excluded from the
# EWMA + trend adjustment.
TEMPORAL_VITALS = es.TEMPORAL_VITALS_DEFAULT   # canonical set (incl. inspired_oxygen);
# defined once in engine_scoring so the app and the pipeline cannot drift apart again.

# O2 fix: recorded delivery category → concern (untuned clinical ramp)
O2CAT_CONCERN = {"Low": 1.0, "Low-moderate": 1.5, "Moderate": 2.0,
                 "High": 2.5, "Very high": 3.0}

TARGET_LABEL = {"death": "Death within 24h", "icu": "ICU within 24h",
                "event": "Event within 24h"}
TARGET_COLOR = {"death": "#E74C3C", "icu": "#3498DB", "event": "#27AE60"}


# ── Load ─────────────────────────────────────────────────────────────────────
def load():
    print("Loading dataset…"); t0 = time.time()
    cols = ["ANON_ADMISSION_ID", "OBS_TIME", "DAYS_SINCE_ADMISSION",
            "HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
            "INSPIRED_O2_TEXT", "INSP_O2_CAT", "TEMPERATURE",
            "COMPLETE_DATA", "NEWS-2", "DEATH_WITHIN_24H", "ICU_WITHIN_24H", "EVENT_FLAG"]
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    df["COMPLETE_DATA"] = pd.to_numeric(df["COMPLETE_DATA"], errors="coerce").fillna(0)
    df = df[df["COMPLETE_DATA"] == 1].copy()
    for c in ["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2",
              "TEMPERATURE", "DAYS_SINCE_ADMISSION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["HEART_RATE", "SYSTOLIC_BP", "RESP_RATE", "SATS_SPO2", "TEMPERATURE"],
              inplace=True)
    # keep INSPIRED_O2_TEXT for LUT fallback; real O2 signal comes from INSP_O2_CAT
    df["INSPIRED_O2_TEXT"] = (pd.to_numeric(df["INSPIRED_O2_TEXT"], errors="coerce")
                              .fillna(21.0).clip(21, 100))
    df["NEWS-2"]   = pd.to_numeric(df["NEWS-2"], errors="coerce").fillna(0)
    df["O2_CONCERN"] = df["INSP_O2_CAT"].map(O2CAT_CONCERN).fillna(0.0).astype(np.float32)
    obs = pd.to_datetime(df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    df["t_minutes"] = (df["DAYS_SINCE_ADMISSION"] * 1440.0
                       + obs.dt.hour.fillna(0) * 60.0
                       + obs.dt.minute.fillna(0)
                       + obs.dt.second.fillna(0) / 60.0).astype(np.float32)
    df["ANON_ADMISSION_ID"] = df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID", "t_minutes"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    on_o2 = df["O2_CONCERN"].values > 0
    print(f"  {len(df):,} rows in {time.time()-t0:.0f}s "
          f"(O2 via INSP_O2_CAT: {on_o2.sum():,} rows = {100*on_o2.mean():.1f}%)")
    return df


def build_pv(df, luts, vitals):
    """Score vitals via LUT, then override inspired_oxygen from clinical category."""
    pv = es.apply_luts(df, luts, vitals)
    pv["inspired_oxygen"] = df["O2_CONCERN"].values.astype(np.float32)
    return pv


# ── AUROC helper ─────────────────────────────────────────────────────────────
def ca(y_d, y_i, y_e, score, target):
    return es.auroc(y_d, y_i, y_e, score, target)


# ── Plotting (unchanged from original) ───────────────────────────────────────
def make_heatmaps(res, target, baselines, out_path):
    best = res.loc[res[target].idxmax()]
    ba, bb, bg = best["alpha"], best["beta"], best["gamma"]
    slices = [
        ("alpha", "beta",  "gamma", bg, ALPHA_VALS, BETA_VALS,  "α (EWMA memory)", "β (trend steepness)"),
        ("alpha", "gamma", "beta",  bb, ALPHA_VALS, GAMMA_VALS, "α (EWMA memory)", "γ (aggregation)"),
        ("beta",  "gamma", "alpha", ba, BETA_VALS,  GAMMA_VALS, "β (trend steepness)", "γ (aggregation)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    vmin, vmax = res[target].quantile(0.05), res[target].max()
    for ax, (xp, yp, fp, fv, xv, yv, xl, yl) in zip(axes, slices):
        sub = res[np.isclose(res[fp], fv)]
        piv = sub.pivot_table(index=yp, columns=xp, values=target)
        piv = piv.reindex(index=sorted(piv.index), columns=sorted(piv.columns))
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="RdYlGn",
                       vmin=vmin, vmax=vmax,
                       extent=[xv.min()-0.025, xv.max()+0.025,
                               yv.min()-0.025, yv.max()+0.025])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("AUROC", fontsize=10)
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.plot(best[xp], best[yp], "w*", ms=16, markeredgecolor="k",
                markeredgewidth=0.8, zorder=10,
                label=f"Best: {xp}={best[xp]:.1f}, {yp}={best[yp]:.1f}")
        ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
        ax.set_title(f"{xl.split('(')[0].strip()} × {yl.split('(')[0].strip()}\n"
                     f"({fp}={fv:.1f} fixed)", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"EWMA + Trend Grid Search — {TARGET_LABEL[target]}\n"
        f"Best: α={ba:.1f}, β={bb:.1f}, γ={bg:.1f}  →  AUROC={best[target]:.5f}   "
        f"│  NEWS-2={baselines['news2_'+target]:.5f}  │  Snapshot={baselines['snap_'+target]:.5f}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out_path.name}")


def make_sensitivity_lines(res, baselines):
    best = res.loc[res["event"].idxmax()]
    ba, bb, bg = best["alpha"], best["beta"], best["gamma"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    param_info = [
        ("alpha", ALPHA_VALS, ba, "α (EWMA memory)",   f"β={bb:.1f}, γ={bg:.1f} fixed"),
        ("beta",  BETA_VALS,  bb, "β (trend steepness)",  f"α={ba:.1f}, γ={bg:.1f} fixed"),
        ("gamma", GAMMA_VALS, bg, "γ (aggregation)",   f"α={ba:.1f}, β={bb:.1f} fixed"),
    ]
    for ax, (param, vals, bv, xlabel, subtitle) in zip(axes, param_info):
        for tgt, col in TARGET_COLOR.items():
            if param == "alpha":  sub = res[np.isclose(res["beta"], bb)  & np.isclose(res["gamma"], bg)]
            elif param == "beta": sub = res[np.isclose(res["alpha"], ba) & np.isclose(res["gamma"], bg)]
            else:                 sub = res[np.isclose(res["alpha"], ba) & np.isclose(res["beta"], bb)]
            sub = sub.sort_values(param)
            ax.plot(sub[param], sub[tgt], "o-", color=col, lw=2, ms=5,
                    label=TARGET_LABEL[tgt])
            ax.axhline(baselines[f"news2_{tgt}"], color=col, ls=":",  lw=1.3, alpha=0.7)
            ax.axhline(baselines[f"snap_{tgt}"],  color=col, ls="--", lw=1.0, alpha=0.5)
        ax.axvline(bv, color="black", ls="--", lw=1.2, alpha=0.6, label=f"Best {param}={bv:.1f}")
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title(f"Sensitivity to {param}\n({subtitle})", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(True, alpha=0.3)
        if param == "alpha":
            from matplotlib.lines import Line2D
            h, l = ax.get_legend_handles_labels()
            h += [Line2D([0],[0], color="grey", ls=":",  lw=1.3, label="NEWS-2"),
                  Line2D([0],[0], color="grey", ls="--", lw=1.0, label="Snapshot")]
            ax.legend(handles=h, fontsize=8)
    fig.suptitle("Sensitivity at Optimal Values — EWMA + Trend Temporal System\n"
                 "Dotted = NEWS-2  │  Dashed = Snapshot", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "sensitivity_lines.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  Saved sensitivity_lines.png")


def make_top_table(res, baselines):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, tgt in zip(axes, ["death", "icu", "event"]):
        top = res.nlargest(10, tgt)[["alpha", "beta", "gamma", tgt]].reset_index(drop=True)
        top.columns = ["α", "β", "γ", "AUROC"]; top["AUROC"] = top["AUROC"].round(5)
        ax.axis("off")
        tbl = ax.table(cellText=top.values, colLabels=top.columns, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.1, 1.55)
        for j in range(4):
            tbl[0, j].set_facecolor("#1A252F"); tbl[0, j].set_text_props(color="white", weight="bold")
        for i in range(1, 11):
            fc = "#EBF5FB" if i % 2 == 0 else "white"
            for j in range(4): tbl[i, j].set_facecolor(fc)
        for j in range(4):
            tbl[1, j].set_facecolor("#D5F5E3"); tbl[1, j].set_text_props(weight="bold")
        n2 = baselines[f"news2_{tgt}"]; sn = baselines[f"snap_{tgt}"]
        ax.set_title(f"Top 10 — {TARGET_LABEL[tgt]}\nNEWS-2: {n2:.4f}  │  Snapshot: {sn:.4f}",
                     fontsize=11, pad=14)
    fig.suptitle("Top Configurations — EWMA + Trend System (improved)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_configs_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  Saved top_configs_table.png")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(RANDOM_SEED); t_total = time.time()

    df = load()
    vitals = es.VITALS_BASE   # 6 vitals (ACVPU added only in auroc_target_comparison)

    # Build LUTs using engine (A1 defuzz fix + five-set SBP)
    print("Building fuzzy LUTs (engine, five-set SBP)…")
    luts = {v: es.build_lut(v) for v in vitals}

    # Sample: all event patients + NE_PATIENTS random non-event patients
    event_ids = set(df.loc[df["EVENT_FLAG"] == 1, "ANON_ADMISSION_ID"].unique())
    ne_ids    = list(set(df["ANON_ADMISSION_ID"].unique()) - event_ids)
    ne_sample = rng.choice(ne_ids, size=min(NE_PATIENTS, len(ne_ids)), replace=False)
    keep = set(event_ids) | set(ne_sample)
    df = df[df["ANON_ADMISSION_ID"].isin(keep)].copy()
    print(f"  Computation dataset: {len(df):,} rows")

    gs, ge = es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    t_arr  = df["t_minutes"].values.astype(np.float64)

    # Per-vital scores (with O2 override from clinical category)
    pv = build_pv(df, luts, vitals)
    snapshot = sum(pv[v] for v in vitals).astype(np.float32)

    # Patient-level labels
    d_rows = df["DEATH_WITHIN_24H"].values
    i_rows = df["ICU_WITHIN_24H"].values
    e_rows = df["EVENT_FLAG"].values
    y_d = np.maximum.reduceat(d_rows, gs)
    y_i = np.maximum.reduceat(i_rows, gs)
    y_e = np.maximum.reduceat(e_rows, gs)
    n_patients = len(gs)
    print(f"  Patient samples: {n_patients:,}  "
          f"(event pos={int(y_e.sum()):,}, neg={int((y_e==0).sum()):,})")

    news2_pat = np.maximum.reduceat(df["NEWS-2"].values.astype(np.float64), gs)
    snap_pat  = np.maximum.reduceat(snapshot.astype(np.float64), gs)
    baselines = {f"news2_{t}": ca(y_d, y_i, y_e, news2_pat, t) for t in ["death", "icu", "event"]}
    baselines.update({f"snap_{t}": ca(y_d, y_i, y_e, snap_pat, t) for t in ["death", "icu", "event"]})
    print("\nBaselines (patient-level, peak score):")
    for t in ["death", "icu", "event"]:
        print(f"  {t:6s}  NEWS-2={baselines[f'news2_{t}']:.4f}  "
              f"Snapshot={baselines[f'snap_{t}']:.4f}")

    # Precompute EWMA for each α (shared across β, γ)
    print("\nPrecomputing EWMA for each α…")
    ewma_cache = {}
    for a in ALPHA_VALS:
        t0s = time.time()
        alphas = {v: float(a) for v in vitals}
        refs   = {v: es.EWMA_REF_DEFAULT for v in vitals}
        if np.isclose(a, 1.0):
            ewma_cache[a] = {v: pv[v].astype(np.float64) for v in vitals}
        else:
            ewma_cache[a] = es.compute_ewma(t_arr, pv, gs, ge, vitals, alphas, refs)
        print(f"  α={a:.1f}  {time.time()-t0s:.0f}s")

    # Trend slopes (OLS over the 24h look-back window) are parameter-independent —
    # precompute once and reuse across every α/β/γ combo (mirrors app/grid_search_auroc.py).
    print("\nPrecomputing OLS trend slopes…")
    t0sl = time.time()
    slopes = es.compute_slopes(t_arr, pv, gs, ge, vitals)
    print(f"  done in {time.time()-t0sl:.0f}s")

    # Grid search
    n_combos = len(ALPHA_VALS) * len(BETA_VALS) * len(GAMMA_VALS)
    print(f"\nGrid search: {n_combos} combos (EWMA + sigmoid trend, additive)")
    results = []; combo = 0; t0g = time.time()

    # EWMA + trend adjustment applies to TEMPORAL_VITALS only; ACVPU stays at its
    # snapshot value (genuinely categorical → spurious trend otherwise) and carries a
    # flat bonus instead. All scoring goes through engine_scoring.temporal_score
    # (single source of truth, matching app/streamlit_app.py); the per-α EWMA cache
    # stays hoisted here because it is the expensive part.
    tv_set = set(TEMPORAL_VITALS)
    for a in ALPHA_VALS:
        ew = ewma_cache[a]
        for b in BETA_VALS:
            for g in GAMMA_VALS:
                final = es.temporal_score(pv, ew, slopes, vitals, b, g,
                                          method="additive", temporal_vitals=tv_set)
                patient_score = np.maximum.reduceat(final, gs)

                row = {"alpha": a, "beta": b, "gamma": g}
                for tgt in ["death", "icu", "event"]:
                    row[tgt] = ca(y_d, y_i, y_e, patient_score, tgt)
                results.append(row); combo += 1
                if combo % 200 == 0:
                    print(f"  [{combo:>4d}/{n_combos}]  α={a} β={b} γ={g}  "
                          f"event={row['event']:.4f}  ({time.time()-t0g:.0f}s)")

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "grid_results.csv", index=False)
    print(f"\nSaved grid_results.csv  ({len(res)} rows)  Total: {time.time()-t_total:.0f}s")

    print("\n═══ BEST CONFIGURATIONS (improved system) ═══════════════════════════")
    for tgt in ["death", "icu", "event"]:
        b = res.loc[res[tgt].idxmax()]
        print(f"  {tgt:6s}  α={b['alpha']:.1f} β={b['beta']:.1f} γ={b['gamma']:.1f} "
              f"AUROC={b[tgt]:.5f}"
              f"  vs Snapshot {b[tgt]-baselines[f'snap_{tgt}']:+.5f}"
              f"  vs NEWS-2 {b[tgt]-baselines[f'news2_{tgt}']:+.5f}")
    print(f"\n  Grid AUROC range (event): "
          f"{res['event'].min():.5f} → {res['event'].max():.5f}")

    print("\nGenerating figures…")
    for tgt in ["death", "icu", "event"]:
        make_heatmaps(res, tgt, baselines, OUT_DIR / f"heatmap_{tgt}.png")
    make_sensitivity_lines(res, baselines)
    make_top_table(res, baselines)
    print("\nDone.")


if __name__ == "__main__":
    main()
