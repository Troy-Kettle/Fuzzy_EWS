"""
PATIENT-LEVEL grid search with the EXCESS-ABOVE-EWMA temporal signal.

Identical to the baseline (row-level) grid search except AUROC is computed per
ADMISSION rather than per observation row: each admission contributes ONE sample,
scored by the MAXIMUM (peak) score the system assigned across that admission's stay,
labelled by whether the admission ever had the event (clean per-target pools).

New temporal formula:
    ewma[i]  = α_eff * raw[i] + (1-α_eff) * ewma[i-1]   (time-adjusted)
    excess   = max(0, raw - ewma)      <- only positive when DETERIORATING above baseline
    adj      = clip(raw + β * excess, 0, 3)

This means:
  - Patient stable or improving:  excess=0  →  adj=raw  →  total=snapshot
  - Patient deteriorating above their own EWMA baseline:
      excess>0  →  adj>raw  →  total>snapshot  (exactly the intended clinical signal)

The max(total, snapshot) floor is now redundant (adj >= raw always) but kept for safety.
The old OLS slope / sigmoid trend factor is removed — excess IS the trend signal.

Grid:
    α: 0.1 → 1.0, step 0.1   (10 values, controls EWMA baseline length)
    β: 0.0 → 4.5, step 0.5   (10 values, controls excess amplification)
    γ: 0.1 → 1.0, step 0.1   (10 values, controls aggregation)
    Total: 1,000 combinations

Outputs → results/grid_search_excess/
"""

import time, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
REPO        = Path(__file__).resolve().parent
DATA_PATH   = REPO / "datasets" / "final_observations_with_targets.csv"
SIGMOID_DIR = REPO / "membership_functions" / "sigmoid"
OUT_DIR     = REPO / "patient_level_results" / "results" / "grid_search_excess"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Grids ──────────────────────────────────────────────────────────────────
ALPHA_VALS = np.round(np.arange(0.1, 1.05, 0.1), 2)    # 0.1 … 1.0
BETA_VALS  = np.round(np.arange(0.0, 4.55, 0.5), 1)    # 0.0 … 4.5
GAMMA_VALS = np.round(np.arange(0.1, 1.05, 0.1), 2)    # 0.1 … 1.0

# Time-decay reference: a_eff = 1 − (1−α)^(Δt/EWMA_REF_MIN) is proper exponential
# time-decay for irregular sampling (prior weight ∝ exp(−λ·Δt)). Calibrated to the
# empirical median within-ward observation gap (~5.5h = 331 min → rounded to 360 min)
# so α controls memory retained per typical observation interval. The previous 60-min
# value discounted the prior baseline to ~0.1% by the next obs, collapsing temporal→snapshot.
EWMA_REF_MIN  = 360.0
NE_PATIENTS   = 22_336
MAX_NEG_AUROC = 500_000
RANDOM_SEED   = 42

# ── Vitals (AVPU excluded) ─────────────────────────────────────────────────
VITALS    = ["heart_rate","blood_pressure","temperature",
             "respiratory_rate","oxygen_saturation","inspired_oxygen"]
VITAL_COL = {"heart_rate":"HEART_RATE","blood_pressure":"SYSTOLIC_BP",
             "temperature":"TEMPERATURE","respiratory_rate":"RESP_RATE",
             "oxygen_saturation":"SATS_SPO2","inspired_oxygen":"INSPIRED_O2_TEXT"}
MF_FILE   = {"heart_rate":"heart_rate_membership_functions.csv",
             "blood_pressure":"systolic_blood_pressure_membership_functions.csv",
             "temperature":"temperature_membership_functions.csv",
             "respiratory_rate":"respiratory_rate_membership_functions.csv",
             "oxygen_saturation":"oxygen_saturation_membership_functions.csv",
             "inspired_oxygen":"inspired_oxygen_concentration_membership_functions.csv"}
VITAL_TYPE = {"heart_rate":"7var","blood_pressure":"7var","temperature":"7var",
              "respiratory_rate":"7var","oxygen_saturation":"3var_down","inspired_oxygen":"3var_up"}
LABELS_7      = ["Below normal - severe concern","Below normal - moderate concern",
                 "Below normal - mild concern","No concern","Above normal - mild concern",
                 "Above normal - moderate concern","Above normal - severe concern"]
LABELS_3_DOWN = ["Below normal - severe concern","Below normal - moderate concern",
                 "Below normal - mild concern","No concern"]
LABELS_3_UP   = ["No concern","Above normal - mild concern",
                 "Above normal - moderate concern","Above normal - severe concern"]
OUTPUT_MF = {"No concern":(-0.5,0,0,0.75),"Mild concern":(0.25,1,1,1.75),
             "Moderate concern":(1.25,2,2,2.75),"Severe concern":(2.25,3,3,3.5)}
_OUTPUT_X = np.arange(0,3.01,0.01)
_OUTPUT_GRID = {lbl: np.array([(1. if b<=x<=c else (0. if x<=a or x>=d else
    (x-a)/(b-a) if a<x<b else (d-x)/(d-c))) for x in _OUTPUT_X])
    for lbl,(a,b,c,d) in OUTPUT_MF.items()}


# ── Fuzzy LUT ──────────────────────────────────────────────────────────────
def _defuzz(mem):
    con={"No concern":0.,"Mild concern":0.,"Moderate concern":0.,"Severe concern":0.}
    for k,v in mem.items():
        kl=k.lower()
        if "severe" in kl:     con["Severe concern"]   = max(con["Severe concern"],v)
        elif "moderate" in kl: con["Moderate concern"] = max(con["Moderate concern"],v)
        elif "mild" in kl:     con["Mild concern"]     = max(con["Mild concern"],v)
        else:                  con["No concern"]        = max(con["No concern"],v)
    agg=np.zeros(301)
    for lev,f in con.items():
        if f>=0.05: np.maximum(agg,np.minimum(f,_OUTPUT_GRID[lev]),out=agg)
    d=agg.sum()
    return 0. if d==0 else float(np.dot(_OUTPUT_X,agg)/d)

def _build_lut(vital):
    df=pd.read_csv(SIGMOID_DIR/MF_FILE[vital]); x=df["Value"].values.astype(float)
    labs={"7var":LABELS_7,"3var_down":LABELS_3_DOWN,"3var_up":LABELS_3_UP}[VITAL_TYPE[vital]]
    y=np.array([_defuzz({l:float(np.interp(v,x,df[l].values)) for l in labs}) for v in x])
    return x,y

def apply_luts(df,luts):
    return {v: np.interp(np.clip(df[VITAL_COL[v]].values.astype(np.float64),
                luts[v][0][0],luts[v][0][-1]),luts[v][0],luts[v][1]).astype(np.float32)
            for v in VITALS}


# ── EWMA ───────────────────────────────────────────────────────────────────
def group_boundaries(ids):
    ch=np.empty(len(ids),bool); ch[0]=True; ch[1:]=ids[1:]!=ids[:-1]
    st=np.where(ch)[0]; return st, np.append(st[1:],len(ids))

def ewma_compute(times, raw, alpha, gs, ge):
    out=np.empty_like(raw,dtype=np.float64)
    for g in range(len(gs)):
        s,e=gs[g],ge[g]; out[s]=raw[s]
        for i in range(s+1,e):
            dt=max(float(times[i]-times[i-1]),0.)
            a=1.-(1.-alpha)**(dt/EWMA_REF_MIN)
            out[i]=a*raw[i]+(1.-a)*out[i-1]
    return out


# ── AUROC helpers ───────────────────────────────────────────────────────────
def ca(y_d, y_i, y_e, score, target):
    if target=="death": pm,nm = y_d==1,(y_d==0)&(y_i==0)
    elif target=="icu":  pm,nm = y_i==1,(y_i==0)&(y_d==0)
    else:               pm,nm = y_e==1, y_e==0
    mask=pm|nm; y=pm[mask].astype(int); s=score[mask]; ok=np.isfinite(s)
    if y[ok].sum()==0 or y[ok].sum()==ok.sum(): return float("nan")
    return float(roc_auc_score(y[ok],s[ok]))


# ── Plotting ────────────────────────────────────────────────────────────────
TARGET_LABEL = {"death":"Death within 24h","icu":"ICU within 24h","event":"Event within 24h"}
TARGET_COLOR = {"death":"#E74C3C","icu":"#3498DB","event":"#27AE60"}

def make_heatmaps(res, target, baselines, out_path):
    best     = res.loc[res[target].idxmax()]
    ba,bb,bg = best["alpha"],best["beta"],best["gamma"]

    slices = [
        ("alpha","beta", "gamma",bg, ALPHA_VALS,BETA_VALS,  "α (EWMA memory)","β (excess amplification)"),
        ("alpha","gamma","beta", bb, ALPHA_VALS,GAMMA_VALS, "α (EWMA memory)","γ (aggregation)"),
        ("beta", "gamma","alpha",ba, BETA_VALS, GAMMA_VALS, "β (excess amplification)","γ (aggregation)"),
    ]
    fig, axes = plt.subplots(1,3,figsize=(19,6))
    vmin,vmax = res[target].quantile(0.05), res[target].max()

    for ax,(xp,yp,fp,fv,xv,yv,xl,yl) in zip(axes,slices):
        sub = res[np.isclose(res[fp],fv)]
        piv = sub.pivot_table(index=yp,columns=xp,values=target)
        piv = piv.reindex(index=sorted(piv.index),columns=sorted(piv.columns))
        im  = ax.imshow(piv.values,aspect="auto",origin="lower",cmap="RdYlGn",
                        vmin=vmin,vmax=vmax,
                        extent=[xv.min()-0.025,xv.max()+0.025,
                                yv.min()-0.025,yv.max()+0.025])
        cb  = plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        cb.set_label("AUROC",fontsize=10)
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        # annotate best cell
        ax.plot(best[xp],best[yp],"w*",ms=16,markeredgecolor="k",
                markeredgewidth=0.8,zorder=10,
                label=f"Best: {xp}={best[xp]:.1f}, {yp}={best[yp]:.1f}")
        ax.set_xlabel(xl,fontsize=11)
        ax.set_ylabel(yl,fontsize=11)
        ax.set_title(f"{xl.split('(')[0].strip()} × {yl.split('(')[0].strip()}\n"
                     f"({fp}={fv:.1f} fixed)",fontsize=11)
        ax.legend(fontsize=8,loc="upper left")

    fig.suptitle(
        f"Excess-EWMA Grid Search — {TARGET_LABEL[target]}\n"
        f"Best: α={ba:.1f}, β={bb:.1f}, γ={bg:.1f}  →  AUROC={best[target]:.5f}   "
        f"│  NEWS-2={baselines['news2_'+target]:.5f}  │  Snapshot={baselines['snap_'+target]:.5f}",
        fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.91])
    fig.savefig(out_path,dpi=200,bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def make_sensitivity_lines(res, baselines):
    best = res.loc[res["event"].idxmax()]
    ba,bb,bg = best["alpha"],best["beta"],best["gamma"]

    fig,axes = plt.subplots(1,3,figsize=(17,5.5))
    param_info = [
        ("alpha",ALPHA_VALS,ba,"α (EWMA memory)",   f"β={bb:.1f}, γ={bg:.1f} fixed"),
        ("beta", BETA_VALS, bb,"β (excess boost)",  f"α={ba:.1f}, γ={bg:.1f} fixed"),
        ("gamma",GAMMA_VALS,bg,"γ (aggregation)",   f"α={ba:.1f}, β={bb:.1f} fixed"),
    ]
    for ax,(param,vals,bv,xlabel,subtitle) in zip(axes,param_info):
        for tgt,col in TARGET_COLOR.items():
            if param=="alpha":   sub=res[np.isclose(res["beta"],bb)  & np.isclose(res["gamma"],bg)]
            elif param=="beta":  sub=res[np.isclose(res["alpha"],ba) & np.isclose(res["gamma"],bg)]
            else:                sub=res[np.isclose(res["alpha"],ba) & np.isclose(res["beta"],bb)]
            sub=sub.sort_values(param)
            ax.plot(sub[param],sub[tgt],"o-",color=col,lw=2,ms=5,label=TARGET_LABEL[tgt])
            ax.axhline(baselines[f"news2_{tgt}"],color=col,ls=":",lw=1.3,alpha=0.7)
            ax.axhline(baselines[f"snap_{tgt}"], color=col,ls="--",lw=1.0,alpha=0.5)
        ax.axvline(bv,color="black",ls="--",lw=1.2,alpha=0.6,label=f"Best {param}={bv:.1f}")
        ax.set_xlabel(xlabel,fontsize=11)
        ax.set_ylabel("AUROC",fontsize=11)
        ax.set_title(f"Sensitivity to {param}\n({subtitle})",fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(True,alpha=0.3)
        if param=="alpha":
            from matplotlib.lines import Line2D
            h,l=ax.get_legend_handles_labels()
            h+=[Line2D([0],[0],color="grey",ls=":",lw=1.3,label="NEWS-2"),
                Line2D([0],[0],color="grey",ls="--",lw=1.0,label="Snapshot")]
            ax.legend(handles=h,fontsize=8)
    fig.suptitle("Sensitivity at Optimal Values — Excess-EWMA Temporal System\n"
                 "Dotted = NEWS-2  │  Dashed = Snapshot",fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.93])
    fig.savefig(OUT_DIR/"sensitivity_lines.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    print("  Saved sensitivity_lines.png")


def make_top_table(res, baselines):
    fig,axes = plt.subplots(1,3,figsize=(18,5.5))
    for ax,tgt in zip(axes,["death","icu","event"]):
        top = res.nlargest(10,tgt)[["alpha","beta","gamma",tgt]].reset_index(drop=True)
        top.columns=["α","β","γ","AUROC"]; top["AUROC"]=top["AUROC"].round(5)
        ax.axis("off")
        tbl=ax.table(cellText=top.values,colLabels=top.columns,loc="center",cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.1,1.55)
        for j in range(4):
            tbl[0,j].set_facecolor("#1A252F"); tbl[0,j].set_text_props(color="white",weight="bold")
        for i in range(1,11):
            fc="#EBF5FB" if i%2==0 else "white"
            for j in range(4): tbl[i,j].set_facecolor(fc)
        for j in range(4):
            tbl[1,j].set_facecolor("#D5F5E3"); tbl[1,j].set_text_props(weight="bold")
        n2=baselines[f"news2_{tgt}"]; sn=baselines[f"snap_{tgt}"]
        ax.set_title(f"Top 10 — {TARGET_LABEL[tgt]}\nNEWS-2: {n2:.4f}  │  Snapshot: {sn:.4f}",
                     fontsize=11,pad=14)
    fig.suptitle("Top Configurations — Excess-EWMA System",fontsize=13,y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"top_configs_table.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    print("  Saved top_configs_table.png")


def make_comparison_fig(res, old_res_path, baselines):
    """Compare best excess-EWMA vs best old temporal vs snapshot vs NEWS-2."""
    if not Path(old_res_path).exists():
        return
    old = pd.read_csv(old_res_path)
    fig,ax = plt.subplots(figsize=(10,5.5))
    targets = ["death","icu","event"]
    xlabels = [TARGET_LABEL[t] for t in targets]
    x = np.arange(len(targets)); w = 0.18

    systems = {
        "NEWS-2":           [baselines[f"news2_{t}"] for t in targets],
        "Snapshot":         [baselines[f"snap_{t}"]  for t in targets],
        "Old Temporal\n(best from prior grid)":
                            [old.loc[old[t].idxmax(), t] for t in targets],
        "Excess-EWMA\n(best from new grid)":
                            [res.loc[res[t].idxmax(), t] for t in targets],
    }
    colors = ["#7F8C8D","#3498DB","#E67E22","#27AE60"]
    for i,(name,vals) in enumerate(systems.items()):
        bars=ax.bar(x+(i-1.5)*w,vals,w,label=name,color=colors[i],
                    edgecolor="white",lw=0.5,alpha=0.9)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,
                    f"{v:.4f}",ha="center",va="bottom",fontsize=7.5,rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(xlabels,fontsize=12)
    ax.set_ylabel("AUROC",fontsize=12)
    ax.set_title("System Comparison: Excess-EWMA vs Old Temporal vs Baselines",fontsize=12)
    ylo = min(min(v) for v in systems.values()) - 0.02
    yhi = max(max(v) for v in systems.values()) + 0.04
    ax.set_ylim(max(0.5,ylo), min(1.0,yhi))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=9,loc="lower right"); ax.grid(True,axis="y",alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"comparison_vs_old.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    print("  Saved comparison_vs_old.png")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng=np.random.default_rng(RANDOM_SEED); t_total=time.time()

    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading dataset…")
    usecols=["ANON_ADMISSION_ID","OBS_TIME","DAYS_SINCE_ADMISSION",
             "HEART_RATE","SYSTOLIC_BP","RESP_RATE","SATS_SPO2",
             "INSPIRED_O2_TEXT","TEMPERATURE","COMPLETE_DATA","NEWS-2",
             "DEATH_WITHIN_24H","ICU_WITHIN_24H","EVENT_FLAG"]
    df_full=pd.read_csv(DATA_PATH,usecols=usecols,low_memory=False)
    df_full["COMPLETE_DATA"]=pd.to_numeric(df_full["COMPLETE_DATA"],errors="coerce").fillna(0)
    df_full=df_full[df_full["COMPLETE_DATA"]==1].copy()
    for c in ["HEART_RATE","SYSTOLIC_BP","RESP_RATE","SATS_SPO2","TEMPERATURE","DAYS_SINCE_ADMISSION"]:
        df_full[c]=pd.to_numeric(df_full[c],errors="coerce")
    df_full.dropna(subset=["HEART_RATE","SYSTOLIC_BP","RESP_RATE","SATS_SPO2","TEMPERATURE"],inplace=True)
    df_full["INSPIRED_O2_TEXT"]=pd.to_numeric(df_full["INSPIRED_O2_TEXT"],errors="coerce").fillna(21.).clip(21,100)
    df_full["NEWS-2"]=pd.to_numeric(df_full["NEWS-2"],errors="coerce").fillna(0)
    df_full["ANON_ADMISSION_ID"]=df_full["ANON_ADMISSION_ID"].astype("int32")
    print(f"  {len(df_full):,} rows after filter")

    # ── Patient sample (identical to full_grid_search.py for fair comparison) ─
    event_ids=set(df_full.loc[df_full["EVENT_FLAG"]==1,"ANON_ADMISSION_ID"].unique())
    ne_ids=list(set(df_full["ANON_ADMISSION_ID"].unique())-event_ids)
    ne_sample=rng.choice(ne_ids,size=min(NE_PATIENTS,len(ne_ids)),replace=False)
    keep=set(event_ids)|set(ne_sample)
    df=df_full[df_full["ANON_ADMISSION_ID"].isin(keep)].copy(); del df_full
    print(f"  Computation dataset: {len(df):,} rows")

    obs=pd.to_datetime(df["OBS_TIME"],format="%H:%M:%S",errors="coerce")
    df["t_minutes"]=(df["DAYS_SINCE_ADMISSION"]*1440+obs.dt.hour.fillna(0)*60
                    +obs.dt.minute.fillna(0)+obs.dt.second.fillna(0)/60).astype(np.float32)
    df.sort_values(["ANON_ADMISSION_ID","t_minutes"],inplace=True)
    df.reset_index(drop=True,inplace=True)
    gs,ge=group_boundaries(df["ANON_ADMISSION_ID"].values)
    t_arr=df["t_minutes"].values.astype(np.float64)

    # ── LUTs + per-vital scores ───────────────────────────────────────────────
    print("Building fuzzy LUTs…")
    luts={v:_build_lut(v) for v in VITALS}
    pv=apply_luts(df,luts)
    snapshot=sum(pv[v] for v in VITALS).astype(np.float32)

    # ── Patient-level labels & scores (one sample per admission) ─────────────
    # gs are the group-start row indices (df is sorted by patient); np.maximum.reduceat
    # over gs reduces each admission's rows to its peak value.
    d_rows = df["DEATH_WITHIN_24H"].values
    i_rows = df["ICU_WITHIN_24H"].values
    e_rows = df["EVENT_FLAG"].values
    y_d = np.maximum.reduceat(d_rows, gs)          # 1 if admission EVER death-within-24h
    y_i = np.maximum.reduceat(i_rows, gs)          # 1 if admission EVER icu-within-24h
    y_e = np.maximum.reduceat(e_rows, gs)          # 1 if admission EVER event
    n_patients = len(gs)
    print(f"  Patient samples: {n_patients:,}  (event pos={int(y_e.sum()):,}, neg={int((y_e==0).sum()):,})")

    # Patient-level baselines: peak NEWS-2 / peak snapshot per admission
    news2_pat = np.maximum.reduceat(df["NEWS-2"].values.astype(np.float64), gs)
    snap_pat  = np.maximum.reduceat(snapshot.astype(np.float64), gs)
    baselines={f"news2_{t}":ca(y_d,y_i,y_e,news2_pat,t) for t in ["death","icu","event"]}
    baselines.update({f"snap_{t}":ca(y_d,y_i,y_e,snap_pat,t) for t in ["death","icu","event"]})
    print("\nBaselines (patient-level, peak score):")
    for t in ["death","icu","event"]:
        print(f"  {t:6s}  NEWS-2={baselines[f'news2_{t}']:.4f}  Snapshot={baselines[f'snap_{t}']:.4f}")

    # ── Precompute EWMA for each α ────────────────────────────────────────────
    print("\nPrecomputing EWMA for each α…")
    ewma_cache={}
    for a in ALPHA_VALS:
        t0s=time.time()
        if np.isclose(a,1.0):
            ewma_cache[a]={v:pv[v].astype(np.float64) for v in VITALS}
        else:
            ewma_cache[a]={v:ewma_compute(t_arr,pv[v].astype(np.float64),a,gs,ge) for v in VITALS}
        print(f"  α={a:.1f}  {time.time()-t0s:.0f}s")

    # ── Grid search ───────────────────────────────────────────────────────────
    n_combos=len(ALPHA_VALS)*len(BETA_VALS)*len(GAMMA_VALS)
    print(f"\nGrid search: {n_combos} combos  (excess-EWMA signal)")
    results=[]; combo=0; t0g=time.time()

    for a in ALPHA_VALS:
        ew=ewma_cache[a]
        # excess per vital: positive only when raw > ewma  (patient deteriorating above baseline)
        excess={v:np.maximum(0., pv[v].astype(np.float64) - ew[v]) for v in VITALS}

        for b in BETA_VALS:
            # adjusted: raw + β * excess, clipped to [0,3]
            adj={v:np.clip(pv[v].astype(np.float64) + b*excess[v], 0., 3.).astype(np.float32)
                 for v in VITALS}

            additive=sum(adj[v] for v in VITALS)
            if any(not np.isclose(b,0.) for _ in [None]):  # always compute for γ<1
                stacked=np.column_stack([adj[v] for v in VITALS])
                max_v=stacked.max(axis=1)
            else:
                max_v=None

            for g in GAMMA_VALS:
                if np.isclose(g,1.):
                    total=additive
                else:
                    if max_v is None:
                        stacked=np.column_stack([adj[v] for v in VITALS])
                        max_v=stacked.max(axis=1)
                    total=(1.-g)*(len(VITALS)*3./3.)*max_v + g*additive

                # max() with snapshot is now redundant (adj>=raw always) but kept as safety
                final=np.maximum(total,snapshot).astype(np.float32)
                # aggregate to one peak score per admission, then patient-level AUROC
                patient_score=np.maximum.reduceat(final, gs)

                row={"alpha":a,"beta":b,"gamma":g}
                for tgt in ["death","icu","event"]:
                    row[tgt]=ca(y_d,y_i,y_e,patient_score,tgt)
                results.append(row); combo+=1
                if combo%100==0:
                    print(f"  [{combo:>4d}/{n_combos}]  α={a} β={b} γ={g}  "
                          f"event={row['event']:.4f}  ({time.time()-t0g:.0f}s)")

    res=pd.DataFrame(results)
    res.to_csv(OUT_DIR/"grid_results.csv",index=False)
    print(f"\nSaved grid_results.csv  ({len(res)} rows)  Total: {time.time()-t_total:.0f}s")

    # ── Best configs ──────────────────────────────────────────────────────────
    print("\n═══ BEST CONFIGURATIONS (Excess-EWMA) ═══════════════════════════")
    for tgt in ["death","icu","event"]:
        b=res.loc[res[tgt].idxmax()]
        gain_vs_snap = b[tgt]-baselines[f"snap_{tgt}"]
        gain_vs_n2   = b[tgt]-baselines[f"news2_{tgt}"]
        print(f"  {tgt:6s}  α={b['alpha']:.1f} β={b['beta']:.1f} γ={b['gamma']:.1f}"
              f"  AUROC={b[tgt]:.5f}"
              f"  vs Snapshot {gain_vs_snap:+.5f}"
              f"  vs NEWS-2 {gain_vs_n2:+.5f}")

    print(f"\n  Grid AUROC range (event): "
          f"{res['event'].min():.5f} → {res['event'].max():.5f}  "
          f"(spread={res['event'].max()-res['event'].min():.5f})")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures…")
    for tgt in ["death","icu","event"]:
        make_heatmaps(res,tgt,baselines,OUT_DIR/f"heatmap_{tgt}.png")
    make_sensitivity_lines(res,baselines)
    make_top_table(res,baselines)
    make_comparison_fig(res, REPO/"results"/"full_grid_search"/"grid_results.csv", baselines)
    print("\nDone.")

if __name__=="__main__":
    main()
