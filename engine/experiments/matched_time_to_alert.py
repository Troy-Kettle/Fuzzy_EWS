"""
Time-to-alert compared at MATCHED detection accuracy (sensitivity).

Earlier the threshold was set to a fixed false-alarm rate, so systems detected
different fractions of patients — unfair for comparing lead time. Here we instead,
for each system, find the threshold that detects a TARGET fraction of event patients
(before the event-window onset), then report the median lead time and the false-alarm
cost at that operating point. This answers: "at the same accuracy as NEWS-2, how much
earlier does each system warn?"

Fuzzy = best config (05_combined_best): additive, α=0.1 β=0.5 γ=1.0,
relative excess, +ACVPU (so it has a consciousness input like NEWS-2).
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import engine_scoring as es

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "datasets" / "final_observations_with_targets.csv"
OUT  = REPO / "improved_results" / "time_to_alert_matched"; OUT.mkdir(parents=True, exist_ok=True)

def load():
    cols = ["ANON_ADMISSION_ID","OBS_TIME","DAYS_SINCE_ADMISSION","HEART_RATE","SYSTOLIC_BP",
            "RESP_RATE","SATS_SPO2","INSPIRED_O2_TEXT","AVPU_ACVPU","TEMPERATURE","COMPLETE_DATA",
            "NEWS-2","DEATH_WITHIN_24H","ICU_WITHIN_24H","EVENT_FLAG"]
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    df["COMPLETE_DATA"]=pd.to_numeric(df["COMPLETE_DATA"],errors="coerce").fillna(0)
    df=df[df["COMPLETE_DATA"]==1].copy()
    for c in ["HEART_RATE","SYSTOLIC_BP","RESP_RATE","SATS_SPO2","TEMPERATURE","DAYS_SINCE_ADMISSION"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df.dropna(subset=["HEART_RATE","SYSTOLIC_BP","RESP_RATE","SATS_SPO2","TEMPERATURE"],inplace=True)
    df["INSPIRED_O2_TEXT"]=pd.to_numeric(df["INSPIRED_O2_TEXT"],errors="coerce").fillna(21.).clip(21,100)
    df["NEWS-2"]=pd.to_numeric(df["NEWS-2"],errors="coerce").fillna(0)
    df["ACVPU_NUM"]=df["AVPU_ACVPU"].map(es.ACVPU_MAP).fillna(0.0)
    obs=pd.to_datetime(df["OBS_TIME"],format="%H:%M:%S",errors="coerce")
    df["t_minutes"]=(df["DAYS_SINCE_ADMISSION"]*1440.+obs.dt.hour.fillna(0)*60.
                     +obs.dt.minute.fillna(0)+obs.dt.second.fillna(0)/60.).astype(np.float32)
    df["ANON_ADMISSION_ID"]=df["ANON_ADMISSION_ID"].astype("int32")
    df.sort_values(["ANON_ADMISSION_ID","t_minutes"],inplace=True); df.reset_index(drop=True,inplace=True)
    return df


def curves(score, times, gs, ge, e_pat, e_row):
    """Per-patient detection & lead as a function of threshold, plus neg alert rate."""
    n=len(gs)
    onset=np.full(n,np.nan)
    pre_cummax=[None]*n; pre_times=[None]*n
    peak=np.empty(n)
    for g in range(n):
        s,e=gs[g],ge[g]
        tt=times[s:e]; sc=score[s:e]
        peak[g]=sc.max()
        ew=np.where(e_row[s:e]==1)[0]
        if len(ew):
            on=tt[ew[0]]; onset[g]=on
            mask=tt<=on
            if mask.any():
                ts=tt[mask]; ss=sc[mask]
                order=np.argsort(ts); ts=ts[order]; ss=ss[order]
                pre_cummax[g]=np.maximum.accumulate(ss); pre_times[g]=ts
    ev=np.where(e_pat==1)[0]; ne=np.where(e_pat==0)[0]
    neg_peak=peak[ne]
    # threshold grid spanning the score range
    thr=np.quantile(score, np.linspace(0.50,0.9995,80))
    thr=np.unique(np.round(thr,4))
    det=[]; lead=[]; falsealarm=[]
    for T in thr:
        d=0; leads=[]
        for g in ev:
            cm=pre_cummax[g]
            if cm is None or cm[-1]<T: continue
            idx=np.searchsorted(cm,T,side="left")
            d+=1; leads.append((onset[g]-pre_times[g][idx])/60.0)
        det.append(d/len(ev))
        lead.append(np.median(leads) if leads else np.nan)
        falsealarm.append(np.mean(neg_peak>=T))
    return np.array(thr),np.array(det),np.array(lead),np.array(falsealarm)


def main():
    t0=time.time(); print("Loading…"); df=load(); print(f"  {len(df):,} rows ({time.time()-t0:.0f}s)")
    vitals=es.VITALS_BASE+[es.ACVPU]
    luts=dict({v:es.build_lut(v) for v in vitals}); luts["blood_pressure"]=es.build_lut("blood_pressure")
    pv=es.apply_luts(df,luts,vitals)
    times=df["t_minutes"].values.astype(np.float64)
    gs,ge=es.group_boundaries(df["ANON_ADMISSION_ID"].values)
    e_row=df["EVENT_FLAG"].values; e_pat=np.maximum.reduceat(e_row,gs)
    news2=df["NEWS-2"].values.astype(np.float64)
    ew=es.compute_ewma(times,pv,gs,ge,vitals,{v:0.1 for v in vitals},{v:es.EWMA_REF_DEFAULT for v in vitals})
    snap=es.snapshot_score(pv,vitals,"additive",1.0)
    temp=es.temporal_score(pv,ew,vitals,0.5,1.0,"raise_only","additive",2.0,"relative")

    systems={"NEWS-2":news2,"Snapshot+ACVPU":snap,"Temporal+ACVPU(best)":temp}
    cv={name:curves(sc,times,gs,ge,e_pat,e_row) for name,sc in systems.items()}

    # Report median lead + false-alarm at MATCHED detection levels
    targets=[0.60,0.70,0.80,0.90]
    rows=[]
    for name,(thr,det,lead,fa) in cv.items():
        order=np.argsort(det)
        for D in targets:
            li=float(np.interp(D,det[order],lead[order]))
            fi=float(np.interp(D,det[order],fa[order]))
            rows.append(dict(system=name,detection=D,median_lead_h=round(li,1),false_alarm_rate=round(fi,3)))
    tab=pd.DataFrame(rows)
    tab.to_csv(OUT/"matched_detection_leadtime.csv",index=False)
    print("\n=== Median lead time (h) at MATCHED detection accuracy ===")
    print(tab.pivot_table(index="detection",columns="system",values="median_lead_h").to_string())
    print("\n=== False-alarm rate (non-event patients alerting) at same points ===")
    print(tab.pivot_table(index="detection",columns="system",values="false_alarm_rate").to_string())

    # plot: detection vs lead
    fig,ax=plt.subplots(1,2,figsize=(14,5.5))
    for name,(thr,det,lead,fa) in cv.items():
        o=np.argsort(det)
        ax[0].plot(det[o]*100,lead[o],marker="o",ms=3,label=name)
        ax[1].plot(det[o]*100,fa[o]*100,marker="o",ms=3,label=name)
    ax[0].set_xlabel("Detection accuracy (% event patients caught before onset)")
    ax[0].set_ylabel("Median lead time (h)"); ax[0].set_title("Lead time vs detection"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].set_xlabel("Detection accuracy (%)"); ax[1].set_ylabel("False-alarm rate (% non-event patients)")
    ax[1].set_title("False-alarm cost vs detection"); ax[1].grid(alpha=.3); ax[1].legend()
    fig.suptitle("Time-to-alert at matched detection accuracy (best fuzzy vs NEWS-2)",fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"matched_detection.png",dpi=170,bbox_inches="tight")
    print(f"\nSaved → {OUT}")

if __name__=="__main__": main()
