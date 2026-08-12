# Fuzzy EWS — Testing Summary & Comparison

A plain-English summary of everything we have tested. Numbers are **AUROC** (higher =
better discrimination). Two reference systems throughout: **NEWS-2** (the clinical
standard — note it also uses consciousness/ACVPU) and **Snapshot Fuzzy** (our fuzzy score
with no temporal memory). "Temporal" is the fuzzy score with the time-decay excess-EWMA
memory.

Everything lives in three result trees:
- `previous_results/` — frozen earlier work (row-level, patient-level, uncapped).
- `improved_results/` — the new improvement experiments (this round).
- raw machine-readable numbers: `improved_results/ALL_RESULTS.csv`.

---

## 1. The headline

1. **Row-level vs patient-level matters more than any model tweak.** The same model scores
   ~0.82 at row level but ~0.90 at patient level (peak score per admission). Patient-level
   is the meaningful early-warning metric.
2. **The temporal memory's real value is EARLIER warning, not higher AUROC.** At a matched
   false-alert rate, the temporal system flags deteriorating patients **~12 hours earlier**
   than NEWS-2 or snapshot — a benefit AUROC structurally cannot show (see §4).
3. **Of the new ideas tested, only "sharper SBP" gave an AUROC gain** (small but consistent,
   best on ICU). Aggregation-method and per-vital-memory changes gave no gain — useful
   negative results.
4. **The fidelity fixes restored interpretability at no cost:** a perfectly normal patient
   now scores **0** (was ~1.5 from a defuzzification offset), with AUROC unchanged.

---

## 2. Best numbers so far (patient-level AUROC)

| System | Event-24h | ICU-24h | Death-24h | Notes |
|---|---|---|---|---|
| NEWS-2 | **0.914** | 0.818 | **0.922** | clinical standard, *includes consciousness* |
| Snapshot Fuzzy (6 vitals) | 0.896 | 0.806 | 0.898 | no temporal, no ACVPU |
| **Temporal, best (6 vitals)** | 0.897 | 0.827 | 0.899 | corrected + sharper SBP |
| **Temporal, best + ACVPU** | **0.905** | **0.828** | **0.909** | closest to NEWS-2 |

Takeaway: the fuzzy system **matches NEWS-2 on ICU** and lands ~1 point behind on
event/death — the gap is essentially the consciousness component. Adding ACVPU closes most
of it while keeping the model interpretable.

---

## 3. What each improvement did (the experiment ladder)

Each row adds one idea on top of the fixes. Values = patient-level **event** AUROC,
6 vitals (Temporal system).

| Experiment | Idea tested | Event AUROC | Verdict |
|---|---|---|---|
| `01_fidelity` | A1 exact-zero defuzz + A2 ACVPU map + A3 shared engine | 0.8935 | ✅ interpretability fixed, AUROC unchanged |
| `02_aggregation` | additive vs multiplicative (noisy-OR) vs power-mean | 0.8935 | ➖ additive still wins — no gain |
| `03_sharper_sbp` | steeper hypertension membership (engine's `custom_mf_sbp_sharper`) | 0.8964 | ✅ small consistent gain (best on ICU: +0.007) |
| `04_pervital_temporal` | per-vital EWMA memory (temp slow, HR fast) + relative excess | 0.8936 | ➖ no AUROC gain (changes the lead-time tradeoff, §4) |
| `05_combined_best` | additive + sharper SBP + relative excess | **0.8974** | ✅ best overall fuzzy |

**Interpretation:** the model's discriminative power comes from the snapshot (instantaneous
concern) plus ACVPU plus a sharper BP curve. Fancier aggregation and per-vital temporal
memory don't improve ranking — additive equal-weight is hard to beat and is the most
interpretable choice, so we keep it.

---

## 4. Time-to-alert — where temporal actually wins ★

Row/patient AUROC can't reward *earlier* alerts (an early temporal flag lands on an
observation still labelled "negative", so it looks like a false positive). So we measured
**lead time** directly: set each system's threshold so the same fraction (20%) of
non-event patients ever alert, then for event patients measure how early the first alert
fires before the deterioration window.

| System | Detected (%) | Median lead time |
|---|---|---|
| NEWS-2 | 82% | 81.6 h |
| Snapshot Fuzzy | 68% | 81.8 h |
| **Temporal Fuzzy** | 65% | **94.1 h** |

**The temporal system alerts ~12 h earlier** than snapshot/NEWS-2 at the same alert budget
— the clinically valuable property the AUROC tables hide. The trade-off: at this threshold
it catches slightly fewer patients (65% vs 82%). The per-vital/relative-excess variant
(`05`) shifts the trade-off — 68% detected, ~86 h lead — i.e. you can tune toward
"earlier but fewer" or "more but later."

*(Absolute lead-times are large because the event-onset marker is the start of the 24 h
window and many admissions are long; the ~12 h **relative** advantage is the real result.)*

---

## 5. Recommendation

- **Keep:** additive aggregation (interpretable, best), the corrected defuzz (normal = 0),
  the time-decay excess-EWMA temporal layer, and the **sharper SBP** membership.
- **Adopt ACVPU** if a consciousness input is acceptable in the final product — it closes
  most of the gap to NEWS-2 and is fully interpretable.
- **Drop:** multiplicative/power-mean aggregation and per-vital temporal memory — no benefit.
- **Lead with the time-to-alert story**, not AUROC: the fuzzy temporal system's selling
  point is earlier warning at matched specificity, plus full interpretability (every
  point traces to one vital's concern level).

---

## 6. Where to find things

| Folder | Contents |
|---|---|
| `previous_results/baseline_results/` | row-level AUROC (frozen) |
| `previous_results/patient_level_results/` | patient-level AUROC (frozen) |
| `previous_results/uncapped_results/` | uncapping experiment (frozen) |
| `improved_results/0X_*/` | each experiment: `grid_results.csv`, `auroc_event_optimal.csv`, `auroc_sherifs.csv`, `summary_bars.png`, `time_to_alert.csv/.png`, `best_config.json` |
| `improved_results/ALL_RESULTS.csv` | every number in one tidy file |
| `experiment_code/` | the new scripts (`engine_scoring.py`, `run_all_experiments.py`); core root scripts left unchanged |

Each experiment folder also reports the same two parameter sets as before:
**event-optimal** (grid best) and **Sherif's** (α=0.5, β=5.0, γ=0.75), at both row and
patient level, with and without ACVPU.

---

## 7. Methods caveat — why Snapshot and Temporal AUROC are nearly identical ★

**Question raised:** the patient-level AUROCs for Snapshot and Temporal differ by only
~0.0015 across *every* experiment, even though they are different scores. Why?

**Answer (measured, not asserted):** AUROC depends only on the *rank order* of patients.
At the patient level both systems are reduced to a **peak score over the stay**, and in
`raise_only` mode the temporal score is **floored at snapshot** (`temp = max(total, snap)`),
so temporal is, by construction, "snapshot + a small non-negative bump." The two
patient-level rankings are therefore almost the same:

| Coupling of Snapshot-peak vs Temporal-peak | Value |
|---|---|
| Spearman rank correlation | **0.9953** |
| Patients with identical score | 23% |
| Patients whose peak is the *same observation* row | 91% |
| DeLong test, event target (N≈39k) | Δ = **+0.0015**, p < 0.001 |

So the difference is **statistically real but clinically negligible**: with ~39,000
patients even a 0.0015 gap is significant (p < 0.001), yet it is an order of magnitude
below any meaningful threshold (≈0.02). Two scores rank-correlated at 0.995 *cannot*
produce materially different AUROCs — this is a property of the evaluation, not a bug.

**We then tried hard to make Temporal genuinely distinct** (see `results/trajectory/`):

| Intervention | What it changes | Event AUROC effect |
|---|---|---|
| `boost_mode=uncapped` | stop the per-vital [0,3] clip deleting the boost at the peak | +0.00003 (none) |
| EWMA `seed=0` | give short stays a baseline so the first reading shows excess | **−0.008 (hurts)** |
| Trajectory aggregates (mean/area/slope/time-above/score-at-lead) instead of peak | keep the deterioration *path*, not just its height | all **worse** than peak |
| `excess_only` channel (raw-free, unfloored) | a score orthogonal to snapshot | 0.827 standalone (weaker) |
| Re-grid: full β-sweep × every boost_mode | give the optimiser its best chance | event range ≤ **0.006** (flat) |

These **decouple the rankings** (Spearman falls to 0.94–0.96 under slope/time-above) but
**do not improve discrimination**. The one apparent large gain — ICU `area` jumping from
0.66 to 0.77 — is a **stay-length artefact** (long ICU stays accumulate area); Snapshot's
`area` jumps identically, so it is not a temporal effect.

**Conclusion.** The temporal layer adds **earlier warning (timing)**, not **patient-level
discrimination (AUROC)** — exactly consistent with §4, where the trajectory-aware
lead-time analysis is the only place temporal wins. Reporting the Snapshot-vs-Temporal
AUROC difference as a "win" overstates it; the honest framing is the lead-time result plus
this coupling/DeLong caveat. Reproduce with `python trajectory_auroc_patient.py`.

*(Numbers above are from a 38,934-patient sample: all event patients + 25,000 controls,
event-optimal params α=0.1, β=1.0, γ=1.0, relative excess. `results/current/` is unchanged.)*
