# Fuzzy EWS — Repository Contents

---

## Root

| File | Purpose |
|------|---------|
| `grid_search_excess_patient.py` | **Run first.** Sweeps 2,000 parameter combinations (α × β × γ × excess_mode) to find the EWMA config that maximises patient-level event AUROC. Writes to `results/current/grid_search/`. |
| `auroc_target_comparison_patient.py` | **Run second.** Loads grid results, evaluates all config leaves at patient- and row-level. Outputs AUROC, AUPRC, sens/spec, and a lead-time table to `results/current/`. |
| `trajectory_auroc_patient.py` | **Run third (optional).** Tests why Snapshot ≈ Temporal at the patient level: boost_mode / EWMA-seed / excess-only / trajectory aggregates, with a rank-coupling diagnostic and DeLong significance test. Writes to `results/trajectory/` and `results/headroom/`; **does not touch `results/current/`**. See MODEL_COMPARISON.md §7. |
| `README.md` | Quick-start and system overview. |
| `CONTENTS.md` | This file. |
| `MODEL_COMPARISON.md` | Plain-English summary of all experiments and key findings. |
| `requirements.txt` | Python dependencies. |

---

## `engine/`

All scoring logic and experiment scripts.

| File | Purpose |
|------|---------|
| `engine_scoring.py` | **Single source of truth.** A1 (exact-zero defuzz), A2 (ACVPU map), `build_lut()` (optional sharper SBP), `compute_ewma()` (time-decay, optional `seed`), `temporal_score()` (now with `temporal_vitals` + `boost_mode`), `snapshot_score()`, `excess_score()` + `blend_scores()` (raw-free temporal channel), `patient_peak()` / `patient_aggregate()` (trajectory summaries), `auroc()`. All three root scripts call into here. |
| `stats.py` | DeLong's test for two correlated ROC curves — gives Δ-AUROC a CI and p-value. |
| `diagnostics.py` | Rank-coupling table (Spearman, %identical, %same-peak-row) and stay-length-stratified AUROC. |
| `build_target_dataset.py` | One-time pipeline: joins raw observations with outcome labels to produce `datasets/final_observations_with_targets.csv`. Re-run only if source data changes. |
| `experiments/run_all_experiments.py` | Improvement-experiment ladder driver (exp 01–07). Already run — historical reference. |
| `experiments/run_oxygen_fix.py` | Standalone exp 07 (O2 categorical fix). Already run. |
| `experiments/run_supplemental_o2.py` | Standalone exp 06 (supplemental O2 exploration). Already run. |
| `experiments/matched_time_to_alert.py` | Lead-time analysis used by the experiment ladder. Already run. |

---

## `app/`

Interactive scoring demo.

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Streamlit web app — enter vitals, see fuzzy concern score with per-vital breakdown. |
| `grid_search_auroc.py` | Original (pre-engine) grid search. Superseded by `grid_search_excess_patient.py`. |

---

## `tests/`

Pytest suite (the first in the repo) locking engine behaviour and the consolidation:
`test_engine_scoring.py` (A1 zero-rule, raise-only floor, `temporal_vitals` exclusion,
boost_mode bounds, EWMA seed, trajectory aggregates, and a **regression** check that the
consolidated `temporal_score` reproduces the old inline maths bit-for-bit) and
`test_stats_diagnostics.py` (DeLong vs sklearn AUC, coupling helpers). Run: `pytest tests/`.

---

## `membership_functions/`

Lookup tables mapping each vital to fuzzy concern levels.

| Subfolder | Contents |
|-----------|---------|
| `sigmoid/` | **Active set.** Sigmoid MFs for all 7 vitals — what `engine_scoring.build_lut()` reads. |
| `trapezoidal/` | Alternative trapezoidal MFs. Not used in the current system. |
| `original/` | First-version MF definitions (pre-sigmoid redesign). Historical only. |

---

## `datasets/`

| File | Purpose |
|------|---------|
| `final_observations_with_targets.csv` | Primary validation dataset — 9.3M observations, 392,931 admissions, with `DEATH_WITHIN_24H` / `ICU_WITHIN_24H` / `EVENT_FLAG`. Gitignored (large). |
| `final_observations_with_scores.csv` | Pre-scored version including NEWS-2. Gitignored (large). |
| Training CSVs / XLSX files | Source annotation and training splits. Gitignored. |

---

## `results/`

All outputs in one place.

### `results/current/` — live validated results

```
current/
├── grid_search/                grid_results.csv + heatmaps + sensitivity plots
├── raise_only/
│   ├── event_optimal/          metrics_full.csv, auroc_table.csv, ROC/PR plots, bar charts
│   │   └── acvpu/              same but with ACVPU as 7th vital
│   └── sherifs_params/         same for Sherif's fixed params (α=0.5, β=5.0, γ=0.75)
│       └── acvpu/
├── bidirectional/              same structure, bidirectional temporal mode
│   ├── event_optimal/
│   └── sherifs_params/
├── all_metrics_summary.csv     all 144 metric rows (8 leaves × 3 targets × 2 levels × 3 systems)
├── lead_time_table.csv         median lead time at 60/70/80/90% detection
└── lead_time_plot.png
```

**Headline results (event-optimal, +ACVPU, raise-only, patient-level):**

| System | Death AUROC | ICU AUROC | Event AUROC | Event AUPRC |
|--------|------------|-----------|-------------|-------------|
| NEWS-2 | 0.92176 | 0.81791 | 0.91447 | 0.36292 |
| Snapshot Fuzzy | 0.91249 | 0.82776 | 0.90642 | 0.37338 |
| Temporal Fuzzy | **0.91390** | **0.83114** | **0.90798** | **0.38391** |

Temporal beats Snapshot on every target. Temporal beats NEWS-2 on ICU AUROC and all AUPRC targets. At 80% detection temporal fires **4.6 hours earlier** than snapshot.

> ⚠️ The Temporal-vs-Snapshot AUROC gap is ~0.0015 because the two patient-level scores are rank-correlated at **Spearman 0.995** (peak aggregation + snapshot floor). It is statistically significant (DeLong p < 0.001 at N≈39k) but clinically negligible. Temporal's real benefit is **lead time**, not AUROC. Full analysis: **MODEL_COMPARISON.md §7** / `results/trajectory/`.

### `results/experiments/` — improvement experiment ladder (01–07)

Each subfolder is one experiment. Each adds one layer of improvement and contains `grid_results.csv`, `auroc_event_optimal.csv`, `auroc_sherifs.csv`, `time_to_alert.csv`, and summary plots. `ALL_RESULTS.csv` aggregates everything.

| Folder | Change | Finding |
|--------|--------|---------|
| `01_fidelity/` | A1 + A2 + A3 fixes | Normal patient now scores 0; AUROC unchanged |
| `02_aggregation/` | Aggregation method search | Additive remains best |
| `03_sharper_sbp/` | Sharper SBP MF (C1) | Small consistent gain, esp. ICU |
| `04_pervital_temporal/` | Per-vital EWMA α | No gain over global α |
| `05_combined_best/` | Best of 01–04 combined | Temporal marginally beats snapshot |
| `06_supplemental_o2/` | O2 from broken L/min field | No improvement |
| `07_oxygen_fix/` | O2 from `INSP_O2_CAT` | 16.4% of rows correctly scored on O2 |

### `results/archive/` — frozen outputs from earlier work

| Subfolder | Contents |
|-----------|---------|
| `baseline/` | Original row-level validation (pre-patient-level work). Includes original scripts. |
| `patient_level_pre_fix/` | Patient-level results before A1/A2/C1/O2 fixes were applied. |
| `uncapped/` | Row/patient results without the 500k non-event row cap (bias check). |

### `results/trajectory/` — Snapshot-vs-Temporal coupling analysis (from `trajectory_auroc_patient.py`)

| File | Contents |
|------|----------|
| `auroc_variants.csv` | AUROC for every (score variant × patient aggregate × target). |
| `coupling.csv` | Spearman / %identical / %same-peak-row for Snapshot vs each temporal variant. |
| `delong.csv` | DeLong Δ-AUROC and p-value vs Snapshot-peak for every variant. |
| `stratified_auroc.csv` | Event AUROC by stay-length bin. |
| `blend_sweep.csv` | AUROC of (1−w)·Snapshot ⊕ w·Excess as w goes 0→1. |

### `results/headroom/` — boost_mode / EWMA-seed / β re-grid

| File | Contents |
|------|----------|
| `boost_seed_comparison.csv` | Peak AUROC for boost_mode {clip, headroom, uncapped} × seed {raw0, 0}. |
| `grid_search/grid_results.csv` | β-sweep × boost_mode at the event-optimal α/γ/excess (surface stays flat). |

---

## `archive/`

Pre-fix script snapshots, saved before any improvements were applied.

| File | Purpose |
|------|---------|
| `original_system/grid_search_excess_patient.py` | Grid search as it was before all fixes. |
| `original_system/auroc_target_comparison_patient.py` | Validation script before fixes. |

---

## Script dependency flow

```
datasets/final_observations_with_targets.csv
        │
        ▼
grid_search_excess_patient.py   (imports engine/engine_scoring.py)
        │
        └─► results/current/grid_search/grid_results.csv
                        │
                        ▼
auroc_target_comparison_patient.py   (imports engine/engine_scoring.py)
        │
        ├─► results/current/raise_only/{event_optimal,sherifs_params}/{,acvpu}/
        ├─► results/current/bidirectional/{event_optimal,sherifs_params}/{,acvpu}/
        ├─► results/current/all_metrics_summary.csv
        ├─► results/current/lead_time_table.csv
        └─► results/current/lead_time_plot.png
```
