# Fuzzy EWS

Fuzzy Early Warning Score — six vitals through sigmoid membership functions, centroid
defuzzification, additive aggregation, plus an EWMA + worsening-trend temporal layer.

## Quick start

```bash
pip install -r requirements.txt

streamlit run app/streamlit_app.py          # interactive scorer
pytest tests/                               # engine tests
```

## Repository layout

Every file is either the engine, the app, one pipeline stage, or a test. There is exactly
one script per job.

```
streamlit_app.py            Root entry point — Streamlit Cloud requires it here; it loads app/
app/streamlit_app.py        The interactive scorer (Snapshot + Temporal tabs)

engine/
  engine_scoring.py         THE scoring engine: membership LUTs, defuzz, EWMA, trend, NEWS-2
  stats.py                  DeLong AUROC CIs and paired AUROC tests
  diagnostics.py            Shared metric helpers
  build_target_dataset.py   One-time: joins raw observations with outcome labels

pipeline/
  common.py                 Shared loader, LUTs, cohort, α/β/γ grid, lead-time definition
  grid_search_main.py       α/β/γ grid search — main dataset, death/ICU/event
  grid_search_annotated.py  α/β/γ grid search — annotated dataset, event only
  validation.py             AUROC, AUPRC, sens/spec and lead time (reads the grid winner)
  sensitivity_oat.py        One-at-a-time α/β/γ sensitivity, row level
  build_scored_spreadsheet_main.py       500k scored-observation spreadsheet
  build_scored_spreadsheet_annotated.py  the same for the annotated dataset

membership_functions/sigmoid/   The membership function CSVs actually used
datasets/                       Input data (large files gitignored)
results/                        Outputs, one folder per pipeline stage (gitignored)
tests/
```

### Run order

```bash
python pipeline/grid_search_main.py        # → results/main_dataset/…/grid_search/
python pipeline/validation.py              # → results/main_dataset/… (needs the grid winner)
python pipeline/grid_search_annotated.py   # → results/annotated_dataset/
python pipeline/sensitivity_oat.py         # → results/sensitivity_one_at_a_time/
python pipeline/build_scored_spreadsheet_main.py       # → fuzzy_ews_scored_500k.xlsx
python pipeline/build_scored_spreadsheet_annotated.py  # → fuzzy_ews_scored_annotated.xlsx
```

`results/_superseded/` holds output from removed scripts and from before the model changes
below. Nothing reads it — delete it when you no longer want the history (it is ~1.8 GB and
is not in git).

## The model

Six scored vitals: heart rate, systolic BP, temperature, respiratory rate, SpO2, inspired
oxygen. Each is passed through its sigmoid membership functions → four concern levels →
centroid defuzzification → a 0–3 concern score, summed to a 0–18 total.

The temporal layer smooths each vital's score with an EWMA (never downward: it takes
`max(EWMA, latest)`), then pushes it toward 3 by a sigmoid factor of the OLS slope of the
raw scores over a look-back window. The slope must clear a dead zone and hold across two
consecutive readings, so a single noisy uptick cannot inflate the score. Improving or
stable trends produce no adjustment, which makes the temporal total structurally ≥ the
snapshot total.

### Decisions baked in

- **Exact-zero defuzz (A1)** — a fully normal vital scores exactly 0. Depends on the
  `MIN_FIRING = 0.05` gate, which is duplicated in `engine_scoring.py` and
  `app/streamlit_app.py` (twice) and must stay in sync.
- **Systolic BP has five sets, not seven (C1)** — above-mild and above-moderate are
  absorbed into a widened No concern, leaving above-severe as the only above-normal set
  (`_merge_sbp_no_concern`). Sustained hypertension therefore scores 0 up to ~170 mmHg;
  only a crisis registers. This is the only SBP model.
- **ACVPU is never scored** — not a vital, no bonus, and no consciousness sub-score even
  in the NEWS-2 baseline, so the two systems are matched on inputs. Any non-Alert reading
  flags the whole row as deterioration instead (`acvpu_deterioration_flag`).
- **Inspired oxygen keeps its recorded units** — FiO2% and supplementary flow (L/min) are
  never interconverted. Each row is scored on the membership function for its own unit;
  flow uses `supplementary_oxygen_lmin_membership_functions.csv`. The main dataset instead
  carries a clinical category (`INSP_O2_CAT`), which is used directly.
- **Temporal layer covers all six vitals**, inspired oxygen included
  (`TEMPORAL_VITALS_DEFAULT`, defined once in the engine).

NEWS-2 is computed as the comparator with chronic-respiratory-aware Scale 1/2 SpO2
thresholds. Its ceiling here is 17, not 20, because consciousness scores 0.
