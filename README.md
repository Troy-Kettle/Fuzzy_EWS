# Fuzzy EWS

Fuzzy Early Warning Score system — per-vital sigmoid membership functions, centroid
defuzzification, additive aggregation, and an excess-EWMA temporal layer.

## Quick start

```bash
pip install -r requirements.txt

# Interactive scorer
streamlit run fuzzy_system/streamlit_app.py

# Re-run parameter optimisation
python grid_search_excess_patient.py

# Full validation (AUROC, AUPRC, sens/spec, lead time)
python auroc_target_comparison_patient.py

# Why Snapshot ≈ Temporal at patient level (coupling + DeLong + trajectory tests)
python trajectory_auroc_patient.py

# Engine tests
pytest tests/
```

## Repository layout

See [CONTENTS.md](CONTENTS.md) for a detailed description of every folder and file.

```
engine/                     Scoring engine (engine_scoring.py) + experiment scripts
app/                        Streamlit interactive scorer
membership_functions/       Membership function CSVs (sigmoid, trapezoidal, original)
datasets/                   Validation data (large files kept locally, gitignored)
results/current/            Live validated results (grid search + full metrics)
results/experiments/        Improvement-experiment ladder outputs (exp 01–07)
results/archive/            Frozen outputs from earlier work
archive/                    Pre-fix copies of the primary scripts
```

## System overview

Each observation is scored by passing six vital signs through sigmoid membership
functions → 4 concern levels → centroid defuzzification → crisp 0–3 concern score per
vital → additive sum (0–18 range, or 0–21 with ACVPU).

The temporal layer (excess-EWMA) applies a time-decaying EWMA baseline to the five
continuously-varying physiological vitals (HR, SBP, RR, SpO2, Temperature). When a
vital rises above its running baseline the excess is amplified, giving earlier alerts
than a pure snapshot score. Inspired O2 and ACVPU are categorical step-signals and
contribute at snapshot level only.

> **Note on Snapshot vs Temporal AUROC.** At the patient level both systems are reduced
> to a peak score and temporal is floored at snapshot, so they rank patients at Spearman
> ≈ 0.995 and their AUROCs differ by only ~0.0015 (significant at N≈39k but clinically
> negligible). The temporal layer's real benefit is **earlier warning (lead time)**, not
> AUROC. See `MODEL_COMPARISON.md` §7 and `results/trajectory/`.

Key correctness fixes applied over the original baseline:
- **A1** — exact-zero defuzz rule: a fully normal vital scores 0 (was ~0.25)
- **A2** — canonical ACVPU map: Alert=0, Voice=1, Confused=2, Pain/Unresponsive=3
- **C1** — sharper SBP membership: hypertension above mild collapses to a single bucket
- **O2** — inspired O2 scored from `INSP_O2_CAT` (recorded clinical category) instead
  of `INSPIRED_O2_TEXT` which mixed L/min flow rates with FiO2 percentages
