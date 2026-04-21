#!/usr/bin/env python3
"""Generate a PDF documenting how the grid search and AUROC evaluation scripts work."""

from fpdf import FPDF
from pathlib import Path

OUTPUT = Path(__file__).parent / "Fuzzy_EWS_Grid_Search_and_AUROC_Documentation.pdf"
IMG_DIR = Path(__file__).parent / "grid_search_results"


class DocPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Fuzzy EWS - Grid Search & AUROC Documentation", align="R")
            self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 60, 120)
        self.ln(4)
        self.cell(0, 10, title)
        self.ln(12)
        self.set_draw_color(30, 60, 120)
        self.line(self.l_margin, self.get_y() - 2, self.w - self.r_margin, self.get_y() - 2)
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(50, 80, 140)
        self.ln(2)
        self.cell(0, 8, title)
        self.ln(10)

    def sub_sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(70, 70, 70)
        self.cell(0, 7, title)
        self.ln(8)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def code_block(self, text):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        w = self.w - self.l_margin - self.r_margin
        self.set_x(x + 4)
        self.multi_cell(w - 8, 4.5, text, fill=True)
        self.ln(3)

    def bullet(self, text, bold_prefix=""):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        x0 = self.get_x()
        self.cell(6, 5.5, "-")
        if bold_prefix:
            self.set_font("Helvetica", "B", 10)
            self.cell(self.get_string_width(bold_prefix) + 1, 5.5, bold_prefix)
            self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def formula(self, text):
        self.set_font("Courier", "", 9.5)
        self.set_text_color(100, 30, 30)
        self.cell(0, 6, f"    {text}")
        self.ln(8)

    def table_row(self, cells, bold=False, fill=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        if fill:
            self.set_fill_color(230, 235, 245)
        col_w = (self.w - self.l_margin - self.r_margin) / len(cells)
        for cell in cells:
            self.cell(col_w, 7, str(cell), border=1, fill=fill, align="C")
        self.ln()

    def add_image_page(self, img_path, caption):
        if not Path(img_path).exists():
            return
        self.add_page()
        self.sub_title(caption)
        avail_w = self.w - self.l_margin - self.r_margin
        self.image(str(img_path), x=self.l_margin, w=avail_w)


def build_pdf():
    pdf = DocPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)

    # ── TITLE PAGE ──────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 15, "Fuzzy EWS", align="C")
    pdf.ln(18)
    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 12, "Grid Search & AUROC Evaluation", align="C")
    pdf.ln(12)
    pdf.cell(0, 12, "Technical Documentation", align="C")
    pdf.ln(30)
    pdf.set_draw_color(30, 60, 120)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "How the temporal context builder parameters are optimised", align="C")
    pdf.ln(6)
    pdf.cell(0, 8, "and how predictive performance is evaluated", align="C")

    # ── TABLE OF CONTENTS ───────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Table of Contents")
    toc = [
        "1. Overview",
        "2. The Three Parameters",
        "3. Grid Search Script (grid_search_auroc.py)",
        "    3.1 Pipeline Architecture",
        "    3.2 Stage 1: Data Loading & Cleaning",
        "    3.3 Stage 2: Pre-compute Fuzzy Scores",
        "    3.4 Stage 3: Pre-compute OLS Trend Slopes",
        "    3.5 Stage 4: The Grid Search Loop",
        "    3.6 Stage 5: Scoring Formula",
        "    3.7 Output & Visualisations",
        "4. AUROC Evaluation Script (auroc_optimal.py)",
        "    4.1 What It Does Differently",
        "    4.2 Point-Estimate AUROC",
        "    4.3 Patient-Level Bootstrap CIs",
        "    4.4 Sensitivity at Fixed Specificity",
        "    4.5 Output Plots",
        "5. Key Formulas Reference",
        "6. Appendix: Grid Search Visualisations",
    ]
    for item in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 7, item)
        pdf.ln(7)

    # ── 1. OVERVIEW ─────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("1. Overview")
    pdf.body_text(
        "The Fuzzy Early Warning System (EWS) uses fuzzy logic to convert raw vital-sign "
        "readings into concern scores, then applies a temporal context builder to incorporate "
        "trend information. Two scripts handle optimisation and evaluation:"
    )
    pdf.bullet(
        "Searches over a 3D parameter grid to find the combination of alpha, beta, "
        "and gamma that maximises AUROC for predicting REVIEW_WITHIN_4HOURS.",
        bold_prefix="grid_search_auroc.py: "
    )
    pdf.bullet(
        "Takes the winning parameters and runs a deeper evaluation with bootstrap "
        "confidence intervals, sensitivity/specificity analysis, and publication-quality plots.",
        bold_prefix="auroc_optimal.py: "
    )
    pdf.ln(4)
    pdf.body_text(
        "Both scripts compare three predictors against the same binary label "
        "(REVIEW_WITHIN_4HOURS = 1 if the patient needed clinical review within 4 hours):"
    )
    pdf.bullet("NEWS-2 (pre-computed integer score from the dataset)")
    pdf.bullet("Snapshot Fuzzy EWS (sum of 6 per-vital fuzzy scores, no temporal adjustment)")
    pdf.bullet("Temporal Context Builder (EWMA + trend sigmoid + aggregation blend)")

    # ── 2. THE THREE PARAMETERS ─────────────────────────────────────
    pdf.add_page()
    pdf.section_title("2. The Three Parameters")

    pdf.sub_sub_title("Alpha (EWMA smoothing factor)")
    pdf.body_text(
        "Controls how much weight the latest observation receives versus historical "
        "observations. At alpha=1.0 only the current observation matters (no smoothing). "
        "At alpha=0.1 the score is heavily smoothed, carrying forward historical values."
    )
    pdf.formula("EWMA[t] = alpha * raw[t] + (1 - alpha) * EWMA[t-1]")
    pdf.body_text("Grid: 0.1, 0.2, 0.3, ..., 1.0 (10 values, step 0.1)")

    pdf.sub_sub_title("Beta (sigmoid trend steepness)")
    pdf.body_text(
        "Controls how aggressively a worsening trend inflates the score. Higher beta means "
        "even small positive slopes produce a large trend adjustment. The trend factor uses "
        "a modified sigmoid that is zero when the slope is non-positive."
    )
    pdf.formula("trend_factor = 2 / (1 + exp(-beta * slope)) - 1    [only when slope > 0]")
    pdf.body_text("Grid: 0.5, 1.0, 1.5, ..., 5.0 (10 values, step 0.5)")

    pdf.sub_sub_title("Gamma (aggregation blend)")
    pdf.body_text(
        "Controls how the six per-vital adjusted scores are combined into a single total. "
        "At gamma=1.0 it is a simple additive sum (range 0-18). At gamma=0.0 it would be "
        "purely max-based (the single worst vital scaled to 0-18). Values in between blend "
        "the two strategies."
    )
    pdf.formula("total = (1 - gamma) * (18/3) * max_vital + gamma * sum(all_vitals)")
    pdf.body_text("Grid: 0.1, 0.2, 0.3, ..., 1.0 (10 values, step 0.1)")
    pdf.ln(4)
    pdf.body_text("Total search space: 10 x 10 x 10 = 1,000 parameter combinations.")

    # ── 3. GRID SEARCH SCRIPT ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("3. Grid Search Script (grid_search_auroc.py)")

    pdf.sub_title("3.1 Pipeline Architecture")
    pdf.body_text(
        "The script is designed for efficiency on a ~9.3 million row dataset. The key "
        "optimisation principle is: pre-compute everything that does not depend on the "
        "three search parameters (alpha, beta, gamma), so the inner loop only performs "
        "fast vectorised arithmetic."
    )
    pdf.ln(2)
    pdf.body_text("The pipeline has five stages:")
    pdf.code_block(
        "Stage 1: Load & clean data                    [one-time]\n"
        "Stage 2: Pre-compute per-vital fuzzy scores   [one-time, parameter-independent]\n"
        "Stage 3: Pre-compute OLS trend slopes         [one-time, parameter-independent]\n"
        "Stage 4: Grid search loop                     [1,000 iterations]\n"
        "Stage 5: Score each combination with AUROC    [inside loop]"
    )

    pdf.sub_title("3.2 Stage 1: Data Loading & Cleaning")
    pdf.body_text(
        "The script loads the training CSV and applies several cleaning steps:"
    )
    pdf.bullet("Coerces vital-sign columns to numeric (handles 'Refused', blanks, etc.)")
    pdf.bullet("Filters to rows where COMPLETE_DATA = 1")
    pdf.bullet("Drops rows with NaN vitals after coercion")
    pdf.bullet("Casts columns to efficient types (float32, int8) to reduce memory")
    pdf.bullet("Clamps inspired O2 to [21, 100]")
    pdf.bullet(
        "Constructs t_minutes (continuous timeline) from DAYS_SINCE_ADMISSION + OBS_TIME"
    )
    pdf.bullet("Sorts by patient ID then time, so per-patient operations work correctly")

    pdf.add_page()
    pdf.sub_title("3.3 Stage 2: Pre-compute Fuzzy Scores")
    pdf.body_text(
        "For each of the six vital signs, the script builds a lookup table (LUT) that maps "
        "every possible input value to a defuzzified concern score (range 0 to 3). The "
        "defuzzification process is:"
    )
    pdf.bullet(
        "Load the sigmoid membership functions from CSV files in the "
        "generated_membership_data/sigmoid/ directory."
    )
    pdf.bullet(
        "For each input value in the membership function grid, evaluate all membership "
        "degrees (e.g., 'Below normal - severe concern' through 'Above normal - severe concern')."
    )
    pdf.bullet(
        "Map the input memberships to four output concern levels: No concern, Mild, "
        "Moderate, Severe (using max-aggregation within each level)."
    )
    pdf.bullet(
        "Defuzzify using centroid method over output membership functions defined as "
        "trapezoids on the range [0, 3]."
    )
    pdf.ln(2)
    pdf.body_text(
        "Once the LUT is built, all 9.3M observations are scored via numpy interpolation "
        "(np.interp), which is nearly instant. This avoids running the full fuzzification "
        "pipeline 9.3M times."
    )
    pdf.ln(2)
    pdf.body_text(
        "The six vitals and their membership function types are:"
    )
    pdf.table_row(["Vital", "Column", "MF Type"], bold=True, fill=True)
    rows = [
        ("Heart rate", "HEART_RATE", "7-variable"),
        ("Blood pressure", "SYSTOLIC_BP", "7-variable"),
        ("Temperature", "TEMPERATURE", "7-variable"),
        ("Respiratory rate", "RESP_RATE", "7-variable"),
        ("Oxygen saturation", "SATS_SPO2", "3-var (downward)"),
        ("Inspired oxygen", "INSPIRED_O2_TEXT", "3-var (upward)"),
    ]
    for r in rows:
        pdf.table_row(r)

    pdf.add_page()
    pdf.sub_title("3.4 Stage 3: Pre-compute OLS Trend Slopes")
    pdf.body_text(
        "For each vital sign and each observation, the script computes the ordinary least "
        "squares (OLS) slope of the fuzzy score within a 24-hour look-back window. This "
        "captures whether the vital is trending better or worse."
    )
    pdf.ln(2)
    pdf.body_text("The computation works as follows:")
    pdf.bullet(
        "For each patient, maintain a sliding window of observations within "
        "the past 24 hours (1440 minutes)."
    )
    pdf.bullet(
        "Convert times to hours relative to the window start, then compute "
        "the OLS slope: beta = sum((t - t_mean)(s - s_mean)) / sum((t - t_mean)^2)."
    )
    pdf.bullet("Require at least 2 observations in the window to compute a slope.")
    pdf.bullet("A positive slope means the concern score is increasing (patient worsening).")
    pdf.ln(2)
    pdf.body_text(
        "This is the most expensive pre-computation step (iterates per-patient, "
        "per-observation), but it only runs once because slopes do not depend on "
        "alpha, beta, or gamma."
    )

    pdf.sub_title("3.5 Stage 4: The Grid Search Loop")
    pdf.body_text(
        "The three parameters are searched in a nested loop, ordered from most expensive "
        "to cheapest recomputation:"
    )
    pdf.code_block(
        "for alpha in [0.1, 0.2, ..., 1.0]:          <-- outermost (10 iters)\n"
        "    compute EWMA for all 6 vitals            <-- expensive: per-patient sequential\n"
        "    clamp: clamped = max(EWMA, raw_score)\n"
        "\n"
        "    for beta in [0.5, 1.0, ..., 5.0]:        <-- middle (10 iters)\n"
        "        compute trend-adjusted scores         <-- moderate: vectorised sigmoid\n"
        "\n"
        "        for gamma in [0.1, 0.2, ..., 1.0]:   <-- innermost (10 iters)\n"
        "            aggregate to single total          <-- cheap: weighted sum\n"
        "            compute AUROC                      <-- sklearn roc_auc_score"
    )
    pdf.ln(2)
    pdf.body_text(
        "Why this nesting order matters:"
    )
    pdf.bullet(
        "The EWMA depends only on alpha and requires a sequential pass over each patient's "
        "timeline. By placing alpha outermost, we compute the EWMA only 10 times total "
        "(once per alpha value), not 1,000 times.",
        bold_prefix="Alpha outermost: "
    )
    pdf.bullet(
        "The sigmoid trend adjustment depends on alpha (via the clamped EWMA) and beta. "
        "It is vectorised element-wise arithmetic, moderately cheap.",
        bold_prefix="Beta middle: "
    )
    pdf.bullet(
        "The gamma aggregation is just a weighted sum of arrays that are already computed. "
        "This is the cheapest operation, so it runs the most iterations (innermost).",
        bold_prefix="Gamma innermost: "
    )

    pdf.add_page()
    pdf.sub_title("3.6 Stage 5: Scoring Formula")
    pdf.body_text(
        "For a given (alpha, beta, gamma), the per-vital adjusted score is computed as:"
    )
    pdf.ln(2)
    pdf.sub_sub_title("Step A: EWMA Smoothing")
    pdf.formula("ewma[t] = alpha * raw[t] + (1 - alpha) * ewma[t-1]")
    pdf.body_text(
        "The EWMA is then clamped so it can only raise the score, never lower it:"
    )
    pdf.formula("clamped[t] = max(ewma[t], raw[t])")

    pdf.sub_sub_title("Step B: Trend Adjustment")
    pdf.body_text(
        "A sigmoid-based trend factor is applied only when the OLS slope is positive "
        "(worsening trend). The factor ranges from 0 (no trend) toward 1 (strong trend)."
    )
    pdf.formula("trend_factor = 2 / (1 + exp(-beta * slope)) - 1    [slope > 0 only]")
    pdf.body_text("The adjusted score combines the clamped EWMA with the trend factor:")
    pdf.formula("adjusted = clamped + trend_factor * (3.0 - clamped)")
    pdf.body_text(
        "The term (3.0 - clamped) means the trend pushes the score toward the maximum of 3. "
        "Scores already near 3 have less room to increase. The result is clipped to [0, 3]."
    )

    pdf.sub_sub_title("Step C: Aggregation Across Vitals")
    pdf.body_text("The six per-vital adjusted scores are combined using gamma:")
    pdf.formula("additive    = sum of all 6 adjusted scores        (range 0-18)")
    pdf.formula("max_based   = (18/3) * max(adjusted scores)       (range 0-18)")
    pdf.formula("total       = (1-gamma) * max_based + gamma * additive")
    pdf.body_text(
        "At gamma=1.0 the total is a simple sum. At lower gamma values, the single worst "
        "vital dominates the total score."
    )

    pdf.sub_sub_title("Step D: Evaluate with AUROC")
    pdf.body_text(
        "Each combination's total score array is evaluated with sklearn's roc_auc_score "
        "against the binary label REVIEW_WITHIN_4HOURS. The combination achieving the "
        "highest AUROC is selected as optimal."
    )

    pdf.add_page()
    pdf.sub_title("3.7 Output & Visualisations")
    pdf.body_text("The grid search produces the following outputs in grid_search_results/:")
    pdf.bullet(
        "CSV file with all 1,000 (alpha, beta, gamma, AUROC) combinations.",
        bold_prefix="grid_search_results.csv: "
    )
    pdf.bullet(
        "Three 2D heatmaps showing AUROC for each pair of parameters "
        "(alpha-beta, alpha-gamma, beta-gamma), with the third fixed at its optimal value.",
        bold_prefix="heatmaps.png: "
    )
    pdf.bullet(
        "Same pairwise views as 3D surface plots with the optimal point marked in red.",
        bold_prefix="surfaces_3d.png: "
    )
    pdf.bullet(
        "1D plots varying each parameter while fixing the other two at optimal.",
        bold_prefix="sensitivity.png: "
    )
    pdf.bullet(
        "Horizontal bar chart of the top 20 parameter configurations by AUROC.",
        bold_prefix="top_n_configs.png: "
    )
    pdf.bullet(
        "Mean, min, and max AUROC marginalised over each parameter.",
        bold_prefix="parameter_distributions.png: "
    )
    pdf.bullet(
        "Full ROC curves comparing NEWS-2, Snapshot Fuzzy, and the best Temporal model.",
        bold_prefix="roc_comparison.png: "
    )

    # ── 4. AUROC EVALUATION SCRIPT ─────────────────────────────────
    pdf.add_page()
    pdf.section_title("4. AUROC Evaluation Script (auroc_optimal.py)")

    pdf.sub_title("4.1 What It Does Differently")
    pdf.body_text(
        "While the grid search evaluates 1,000 configurations quickly, auroc_optimal.py "
        "takes the single best configuration and runs a thorough statistical evaluation. "
        "It imports core functions from grid_search_auroc.py to avoid code duplication."
    )
    pdf.ln(2)
    pdf.body_text(
        "The optimal parameters (found by grid search) are hardcoded at the top of the file:"
    )
    pdf.code_block("ALPHA = 0.70\nBETA  = 0.5\nGAMMA = 0.80")

    pdf.sub_title("4.2 Point-Estimate AUROC")
    pdf.body_text(
        "The script computes all three predictor arrays (NEWS-2, Snapshot Fuzzy, "
        "Temporal Builder) and calls sklearn's roc_auc_score for each. This gives "
        "a single AUROC number per predictor."
    )
    pdf.ln(2)
    pdf.body_text("How roc_auc_score works internally:")
    pdf.bullet("Rank all observations by predicted score, from highest to lowest.")
    pdf.bullet(
        "Sweep a threshold from the maximum score down to the minimum. At each threshold, "
        "compute TPR (sensitivity) and FPR (1 - specificity)."
    )
    pdf.bullet("This traces out the ROC curve (TPR vs FPR).")
    pdf.bullet("Compute the area under that curve using the trapezoidal rule.")
    pdf.ln(2)
    pdf.body_text(
        "Interpretation: AUROC = 0.669 means that if you randomly pick one positive "
        "observation and one negative observation, there is a 66.9% chance the model "
        "assigns the positive a higher score. An AUROC of 0.5 = random guessing; "
        "1.0 = perfect separation."
    )
    pdf.ln(2)
    pdf.body_text(
        "Importantly, AUROC does not choose a threshold. It is purely about ranking: "
        "do positive cases tend to get higher scores than negative cases?"
    )

    pdf.add_page()
    pdf.sub_title("4.3 Patient-Level Bootstrap Confidence Intervals")
    pdf.body_text(
        "A single AUROC number does not tell you how certain you can be. The script "
        "constructs 95% confidence intervals using patient-level bootstrapping "
        "(200 resamples by default)."
    )
    pdf.ln(2)
    pdf.body_text("Why patient-level, not observation-level?")
    pdf.bullet(
        "Observations from the same patient are correlated (if one reading is abnormal, "
        "adjacent readings likely are too). Treating them as independent would "
        "underestimate the variance of the AUROC estimate."
    )
    pdf.bullet(
        "Patient-level resampling keeps all of a patient's observations together, "
        "correctly accounting for within-patient correlation."
    )
    pdf.ln(2)
    pdf.body_text("The bootstrap procedure:")
    pdf.bullet("Identify all unique patient IDs and pre-index their observation rows.")
    pdf.bullet(
        "For each of 200 iterations: sample N patients with replacement (where N = total "
        "unique patients), gather all their observations, compute AUROC on this resample."
    )
    pdf.bullet(
        "Take the 2.5th and 97.5th percentiles of the 200 AUROC values as the 95% CI."
    )

    pdf.sub_title("4.4 Sensitivity at Fixed Specificity")
    pdf.body_text(
        "This answers the clinical question: 'If I want to correctly rule out X% of "
        "healthy patients (specificity), what proportion of deteriorating patients will "
        "I catch (sensitivity)?'"
    )
    pdf.ln(2)
    pdf.body_text("The script evaluates at four specificity targets: 80%, 85%, 90%, 95%.")
    pdf.bullet("Compute the full ROC curve (arrays of FPR, TPR, and thresholds).")
    pdf.bullet("Convert FPR to specificity: specificity = 1 - FPR.")
    pdf.bullet(
        "For each target specificity, find the closest point on the curve and "
        "report the corresponding sensitivity and score threshold."
    )

    pdf.add_page()
    pdf.sub_title("4.5 Output Plots")
    pdf.body_text("The evaluation script produces four plots in auroc_results/:")
    pdf.bullet(
        "Full ROC curves for all three predictors with AUROC and 95% CI in the legend.",
        bold_prefix="roc_comparison.png: "
    )
    pdf.bullet(
        "ROC zoomed to the high-specificity region (FPR 0 to 0.3), where clinical "
        "decisions typically operate.",
        bold_prefix="roc_zoomed.png: "
    )
    pdf.bullet(
        "Precision-Recall curves, which are more informative than ROC under severe "
        "class imbalance (0.79% positive rate).",
        bold_prefix="precision_recall.png: "
    )
    pdf.bullet(
        "Histograms of score distributions for positive vs negative observations, "
        "showing how well the scores separate the two classes.",
        bold_prefix="score_distributions.png: "
    )
    pdf.ln(4)
    pdf.body_text("A summary CSV (auroc_summary.csv) is also saved with:")
    pdf.table_row(["Model", "AUROC", "CI Lower", "CI Upper", "Avg Precision"], bold=True, fill=True)
    pdf.table_row(["NEWS-2", "0.6570", "...", "...", "..."])
    pdf.table_row(["Snapshot Fuzzy", "0.6660", "...", "...", "..."])
    pdf.table_row(["Temporal Builder", "0.6693", "...", "...", "..."])

    # ── 5. KEY FORMULAS ────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("5. Key Formulas Reference")

    pdf.sub_sub_title("Trapezoid Membership Function")
    pdf.formula("mu(x; a,b,c,d) = { (x-a)/(b-a) if a<x<b; 1 if b<=x<=c; (d-x)/(d-c) if c<x<d; 0 otherwise }")

    pdf.sub_sub_title("Centroid Defuzzification")
    pdf.formula("output = integral(x * mu_agg(x)) / integral(mu_agg(x))")
    pdf.body_text(
        "Where mu_agg is the aggregated output membership function (Mamdani-style: "
        "clip each output MF at its firing strength, then take the pointwise max)."
    )

    pdf.sub_sub_title("EWMA (Exponentially Weighted Moving Average)")
    pdf.formula("ewma[t] = alpha * raw[t] + (1 - alpha) * ewma[t-1]")

    pdf.sub_sub_title("OLS Trend Slope (within 24h window)")
    pdf.formula("slope = sum((t_i - t_mean)(s_i - s_mean)) / sum((t_i - t_mean)^2)")

    pdf.sub_sub_title("Sigmoid Trend Factor")
    pdf.formula("trend_factor = 2 / (1 + exp(-beta * slope)) - 1    [slope > 0]")

    pdf.sub_sub_title("Per-Vital Adjusted Score")
    pdf.formula("adjusted = max(ewma, raw) + trend_factor * (3.0 - max(ewma, raw))")

    pdf.sub_sub_title("Total Score Aggregation")
    pdf.formula("total = (1 - gamma) * (18/3) * max_vital + gamma * sum(adjusted)")

    pdf.sub_sub_title("AUROC (Area Under ROC Curve)")
    pdf.body_text(
        "P(score(random positive) > score(random negative)). Computed by sweeping "
        "all possible thresholds and integrating TPR over FPR."
    )

    # ── 6. APPENDIX: IMAGES ────────────────────────────────────────
    images = [
        ("heatmaps.png", "Appendix: Pairwise AUROC Heatmaps"),
        ("surfaces_3d.png", "Appendix: Pairwise AUROC 3D Surfaces"),
        ("sensitivity.png", "Appendix: Parameter Sensitivity Plots"),
        ("top_n_configs.png", "Appendix: Top 20 Parameter Configurations"),
        ("parameter_distributions.png", "Appendix: Marginal Parameter Impact"),
        ("roc_comparison.png", "Appendix: ROC Curve Comparison"),
    ]
    for fname, caption in images:
        path = IMG_DIR / fname
        if path.exists():
            pdf.add_image_page(str(path), caption)

    pdf.output(str(OUTPUT))
    print(f"Saved: {OUTPUT}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_pdf()
