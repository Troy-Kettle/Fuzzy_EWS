#!/usr/bin/env python3
"""Generate a focused overview PDF for the Temporal Context tab."""

from datetime import datetime
from pathlib import Path

from fpdf import FPDF

SCRIPT_DIR = Path(__file__).parent
OUTPUT_PDF = SCRIPT_DIR / "Fuzzy_EWS_System_Overview.pdf"


class OverviewPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Fuzzy EWS - Temporal Context Tab Overview", align="R")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 60, 120)
        self.ln(2)
        self.cell(0, 10, title)
        self.ln(10)
        self.set_draw_color(30, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(50, 80, 140)
        self.cell(0, 8, title)
        self.ln(9)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.cell(6, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def formula(self, text: str):
        self.set_font("Courier", "", 9.5)
        self.set_text_color(100, 30, 30)
        self.multi_cell(0, 6, f"    {text}")
        self.ln(1)

    def code(self, text: str):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        w = self.w - self.l_margin - self.r_margin
        self.set_x(x + 4)
        self.multi_cell(w - 8, 4.5, text, fill=True)
        self.ln(3)

    def table_row(self, cells, bold=False, fill=False, col_widths=None):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        if fill:
            self.set_fill_color(230, 235, 245)
        if col_widths is None:
            col_widths = [(self.w - self.l_margin - self.r_margin) / len(cells)] * len(cells)
        for cell_text, width in zip(cells, col_widths):
            self.cell(width, 7, str(cell_text), border=1, fill=fill, align="C")
        self.ln()


def build_pdf():
    pdf = OverviewPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    w_avail = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 14, "Fuzzy Early Warning System", align="C")
    pdf.ln(16)
    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 12, "Temporal Context Tab Overview", align="C")
    pdf.ln(20)
    pdf.set_draw_color(30, 60, 120)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "Focused on Tab 3 only: membership functions,", align="C")
    pdf.ln(6)
    pdf.cell(0, 8, "snapshot scoring, and temporal adjustment", align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    pdf.add_page()
    pdf.section_title("Table of Contents")
    toc = [
        "1. What the Temporal Context tab does",
        "2. How the input membership functions are made",
        "3. How one observation becomes a snapshot score",
        "4. How the tab builds a timeline",
        "5. EWMA memory, trend detection, and adjustment",
        "6. Final total score and risk bucket",
        "7. Parameters shown in Tab 3",
    ]
    for item in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 7, item)
        pdf.ln(7)

    pdf.add_page()
    pdf.section_title("1. What the Temporal Context tab does")
    pdf.body(
        "The Temporal Context tab is the third tab in the Streamlit app. It starts "
        "with the same per-observation fuzzy scoring pipeline as the Type-1 tab, "
        "but then carries information forward through time. In short, Tab 3 asks "
        "how abnormal the newest observation is, how much recent history matters, "
        "and whether the vital appears to be worsening."
    )
    pdf.body("The full Tab 3 pipeline is:")
    pdf.code(
        "Raw observation\n"
        "    -> clamp to membership-table range\n"
        "    -> evaluate sigmoid input membership functions\n"
        "    -> collapse labels into 4 concern levels\n"
        "    -> centroid defuzzify each vital to a score in [0, 3]\n"
        "    -> aggregate 6 per-vital scores to a snapshot total in [0, 18]\n"
        "    -> store the observation in a timeline\n"
        "    -> EWMA smooth each vital across time\n"
        "    -> detect worsening trend in the raw per-vital scores\n"
        "    -> push worsening vitals upward with a sigmoid trend factor\n"
        "    -> aggregate adjusted per-vital scores into the final temporal total"
    )

    pdf.add_page()
    pdf.section_title("2. How the input membership functions are made")
    pdf.body(
        "Tab 3 uses the generated sigmoid membership tables stored in "
        "generated_membership_data/sigmoid/. Each vital loads a CSV lookup table "
        "with one Value column and one column for each linguistic label."
    )
    pdf.body(
        "That means the membership functions are pre-generated smooth sigmoid "
        "curves sampled over a grid and saved into CSV files. The app reads these "
        "lookup tables directly instead of rebuilding the curves during each run. "
        "Compared with the older hard-edged category tables in "
        "data/membership_function_plots/csv_data/, the generated sigmoid tables let "
        "neighbouring labels overlap smoothly."
    )
    pdf.body(
        "The missing construction step is that these smooth curves were not meant to "
        "be arbitrary sigmoids. In this project, they were described as being built "
        "from clinician-defined thresholds together with the clinicians' spread or "
        "variance, then converted into smooth transitions using Gaussian cumulative "
        "distribution functions (CDFs)."
    )

    pdf.sub_title("CDF-based construction idea")
    pdf.body(
        "Conceptually, each threshold between adjacent clinical categories is treated "
        "as the centre of a smooth transition rather than a hard cut-off. A Gaussian "
        "CDF is then used to turn that boundary into an S-shaped curve. The spread "
        "parameter sigma controls how gradual the transition is: smaller sigma gives "
        "a steeper boundary, while larger sigma gives a softer one."
    )
    pdf.formula("Phi(x; mu, sigma) = normal CDF centred at mu with spread sigma")
    pdf.body(
        "For edge categories, a single CDF can define the membership shape. For "
        "example, a low-extreme category can be represented by a decreasing CDF, "
        "while a high-extreme category can be represented by an increasing CDF."
    )
    pdf.code(
        "low-side category    ~ 1 - Phi(x; boundary, sigma)\n"
        "high-side category   ~ Phi(x; boundary, sigma)"
    )
    pdf.body(
        "For middle categories, two neighbouring CDF transitions are combined so the "
        "membership rises from one boundary and falls at the next, producing the "
        "smooth hill-shaped regions seen in the generated tables."
    )
    pdf.code(
        "middle category ~ Phi(x; left_boundary, sigma_left)\n"
        "                 - Phi(x; right_boundary, sigma_right)"
    )
    pdf.body(
        "The practical outcome is that the old sharp threshold system becomes a set "
        "of overlapping smooth curves whose transitions are controlled by clinician "
        "uncertainty rather than by arbitrary straight-line corners."
    )
    pdf.body(
        "A useful property of this construction is the partition-of-unity behaviour "
        "described elsewhere in the project: across the input range, the category "
        "memberships are designed to sum to approximately 1 at each x-value. That is "
        "why a value can partially belong to two neighbouring labels without the "
        "total membership becoming unstable."
    )
    pdf.body(
        "So, in teaching terms: the generated CSVs in Tab 3 are the final sampled "
        "result of a CDF-based smoothing process. The app itself only loads and "
        "interpolates those saved curves, but the reason they look sigmoid is that "
        "they were originally constructed from Gaussian CDF transitions."
    )

    pdf.sub_title("Membership families")
    cw = [w_avail * 0.22, w_avail * 0.18, w_avail * 0.60]
    pdf.table_row(["Vital", "Family", "Labels"], bold=True, fill=True, col_widths=cw)
    pdf.table_row(
        ["Heart rate", "7-label", "Below severe, below moderate, below mild, no concern, above mild, above moderate, above severe"],
        col_widths=cw,
    )
    pdf.table_row(
        ["Systolic BP", "7-label", "Below severe, below moderate, below mild, no concern, above mild, above moderate, above severe"],
        col_widths=cw,
    )
    pdf.table_row(
        ["Temperature", "7-label", "Below severe, below moderate, below mild, no concern, above mild, above moderate, above severe"],
        col_widths=cw,
    )
    pdf.table_row(
        ["Respiratory rate", "7-label", "Below severe, below moderate, below mild, no concern, above mild, above moderate, above severe"],
        col_widths=cw,
    )
    pdf.table_row(
        ["SpO2", "Downward", "Below severe, below moderate, below mild, no concern"],
        col_widths=cw,
    )
    pdf.table_row(
        ["FiO2", "Upward", "No concern, above mild, above moderate, above severe"],
        col_widths=cw,
    )

    pdf.ln(4)
    pdf.body(
        "Why the families differ: heart rate, blood pressure, temperature, and "
        "respiratory rate can be dangerous when too low or too high, so they use "
        "a symmetric 7-label design. SpO2 is mainly dangerous when it falls, so its "
        "labels only move downward. FiO2 is mainly concerning when it rises, because "
        "a higher oxygen requirement suggests greater support."
    )
    pdf.body(
        "In the CDF view, that means the 7-label vitals use multiple adjacent "
        "Gaussian-CDF transitions on both sides of a central normal region, whereas "
        "SpO2 and FiO2 only need one-sided transition stacks because concern mainly "
        "develops in one direction."
    )

    pdf.sub_title("Lookup and interpolation")
    pdf.body(
        "When the user enters a value, the app finds the two nearest rows in the "
        "CSV and linearly interpolates between them. This turns a discrete lookup "
        "table into a continuous membership function."
    )
    pdf.formula("t = (v - k_i) / (k_(i+1) - k_i)")
    pdf.formula("mu(v) = mu(k_i) * (1 - t) + mu(k_(i+1)) * t")
    pdf.body("Boundary handling in Tab 3:")
    pdf.bullet("Before scoring, every observation is clamped to the minimum and maximum Value in the relevant CSV.")
    pdf.bullet("If the user enters a value outside the available grid, the model uses the nearest edge of the generated table.")
    pdf.bullet("Because neighbouring sigmoid labels overlap, one observation can partially belong to multiple labels at once.")

    pdf.add_page()
    pdf.section_title("3. How one observation becomes a snapshot score")
    pdf.body(
        "Tab 3 first computes an ordinary snapshot fuzzy score for the newest "
        "observation. Temporal logic is added only after that. The snapshot stage "
        "has three core parts: label firing, concern-level consolidation, and "
        "centroid defuzzification."
    )

    pdf.sub_title("Stage A: evaluate the input labels")
    pdf.body(
        "Each vital is evaluated against its sigmoid input membership functions. "
        "The result is a set of firing strengths between 0 and 1."
    )

    pdf.sub_title("Stage B: collapse many labels into 4 concern levels")
    pdf.body(
        "The app maps all input labels into four output concern levels: No concern, "
        "Mild concern, Moderate concern, and Severe concern. The mapping uses "
        "max-aggregation, so both low-severe and high-severe labels feed into the "
        "same Severe concern output."
    )
    pdf.code(
        "for each input label with strength mu:\n"
        "    if 'severe' in label:   Severe   = max(Severe, mu)\n"
        "    if 'moderate' in label: Moderate = max(Moderate, mu)\n"
        "    if 'mild' in label:     Mild     = max(Mild, mu)\n"
        "    if 'no concern' in label: No     = max(No, mu)"
    )

    pdf.sub_title("Stage C: defuzzify to a per-vital score in [0, 3]")
    pdf.body(
        "The output universe for each vital is 0 to 3. Four trapezoidal output "
        "membership functions represent No, Mild, Moderate, and Severe concern."
    )
    cw2 = [w_avail * 0.34, w_avail * 0.165, w_avail * 0.165, w_avail * 0.165, w_avail * 0.165]
    pdf.table_row(["Concern", "a", "b", "c", "d"], bold=True, fill=True, col_widths=cw2)
    pdf.table_row(["No concern", "-0.5", "0", "0", "0.75"], col_widths=cw2)
    pdf.table_row(["Mild concern", "0.25", "1", "1", "1.75"], col_widths=cw2)
    pdf.table_row(["Moderate concern", "1.25", "2", "2", "2.75"], col_widths=cw2)
    pdf.table_row(["Severe concern", "2.25", "3", "3", "3.5"], col_widths=cw2)
    pdf.ln(3)
    pdf.body(
        "The app evaluates these outputs on 301 points from 0.00 to 3.00, clips "
        "each set by its firing strength, takes the pointwise maximum, and then "
        "computes the centroid."
    )
    pdf.formula("score = SUM(x_i * mu_agg(x_i)) / SUM(mu_agg(x_i))")
    pdf.body("Two implementation details matter:")
    pdf.bullet("Firings below 0.05 are zeroed so tiny edge overlaps do not create noise.")
    pdf.bullet("If only the No concern set remains active, the app returns an exact score of 0.0.")
    pdf.body(
        "After this has been done for all six vitals, the per-vital scores are "
        "aggregated into the snapshot total on the 0 to 18 scale."
    )

    pdf.add_page()
    pdf.section_title("4. How the tab builds a timeline")
    pdf.body(
        "Each time the user clicks Add observation, Tab 3 stores the raw values, "
        "the clamped model values, the six per-vital fuzzy scores, and the snapshot "
        "totals. The user also chooses an interval size and unit, so each new row "
        "gets a time in minutes from the start of the sequence."
    )
    pdf.body("Every stored row therefore has both inputs and model outputs:")
    pdf.bullet("Clinical inputs: heart rate, blood pressure, temperature, respiratory rate, SpO2, and FiO2.")
    pdf.bullet("Model outputs: six per-vital fuzzy scores, snapshot fuzzy total, and NEWS-2 snapshot total.")
    pdf.body(
        "That sequence becomes the input to the temporal adjustment stage. The newest "
        "observation is always interpreted against the whole timeline currently built "
        "inside the tab."
    )

    pdf.add_page()
    pdf.section_title("5. EWMA memory, trend detection, and adjustment")
    pdf.sub_title("Step 1: EWMA memory")
    pdf.body(
        "For each vital independently, Tab 3 applies an exponentially weighted moving "
        "average to the raw per-vital fuzzy scores. This is the memory part of the "
        "temporal builder."
    )
    pdf.formula("EWMA[0] = raw[0]")
    pdf.formula("EWMA[t] = alpha * raw[t] + (1 - alpha) * EWMA[t-1]")
    pdf.body(
        "The app then clamps the smoothed value upward so temporal context cannot "
        "hide a suddenly bad current observation."
    )
    pdf.formula("clamped[t] = max(EWMA[t], raw[t])")

    pdf.sub_title("Step 2: estimate worsening or improvement")
    pdf.body(
        "The trend is computed from the raw per-vital scores, not from the EWMA "
        "scores. For the latest observation, the app looks back over the selected "
        "window, converts times to hours since the first point in that window, and "
        "fits a simple ordinary least squares slope."
    )
    pdf.formula("slope = SUM((t_i - t_mean)(s_i - s_mean)) / SUM((t_i - t_mean)^2)")
    pdf.body("Interpretation:")
    pdf.bullet("Positive slope means the vital is worsening.")
    pdf.bullet("Negative slope means the vital is improving.")
    pdf.bullet("If fewer than two observations are in the window, slope is set to 0.")

    pdf.sub_title("Step 3: convert worsening into a bounded trend factor")
    pdf.body(
        "Only positive slopes trigger an upward adjustment. Tab 3 uses a sigmoid "
        "transform so the effect is smooth and bounded."
    )
    pdf.code(
        "if slope > 0:\n"
        "    trend_factor = 2 / (1 + exp(-beta * slope)) - 1\n"
        "else:\n"
        "    trend_factor = 0"
    )
    pdf.body(
        "Stable or improving vitals are therefore not penalised. When the slope is "
        "positive, the factor lies between 0 and 1. Higher beta makes the response "
        "steeper, so smaller positive slopes lead to larger upward pushes."
    )

    pdf.sub_title("Step 4: push the score toward the maximum")
    pdf.formula("adjusted[t] = clamped[t] + trend_factor * (3.0 - clamped[t])")
    pdf.formula("adjusted[t] = clip(adjusted[t], 0.0, 3.0)")
    pdf.body(
        "The term (3.0 - clamped[t]) is the remaining headroom to the maximum. "
        "This means a vital already near 3 can only move a little further, while "
        "a mid-range vital has more room to be pushed upward."
    )

    pdf.add_page()
    pdf.section_title("6. Final total score and risk bucket")
    pdf.body(
        "After all six vitals have an EWMA score and an adjusted score, Tab 3 forms "
        "three headline totals: the latest snapshot total, the EWMA total, and the "
        "trend-adjusted total."
    )
    pdf.body(
        "The totals use the same gamma rule for both the EWMA stage and the final "
        "trend-adjusted stage."
    )
    pdf.formula("additive_total = SUM(per_vital_scores)")
    pdf.formula("max_based_total = (18 / 3) * max(per_vital_scores)")
    pdf.formula("total = (1 - gamma) * max_based_total + gamma * additive_total")
    pdf.formula("total = max(total, snapshot_total)")
    pdf.body(
        "The final max() guarantees that EWMA smoothing (alpha) cannot reduce the "
        "overall concern score below the latest snapshot total."
    )
    pdf.body("How to read gamma:")
    pdf.bullet("gamma = 1.0 keeps the total approximately additive.")
    pdf.bullet("gamma = 0.0 lets the single worst vital dominate the overall total.")
    pdf.bullet("Intermediate gamma values blend both ideas.")
    pdf.body(
        "The final displayed total is clipped to the 0 to 18 range and mapped to "
        "the same risk buckets used elsewhere in the app."
    )
    cw3 = [w_avail * 0.35, w_avail * 0.3, w_avail * 0.35]
    pdf.table_row(["Score range", "Bucket", "Meaning"], bold=True, fill=True, col_widths=cw3)
    pdf.table_row(["0 to < 4", "Low", "Limited current or historical concern"], col_widths=cw3)
    pdf.table_row(["4 to < 8", "Moderate", "Clear abnormality or mild accumulation"], col_widths=cw3)
    pdf.table_row(["8 to < 12", "High", "Substantial concern across one or more vitals"], col_widths=cw3)
    pdf.table_row(["12 to 18", "Critical", "Severe or strongly worsening concern"], col_widths=cw3)

    pdf.add_page()
    pdf.section_title("7. Parameters shown in Tab 3")
    cw4 = [w_avail * 0.22, w_avail * 0.18, w_avail * 0.18, w_avail * 0.42]
    pdf.table_row(["Parameter", "UI range", "Default", "Role"], bold=True, fill=True, col_widths=cw4)
    pdf.table_row(
        ["alpha", "0.05 to 1.0", "0.7", "EWMA smoothing strength; higher alpha gives more weight to the latest value"],
        col_widths=cw4,
    )
    pdf.table_row(
        ["beta", "0.1 to 10.0", "2.0", "Controls how sharply a positive slope activates the sigmoid trend factor"],
        col_widths=cw4,
    )
    pdf.table_row(
        ["Look-back window", "1 to 72 h", "24 h", "Time window used when fitting the raw-score trend slope"],
        col_widths=cw4,
    )
    pdf.table_row(
        ["gamma", "0.0 to 1.0", "1.0", "Blends additive total with worst-vital dominance"],
        col_widths=cw4,
    )
    pdf.ln(4)
    pdf.body(
        "The key learning idea is that Tab 3 is layered. The membership functions "
        "convert vitals into fuzzy scores. The timeline stores those scores. EWMA "
        "gives memory. The raw-score slope detects whether the patient is getting "
        "worse. The sigmoid trend factor keeps that adjustment smooth and bounded. "
        "Gamma then decides whether the total behaves more like a sum of all vitals "
        "or more like a worst-vital alarm."
    )

    pdf.output(str(OUTPUT_PDF))
    return OUTPUT_PDF, pdf.page_no()


if __name__ == "__main__":
    out, pages = build_pdf()
    print(f"Saved: {out}")
    print(f"Pages: {pages}")
