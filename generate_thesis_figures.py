"""
Generate all thesis figures for Chapter 5: Fuzzy System Development.
Outputs publication-quality figures (300 DPI, vector-compatible) to thesis_figures/.

Usage:
    python generate_thesis_figures.py
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style: thesis-quality, LaTeX-compatible
# ---------------------------------------------------------------------------
STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
}
plt.rcParams.update(STYLE)

OUT_DIR = Path(__file__).parent / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)

SIGMOID_DIR = Path(__file__).parent / "generated_membership_data" / "sigmoid"

# Consistent colour palette (colourblind-safe, Tol's vibrant)
COLOURS = {
    "severe_below": "#CC3311",
    "moderate_below": "#EE7733",
    "mild_below": "#E8C547",
    "no_concern": "#009988",
    "mild_above": "#E8C547",
    "moderate_above": "#EE7733",
    "severe_above": "#CC3311",
}

CONCERN_COLOURS = {
    "No concern": "#009988",
    "Mild concern": "#E8C547",
    "Moderate concern": "#EE7733",
    "Severe concern": "#CC3311",
}

SEVEN_COLOURS = [
    "#CC3311", "#EE7733", "#E8C547", "#009988",
    "#E8C547", "#EE7733", "#CC3311",
]

FOUR_COLOURS_DOWN = ["#CC3311", "#EE7733", "#E8C547", "#009988"]
FOUR_COLOURS_UP = ["#009988", "#E8C547", "#EE7733", "#CC3311"]


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.pdf", format="pdf")
    fig.savefig(OUT_DIR / f"{name}.png", format="png")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ========================================================================
# FIGURE 1: High-Level System Architecture
# ========================================================================
def fig_01_system_architecture():
    print("Generating Figure 1: System Architecture...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.4", linewidth=1.5)

    def draw_box(x, y, w, h, text, colour, fontsize=10, bold=False):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3",
            facecolor=colour, edgecolor="#333333",
            linewidth=1.5, alpha=0.85,
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color="#1a1a1a",
        )

    def arrow(x1, y1, x2, y2, text="", colour="#555555"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=colour,
                linewidth=1.8, mutation_scale=15,
            ),
        )
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my + 0.12, text, fontsize=8.5, color="#444444",
                    fontstyle="italic")

    # Input vector
    draw_box(0.2, 2.1, 1.6, 1.8, "$V_t$\n(6 vitals)", "#d4e6f1", fontsize=11, bold=True)

    # Snapshot Layer (big box)
    snap_box = FancyBboxPatch(
        (2.5, 0.5), 3.2, 5.0,
        boxstyle="round,pad=0.25",
        facecolor="#f0f7ee", edgecolor="#5B8C5A",
        linewidth=2.0, alpha=0.4, linestyle="--",
    )
    ax.add_patch(snap_box)
    ax.text(4.1, 5.25, "Snapshot Layer $\\mathcal{F}$",
            ha="center", fontsize=11.5, fontweight="bold", color="#3d6b3a")

    draw_box(2.8, 3.8, 2.6, 0.85, "Fuzzification\n(Membership Functions)", "#c8e6c9", fontsize=9)
    draw_box(2.8, 2.55, 2.6, 0.85, "Concern-Level\nConsolidation", "#c8e6c9", fontsize=9)
    draw_box(2.8, 1.3, 2.6, 0.85, "Centroid\nDefuzzification", "#c8e6c9", fontsize=9)

    arrow(4.1, 3.8, 4.1, 3.45)
    arrow(4.1, 2.55, 4.1, 2.2)

    # Temporal Layer
    temp_box = FancyBboxPatch(
        (6.4, 0.5), 3.2, 5.0,
        boxstyle="round,pad=0.25",
        facecolor="#fef4e8", edgecolor="#D4890B",
        linewidth=2.0, alpha=0.4, linestyle="--",
    )
    ax.add_patch(temp_box)
    ax.text(8.0, 5.25, "Temporal Layer $\\mathcal{T}$",
            ha="center", fontsize=11.5, fontweight="bold", color="#a36b08")

    draw_box(6.7, 3.8, 2.6, 0.85, "EWMA\nMemory Retention", "#fde8c8", fontsize=9)
    draw_box(6.7, 2.55, 2.6, 0.85, "Trend Extraction\n& Sigmoid Penalty", "#fde8c8", fontsize=9)
    draw_box(6.7, 1.3, 2.6, 0.85, "$\\gamma$ Cross-Vital\nAggregation", "#fde8c8", fontsize=9)

    arrow(8.0, 3.8, 8.0, 3.45)
    arrow(8.0, 2.55, 8.0, 2.2)

    # Connections
    arrow(1.8, 3.0, 2.8, 4.2)
    arrow(4.1, 1.3, 4.1, 0.75)
    ax.text(4.1, 0.5, "$S_{snapshot,t}$", ha="center", fontsize=11, fontweight="bold",
            color="#3d6b3a")
    arrow(5.7, 4.2, 6.7, 4.2, "$S_{snapshot,1:t}$")
    arrow(8.0, 1.3, 8.0, 0.75)
    ax.text(8.0, 0.5, "$S_{final,t}$", ha="center", fontsize=11, fontweight="bold",
            color="#a36b08")

    save(fig, "fig_01_system_architecture")


# ========================================================================
# FIGURE 2: NEWS-2 Step Function vs FEWS Smooth Scoring (Heart Rate)
# ========================================================================
def fig_02_news2_vs_fews():
    print("Generating Figure 2: NEWS-2 vs FEWS Comparison...")
    hr_df = pd.read_csv(SIGMOID_DIR / "heart_rate_membership_functions.csv")
    x = hr_df["Value"].values
    cols = [c for c in hr_df.columns if c != "Value"]
    weights = [3, 2, 1, 0, 1, 2, 3]

    numer = np.zeros_like(x, dtype=float)
    denom = np.zeros_like(x, dtype=float)
    for i, col in enumerate(cols):
        m = hr_df[col].values
        numer += m * weights[i]
        denom += m
    denom = np.where(denom == 0, 1, denom)
    fews_score = numer / denom

    def news2_hr(hr):
        if hr <= 40:
            return 3
        elif hr <= 50:
            return 1
        elif hr <= 90:
            return 0
        elif hr <= 110:
            return 1
        elif hr <= 130:
            return 2
        else:
            return 3
    news2_score = np.array([news2_hr(h) for h in x])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1]})

    ax = axes[0]
    ax.step(x, news2_score, where="post", color="#CC3311", linewidth=2.0, label="NEWS-2")
    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("Score")
    ax.set_title("(a) NEWS-2 Step Function", fontweight="bold")
    ax.set_ylim(-0.2, 3.5)
    ax.set_xlim(30, 180)

    for boundary in [40, 51, 91, 111, 131]:
        ax.axvline(boundary, color="#aaaaaa", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.annotate(
        "90→91 bpm\n+1 score",
        xy=(91, 0.1), xytext=(105, 0.8),
        arrowprops=dict(arrowstyle="->", color="#CC3311", lw=1.3),
        fontsize=8.5, color="#CC3311", fontweight="bold",
    )
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(x, fews_score, color="#0077BB", linewidth=2.0, label="FEWS")
    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("Score")
    ax.set_title("(b) FEWS Continuous Score", fontweight="bold")
    ax.set_ylim(-0.2, 3.5)
    ax.set_xlim(30, 180)
    ax.fill_between(x, fews_score, alpha=0.12, color="#0077BB")
    ax.annotate(
        "90→91 bpm\nsmooth transition",
        xy=(91, float(np.interp(91, x, fews_score))),
        xytext=(115, 0.8),
        arrowprops=dict(arrowstyle="->", color="#0077BB", lw=1.3),
        fontsize=8.5, color="#0077BB", fontweight="bold",
    )
    ax.legend(loc="upper left")

    fig.suptitle("Quantisation Error: Discrete vs Continuous Scoring for Heart Rate",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_02_news2_vs_fews_comparison")


# ========================================================================
# FIGURE 3: Individual Gaussian CDF Transition
# ========================================================================
def fig_03_gaussian_cdf_transition():
    print("Generating Figure 3: Gaussian CDF Transition...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    mu, sigma = 91, 5
    x = np.linspace(70, 115, 500)
    from scipy.stats import norm
    cdf = norm.cdf(x, mu, sigma)

    ax = axes[0]
    ax.plot(x, cdf, color="#0077BB", linewidth=2.2)
    ax.axvline(mu, color="#888888", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Vital Sign Value")
    ax.set_ylabel("$\\Phi(x; \\mu, \\sigma)$")
    ax.set_title("(a) Gaussian CDF", fontweight="bold")
    ax.annotate(f"$\\mu = {mu}$", xy=(mu, 0.5), xytext=(mu + 6, 0.35),
                arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9.5)
    ax.annotate(f"$\\sigma = {sigma}$\n(spread)", xy=(mu - sigma, norm.cdf(mu - sigma, mu, sigma)),
                xytext=(mu - 17, 0.6),
                arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9)

    ax = axes[1]
    lower = 1 - cdf
    upper = cdf
    ax.plot(x, lower, color="#CC3311", linewidth=2.2, label="$\\mu_{L_{low}} = 1 - \\Phi$")
    ax.plot(x, upper, color="#0077BB", linewidth=2.2, label="$\\mu_{L_{high}} = \\Phi$")
    ax.axvline(mu, color="#888888", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Vital Sign Value")
    ax.set_ylabel("Membership Degree")
    ax.set_title("(b) Boundary Categories", fontweight="bold")
    ax.legend(loc="center right", fontsize=9)

    ax = axes[2]
    mu_left, sigma_left = 80, 4
    mu_right, sigma_right = 100, 5
    cdf_left = norm.cdf(x, mu_left, sigma_left)
    cdf_right = norm.cdf(x, mu_right, sigma_right)
    mid_mf = cdf_left - cdf_right
    mid_mf = np.clip(mid_mf, 0, None)
    ax.plot(x, mid_mf, color="#EE7733", linewidth=2.2,
            label="$\\mu_{L_{mid}} = \\Phi_{left} - \\Phi_{right}$")
    ax.fill_between(x, mid_mf, alpha=0.15, color="#EE7733")
    ax.axvline(mu_left, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(mu_right, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(mu_left - 0.5, -0.08, f"$\\mu_{{left}}$", ha="center", fontsize=9, color="#666")
    ax.text(mu_right + 0.5, -0.08, f"$\\mu_{{right}}$", ha="center", fontsize=9, color="#666")
    ax.set_xlabel("Vital Sign Value")
    ax.set_ylabel("Membership Degree")
    ax.set_title("(c) Interior Category", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Construction of Membership Functions from Gaussian CDF",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_03_gaussian_cdf_construction")


# ========================================================================
# FIGURE 4: Bidirectional Membership Functions (2x2 grid)
# ========================================================================
def fig_04_bidirectional_mfs():
    print("Generating Figure 4: Bidirectional Membership Functions...")
    vitals = {
        "Heart Rate (bpm)": "heart_rate_membership_functions.csv",
        "Systolic Blood Pressure (mmHg)": "systolic_blood_pressure_membership_functions.csv",
        "Temperature (°C)": "temperature_membership_functions.csv",
        "Respiratory Rate (breaths/min)": "respiratory_rate_membership_functions.csv",
    }
    short_labels = [
        "Below Severe", "Below Moderate", "Below Mild", "No Concern",
        "Above Mild", "Above Moderate", "Above Severe",
    ]
    linestyles = ["-", "--", "-.", "-", "-.", "--", "-"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for idx, (title, fname) in enumerate(vitals.items()):
        df = pd.read_csv(SIGMOID_DIR / fname)
        x = df["Value"].values
        cols = [c for c in df.columns if c != "Value"]
        ax = axes[idx]

        for i, col in enumerate(cols):
            ax.plot(x, df[col].values, color=SEVEN_COLOURS[i],
                    linewidth=1.8, label=short_labels[i],
                    linestyle=linestyles[i])

        ax.set_xlabel(title.split("(")[0].strip())
        ax.set_ylabel("Membership Degree ($\\mu$)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        if idx == 0:
            ax.legend(loc="upper center", ncol=4, fontsize=7.5,
                      bbox_to_anchor=(0.5, 1.35), framealpha=0.9)

    fig.suptitle("Bidirectional Membership Functions (7 Linguistic Terms)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig_04_bidirectional_membership_functions")


# ========================================================================
# FIGURE 5: Unidirectional Membership Functions (SpO2 + FiO2)
# ========================================================================
def fig_05_unidirectional_mfs():
    print("Generating Figure 5: Unidirectional Membership Functions...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # SpO2
    df = pd.read_csv(SIGMOID_DIR / "oxygen_saturation_membership_functions.csv")
    cols = [c for c in df.columns if c != "Value"]
    ax = axes[0]
    for i, col in enumerate(cols):
        ax.plot(df["Value"], df[col], color=FOUR_COLOURS_DOWN[i],
                linewidth=2.0, label=col.replace("Below normal - ", "Below ").replace(" concern", ""))
    ax.set_xlabel("SpO₂ (%)")
    ax.set_ylabel("Membership Degree ($\\mu$)")
    ax.set_title("(a) Oxygen Saturation (Unidirectional ↓)", fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)

    # FiO2
    df = pd.read_csv(SIGMOID_DIR / "inspired_oxygen_concentration_membership_functions.csv")
    cols = [c for c in df.columns if c != "Value"]
    ax = axes[1]
    for i, col in enumerate(cols):
        ax.plot(df["Value"], df[col], color=FOUR_COLOURS_UP[i],
                linewidth=2.0, label=col.replace("Above normal - ", "Above ").replace(" concern", ""))
    ax.set_xlabel("FiO₂ (%)")
    ax.set_ylabel("Membership Degree ($\\mu$)")
    ax.set_title("(b) Inspired Oxygen (Unidirectional ↑)", fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)

    fig.suptitle("Unidirectional Membership Functions (4 Linguistic Terms)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_05_unidirectional_membership_functions")


# ========================================================================
# FIGURE 6: Output Trapezoidal Membership Functions
# ========================================================================
def fig_06_output_trapezoids():
    print("Generating Figure 6: Output Trapezoids...")
    defs = {
        "No Concern ($C_0$, $w=0$)": (-0.5, 0, 0, 0.75),
        "Mild Concern ($C_1$, $w=1$)": (0.25, 1, 1, 1.75),
        "Moderate Concern ($C_2$, $w=2$)": (1.25, 2, 2, 2.75),
        "Severe Concern ($C_3$, $w=3$)": (2.25, 3, 3, 3.5),
    }
    colours = list(CONCERN_COLOURS.values())

    x = np.linspace(-0.5, 3.5, 1000)

    def trap(xv, a, b, c, d):
        if b <= xv <= c:
            return 1.0
        if xv <= a or xv >= d:
            return 0.0
        if a < xv < b:
            return (xv - a) / (b - a)
        if c < xv < d:
            return (d - xv) / (d - c)
        return 0.0

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for idx, (label, params) in enumerate(defs.items()):
        y = [trap(xi, *params) for xi in x]
        ax.plot(x, y, color=colours[idx], linewidth=2.2, label=label)
        ax.fill_between(x, y, alpha=0.12, color=colours[idx])

    ax.set_xlabel("Concern Score (per vital)")
    ax.set_ylabel("Membership Degree")
    ax.set_title("Output Membership Functions (Trapezoidal)", fontweight="bold")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.legend(loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.18),
              framealpha=0.9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, "fig_06_output_membership_functions")


# ========================================================================
# FIGURE 7: Centroid Defuzzification Worked Example
# ========================================================================
def fig_07_defuzzification_example():
    print("Generating Figure 7: Centroid Defuzzification Example...")
    defs = {
        "No Concern": (-0.5, 0, 0, 0.75),
        "Mild Concern": (0.25, 1, 1, 1.75),
        "Moderate Concern": (1.25, 2, 2, 2.75),
        "Severe Concern": (2.25, 3, 3, 3.5),
    }

    def trap(xv, a, b, c, d):
        if b <= xv <= c:
            return 1.0
        if xv <= a or xv >= d:
            return 0.0
        if a < xv < b:
            return (xv - a) / (b - a)
        if c < xv < d:
            return (d - xv) / (d - c)
        return 0.0

    # Example: HR=105 → fuzzy activations approximate to
    # No Concern ~0.15, Mild ~0.70, Moderate ~0.15, Severe ~0.0
    firings = {
        "No Concern": 0.15,
        "Mild Concern": 0.70,
        "Moderate Concern": 0.15,
        "Severe Concern": 0.0,
    }
    colours = list(CONCERN_COLOURS.values())

    x = np.linspace(-0.5, 3.5, 401)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    # (a) Show the firing levels
    ax = axes[0]
    labels = list(firings.keys())
    vals = list(firings.values())
    bars = ax.barh(labels, vals, color=colours, edgecolor="#333", linewidth=0.8, height=0.6)
    ax.set_xlabel("Activation Level ($\\mu_{C_i}$)")
    ax.set_title("(a) Concern-Level Activations", fontweight="bold")
    ax.set_xlim(0, 1.05)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=9.5)

    # (b) Clipped output sets and aggregated envelope
    ax = axes[1]
    aggregated = np.zeros_like(x)
    for idx, (label, params) in enumerate(defs.items()):
        firing = firings[label]
        if firing <= 0:
            continue
        y_full = np.array([trap(xi, *params) for xi in x])
        y_clipped = np.minimum(y_full, firing)
        ax.plot(x, y_full, color=colours[idx], linewidth=0.8, alpha=0.35, linestyle="--")
        ax.fill_between(x, y_clipped, alpha=0.2, color=colours[idx], label=f"{label} (clipped)")
        aggregated = np.maximum(aggregated, y_clipped)

    ax.plot(x, aggregated, color="#333333", linewidth=2.0, label="Aggregated")
    ax.fill_between(x, aggregated, alpha=0.08, color="#333333")
    ax.set_xlabel("Concern Score")
    ax.set_ylabel("Membership")
    ax.set_title("(b) Clipped & Aggregated Output", fontweight="bold")
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.05, 1.1)

    # (c) Centroid calculation
    ax = axes[2]
    ax.fill_between(x, aggregated, alpha=0.25, color="#0077BB")
    ax.plot(x, aggregated, color="#0077BB", linewidth=2.0)
    numerator = np.sum(x * aggregated)
    denominator = np.sum(aggregated)
    centroid = numerator / denominator if denominator > 0 else 0
    ax.axvline(centroid, color="#CC3311", linewidth=2.5, linestyle="-",
               label=f"Centroid = {centroid:.2f}")
    ax.scatter([centroid], [0], color="#CC3311", s=100, zorder=5, marker="^")
    ax.set_xlabel("Concern Score")
    ax.set_ylabel("Aggregated Membership")
    ax.set_title("(c) Centroid Defuzzification", fontweight="bold")
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper right", fontsize=9.5)

    fig.suptitle(
        "Centroid Defuzzification: Worked Example (Heart Rate ≈ 105 bpm)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig_07_centroid_defuzzification_example")


# ========================================================================
# FIGURE 8: Partition of Unity
# ========================================================================
def fig_08_partition_of_unity():
    print("Generating Figure 8: Partition of Unity...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]})

    df = pd.read_csv(SIGMOID_DIR / "heart_rate_membership_functions.csv")
    x = df["Value"].values
    cols = [c for c in df.columns if c != "Value"]
    short_labels = [
        "Below Severe", "Below Moderate", "Below Mild", "No Concern",
        "Above Mild", "Above Moderate", "Above Severe",
    ]
    linestyles = ["-", "--", "-.", "-", "-.", "--", "-"]

    ax = axes[0]
    total = np.zeros_like(x, dtype=float)
    for i, col in enumerate(cols):
        vals = df[col].values
        ax.plot(x, vals, color=SEVEN_COLOURS[i], linewidth=1.5,
                label=short_labels[i], linestyle=linestyles[i])
        total += vals

    ax.set_ylabel("Membership Degree ($\\mu$)")
    ax.set_title("Heart Rate Membership Functions", fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.28),
              framealpha=0.9)

    ax = axes[1]
    ax.plot(x, total, color="#0077BB", linewidth=2.0)
    ax.axhline(1.0, color="#CC3311", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Unity (1.0)")
    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("$\\sum \\mu$")
    ax.set_title("Sum of All Memberships (Partition of Unity)", fontweight="bold")
    ax.set_ylim(0.9, 1.1)
    ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, "fig_08_partition_of_unity")


# ========================================================================
# FIGURE 9: EWMA with Different Alpha Values
# ========================================================================
def fig_09_ewma_alpha():
    print("Generating Figure 9: EWMA Behaviour...")
    np.random.seed(42)
    n_obs = 20
    raw = np.concatenate([
        np.array([0.2, 0.3, 0.2, 0.4, 0.3, 0.5, 0.4, 0.6]),
        np.array([1.0, 1.4, 1.8, 2.0, 2.2, 2.5, 2.3, 2.6]),
        np.array([2.4, 2.1, 1.8, 1.5]),
    ])
    t = np.arange(len(raw))

    alphas = [0.3, 0.5, 0.7, 0.9]
    alpha_colours = ["#CC3311", "#EE7733", "#0077BB", "#009988"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, raw, "o--", color="#888888", markersize=5, linewidth=1.2,
            label="Raw Observations", alpha=0.7)

    for alpha, colour in zip(alphas, alpha_colours):
        ewma = [raw[0]]
        for v in raw[1:]:
            ewma.append(alpha * v + (1 - alpha) * ewma[-1])
        ax.plot(t, ewma, color=colour, linewidth=2.0,
                label=f"$\\alpha = {alpha}$ ({int((1-alpha)*100)}% memory)")

    ax.set_xlabel("Observation Index ($t$)")
    ax.set_ylabel("Score (0–3)")
    ax.set_title("EWMA Memory Retention: Effect of $\\alpha$ on Score Smoothing",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(-0.1, 3.1)

    ax.annotate("Deterioration phase", xy=(11, 2.0), fontsize=9,
                fontstyle="italic", color="#555555")
    ax.axvspan(7.5, 15.5, alpha=0.06, color="#CC3311")
    ax.annotate("Improvement", xy=(16.5, 1.5), fontsize=9,
                fontstyle="italic", color="#555555")
    ax.axvspan(15.5, 19.5, alpha=0.06, color="#009988")

    fig.tight_layout()
    save(fig, "fig_09_ewma_alpha_comparison")


# ========================================================================
# FIGURE 10: Modified Sigmoid Trend Factor
# ========================================================================
def fig_10_trend_sigmoid():
    print("Generating Figure 10: Modified Sigmoid Trend Factor...")
    slopes = np.linspace(-2, 4, 500)
    betas = [0.5, 1.0, 2.0, 5.0]
    beta_colours = ["#009988", "#0077BB", "#EE7733", "#CC3311"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Full sigmoid (both sides)
    ax = axes[0]
    for beta, colour in zip(betas, beta_colours):
        T = 2.0 / (1.0 + np.exp(-beta * slopes)) - 1.0
        ax.plot(slopes, T, color=colour, linewidth=2.0, label=f"$\\beta = {beta}$")

    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.set_xlabel("Slope ($m_j$)")
    ax.set_ylabel("Trend Factor $T_j(m_j)$")
    ax.set_title("(a) Full Modified Sigmoid", fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.set_ylim(-1.2, 1.2)

    # (b) With unidirectional clamping (what the system actually uses)
    ax = axes[1]
    for beta, colour in zip(betas, beta_colours):
        T = 2.0 / (1.0 + np.exp(-beta * slopes)) - 1.0
        T_clamped = np.where(slopes > 0, T, 0.0)
        ax.plot(slopes, T_clamped, color=colour, linewidth=2.0, label=f"$\\beta = {beta}$")

    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.fill_between(slopes, -0.1, 0, where=(slopes <= 0),
                     alpha=0.08, color="#009988", label="Clamped to 0")
    ax.set_xlabel("Slope ($m_j$)")
    ax.set_ylabel("Applied Trend Factor $T_j^*$")
    ax.set_title("(b) With Unidirectional Safety Constraint", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(-0.15, 1.15)

    fig.suptitle("Trend Penalty Function: Sensitivity Controlled by $\\beta$",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_10_trend_sigmoid_beta")


# ========================================================================
# FIGURE 11: Gamma Blending Parameter Effect
# ========================================================================
def fig_11_gamma_blending():
    print("Generating Figure 11: Gamma Blending Effect...")
    # Scenario: one vital at 2.8/3 (severe), rest at 0.5/3
    max_per_vital = 3.0
    n_vitals = 6
    max_total = n_vitals * max_per_vital  # 18

    worst_vital_scores = np.linspace(0, 3, 200)
    other_vital_score = 0.3
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    gamma_colours = ["#CC3311", "#EE7733", "#E8C547", "#0077BB", "#009988"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Final score as worst vital increases, others fixed
    ax = axes[0]
    for gamma, colour in zip(gammas, gamma_colours):
        additive = worst_vital_scores + 5 * other_vital_score
        max_based = 6.0 * worst_vital_scores
        final = (1 - gamma) * max_based + gamma * additive
        ax.plot(worst_vital_scores, final, color=colour, linewidth=2.0,
                label=f"$\\gamma = {gamma:.2f}$")

    ax.set_xlabel("Most Abnormal Vital Score ($\\max_j y_j^\\prime$)")
    ax.set_ylabel("Final Aggregated Score ($S_{final}$)")
    ax.set_title("(a) Varying Worst Vital (Others = 0.3)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 20)
    ax.axhline(7, color="#888", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(0.1, 7.4, "Threshold = 7", fontsize=8, color="#888")

    # (b) Two contrasting patient scenarios
    ax = axes[1]
    gamma_range = np.linspace(0, 1, 200)

    # Patient A: single critical vital (2.8), rest normal (0.1)
    additive_A = 2.8 + 5 * 0.1
    max_based_A = 6.0 * 2.8
    scores_A = (1 - gamma_range) * max_based_A + gamma_range * additive_A

    # Patient B: moderate across all (1.2 each)
    additive_B = 6 * 1.2
    max_based_B = 6.0 * 1.2
    scores_B = (1 - gamma_range) * max_based_B + gamma_range * additive_B

    ax.plot(gamma_range, scores_A, color="#CC3311", linewidth=2.2,
            label="Patient A: single critical (2.8, rest 0.1)")
    ax.plot(gamma_range, scores_B, color="#0077BB", linewidth=2.2,
            label="Patient B: uniform moderate (all 1.2)")
    ax.axhline(7, color="#888", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(0.02, 7.4, "Threshold = 7", fontsize=8, color="#888")

    ax.set_xlabel("$\\gamma$ (Blending Parameter)")
    ax.set_ylabel("Final Score ($S_{final}$)")
    ax.set_title("(b) Clinical Impact of $\\gamma$ on Two Patient Profiles",
                 fontweight="bold")
    ax.legend(fontsize=8.5, loc="center right")
    ax.set_ylim(0, 20)

    fig.suptitle("$\\gamma$ Aggregation: Balancing Single-Vital Dominance vs Additive Scoring",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_11_gamma_blending_effect")


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating thesis figures for Chapter 5")
    print("=" * 60)
    fig_01_system_architecture()
    fig_02_news2_vs_fews()
    fig_03_gaussian_cdf_transition()
    fig_04_bidirectional_mfs()
    fig_05_unidirectional_mfs()
    fig_06_output_trapezoids()
    fig_07_defuzzification_example()
    fig_08_partition_of_unity()
    fig_09_ewma_alpha()
    fig_10_trend_sigmoid()
    fig_11_gamma_blending()
    print("=" * 60)
    print(f"All figures saved to: {OUT_DIR.resolve()}")
    print("=" * 60)
