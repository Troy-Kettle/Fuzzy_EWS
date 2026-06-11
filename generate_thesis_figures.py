"""
Generate all thesis figures for Chapter 5: Fuzzy System Development.
Outputs publication-quality figures (300 DPI, vector-compatible) to thesis_figures/.

Usage:
    python generate_thesis_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style: thesis-quality, restrained academic look
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "regular",
    "axes.labelpad": 4,
    "axes.linewidth": 0.8,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.borderaxespad": 0.4,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.6,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#222222",
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
    "patch.linewidth": 0.8,
})

OUT_DIR = Path(__file__).parent / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)

SIGMOID_DIR = Path(__file__).parent / "generated_membership_data" / "sigmoid"

# Tol "muted" palette — colourblind-safe, restrained, academic
C_RED = "#a4373a"
C_ORANGE = "#d68a3c"
C_YELLOW = "#cab441"
C_TEAL = "#3a8a82"
C_BLUE = "#3b6ea8"
C_GREEN = "#4a8a4a"
C_PURPLE = "#7b5a8c"
C_GREY = "#6e6e6e"

SEVEN_COLOURS = [C_RED, C_ORANGE, C_YELLOW, C_TEAL,
                 C_YELLOW, C_ORANGE, C_RED]

FOUR_COLOURS_DOWN = [C_RED, C_ORANGE, C_YELLOW, C_TEAL]
FOUR_COLOURS_UP = [C_TEAL, C_YELLOW, C_ORANGE, C_RED]

CONCERN_COLOURS = {
    "No Concern": C_TEAL,
    "Mild Concern": C_YELLOW,
    "Moderate Concern": C_ORANGE,
    "Severe Concern": C_RED,
}


def panel_label(ax, label, x=-0.13, y=1.03):
    """Add a small lower-case panel label, e.g. (a), in the upper-left corner."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10.5, fontweight="regular",
            ha="left", va="bottom", color="#222222", style="italic")


def style_axes(ax):
    """Apply uniform academic styling to an axes."""
    ax.tick_params(which="both", color="#888888")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#444444")
    ax.set_axisbelow(True)


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
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def draw_box(x, y, w, h, text, facecolour, edgecolour="#333333",
                 fontsize=9.5, italic=False):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.18,rounding_size=0.12",
            facecolor=facecolour, edgecolor=edgecolour,
            linewidth=0.9,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color="#1a1a1a", style="italic" if italic else "normal",
        )

    def arrow(x1, y1, x2, y2, text="", colour="#555555", offset=(0.08, 0.10)):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=colour,
                            linewidth=1.0, mutation_scale=10),
        )
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + offset[0], my + offset[1], text,
                    fontsize=8.5, color="#444444", style="italic")

    draw_box(0.2, 2.4, 1.6, 1.4, r"$V_t$" "\n" r"(6 vitals)",
             "#eaf1f8", edgecolour="#3b6ea8", fontsize=10.5)

    snap_outline = FancyBboxPatch(
        (2.6, 0.5), 3.1, 5.0,
        boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor="none", edgecolor="#3a8a82",
        linewidth=0.8, linestyle=(0, (4, 3)),
    )
    ax.add_patch(snap_outline)
    ax.text(4.15, 5.20, r"Snapshot Layer  $\mathcal{F}$",
            ha="center", fontsize=10.5, color="#2a5e58")

    draw_box(2.85, 3.85, 2.6, 0.75, "Fuzzification\n(membership functions)",
             "#dfece9", edgecolour="#3a8a82", fontsize=9)
    draw_box(2.85, 2.65, 2.6, 0.75, "Concern-level\nconsolidation",
             "#dfece9", edgecolour="#3a8a82", fontsize=9)
    draw_box(2.85, 1.45, 2.6, 0.75, "Centroid\ndefuzzification",
             "#dfece9", edgecolour="#3a8a82", fontsize=9)

    arrow(4.15, 3.85, 4.15, 3.40)
    arrow(4.15, 2.65, 4.15, 2.20)

    temp_outline = FancyBboxPatch(
        (6.3, 0.5), 3.1, 5.0,
        boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor="none", edgecolor="#d68a3c",
        linewidth=0.8, linestyle=(0, (4, 3)),
    )
    ax.add_patch(temp_outline)
    ax.text(7.85, 5.20, r"Temporal Layer  $\mathcal{T}$",
            ha="center", fontsize=10.5, color="#8b5520")

    draw_box(6.55, 3.85, 2.6, 0.75, "EWMA\nmemory retention",
             "#f6e6d3", edgecolour="#d68a3c", fontsize=9)
    draw_box(6.55, 2.65, 2.6, 0.75, "Trend extraction\n& sigmoid penalty",
             "#f6e6d3", edgecolour="#d68a3c", fontsize=9)
    draw_box(6.55, 1.45, 2.6, 0.75,
             r"$\gamma$ cross-vital" "\n" "aggregation",
             "#f6e6d3", edgecolour="#d68a3c", fontsize=9)

    arrow(7.85, 3.85, 7.85, 3.40)
    arrow(7.85, 2.65, 7.85, 2.20)

    arrow(1.8, 3.1, 2.85, 4.22)
    arrow(4.15, 1.45, 4.15, 0.85)
    ax.text(4.15, 0.55, r"$S_{\mathrm{snapshot},t}$", ha="center",
            fontsize=10, color="#2a5e58")
    arrow(5.7, 4.22, 6.55, 4.22, r"$S_{\mathrm{snapshot},1:t}$",
          offset=(-0.15, 0.18))
    arrow(7.85, 1.45, 7.85, 0.85)
    ax.text(7.85, 0.55, r"$S_{\mathrm{final},t}$", ha="center",
            fontsize=10, color="#8b5520")

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)

    ax = axes[0]
    ax.step(x, news2_score, where="post", color=C_RED, linewidth=1.6)
    ax.set_xlabel("Heart rate (bpm)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.2, 3.5)
    ax.set_xlim(30, 180)
    ax.set_yticks([0, 1, 2, 3])
    for boundary in [40, 51, 91, 111, 131]:
        ax.axvline(boundary, color="#bbbbbb", linestyle=":",
                   linewidth=0.6, alpha=0.7)
    ax.annotate(
        "90 → 91 bpm\n+1 score step",
        xy=(91, 0.05), xytext=(108, 0.85),
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7,
                        connectionstyle="arc3,rad=0.15"),
        fontsize=8.5, color="#555555",
    )
    style_axes(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(x, fews_score, color=C_BLUE, linewidth=1.6)
    ax.fill_between(x, fews_score, alpha=0.08, color=C_BLUE, linewidth=0)
    ax.set_xlabel("Heart rate (bpm)")
    ax.set_ylim(-0.2, 3.5)
    ax.set_xlim(30, 180)
    ax.annotate(
        "smooth response",
        xy=(91, float(np.interp(91, x, fews_score))),
        xytext=(115, 0.85),
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7,
                        connectionstyle="arc3,rad=0.15"),
        fontsize=8.5, color="#555555",
    )
    style_axes(ax)
    panel_label(ax, "(b)")

    fig.tight_layout()
    save(fig, "fig_02_news2_vs_fews_comparison")


# ========================================================================
# FIGURE 3: Individual Gaussian CDF Transition
# ========================================================================
def fig_03_gaussian_cdf_transition():
    print("Generating Figure 3: Gaussian CDF Transition...")
    from scipy.stats import norm

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    mu, sigma = 91, 5
    x = np.linspace(70, 115, 500)
    cdf = norm.cdf(x, mu, sigma)

    ax = axes[0]
    ax.plot(x, cdf, color=C_BLUE, linewidth=1.6)
    ax.axvline(mu, color="#aaaaaa", linestyle="--", linewidth=0.7)
    ax.axhline(0.5, color="#aaaaaa", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Vital sign value")
    ax.set_ylabel(r"$\Phi(x;\, \mu,\, \sigma)$")
    ax.annotate(rf"$\mu = {mu}$", xy=(mu, 0.5), xytext=(mu + 6, 0.32),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
                fontsize=9)
    ax.annotate(rf"$\sigma = {sigma}$", xy=(mu - sigma, norm.cdf(mu - sigma, mu, sigma)),
                xytext=(mu - 17, 0.62),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
                fontsize=9)
    style_axes(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    lower = 1 - cdf
    upper = cdf
    ax.plot(x, lower, color=C_RED, linewidth=1.6,
            label=r"$\mu_{L_{\mathrm{low}}} = 1 - \Phi$")
    ax.plot(x, upper, color=C_BLUE, linewidth=1.6,
            label=r"$\mu_{L_{\mathrm{high}}} = \Phi$")
    ax.axvline(mu, color="#aaaaaa", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Vital sign value")
    ax.set_ylabel("Membership degree")
    ax.legend(loc="center right")
    style_axes(ax)
    panel_label(ax, "(b)")

    ax = axes[2]
    mu_left, sigma_left = 80, 4
    mu_right, sigma_right = 100, 5
    cdf_left = norm.cdf(x, mu_left, sigma_left)
    cdf_right = norm.cdf(x, mu_right, sigma_right)
    mid_mf = np.clip(cdf_left - cdf_right, 0, None)
    ax.plot(x, mid_mf, color=C_ORANGE, linewidth=1.6,
            label=r"$\mu_{L_{\mathrm{mid}}} = \Phi_{\mathrm{left}} - \Phi_{\mathrm{right}}$")
    ax.fill_between(x, mid_mf, alpha=0.10, color=C_ORANGE, linewidth=0)
    ax.axvline(mu_left, color="#aaaaaa", linestyle="--", linewidth=0.6)
    ax.axvline(mu_right, color="#aaaaaa", linestyle="--", linewidth=0.6)
    ax.text(mu_left, 0.05, r"$\mu_{\mathrm{left}}$",
            ha="right", va="bottom", fontsize=9, color="#666666")
    ax.text(mu_right, 0.05, r"$\mu_{\mathrm{right}}$",
            ha="left", va="bottom", fontsize=9, color="#666666")
    ax.set_xlabel("Vital sign value")
    ax.set_ylabel("Membership degree")
    ax.legend(loc="upper left", fontsize=8.5)
    style_axes(ax)
    panel_label(ax, "(c)")

    for a in axes:
        a.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    save(fig, "fig_03_gaussian_cdf_construction")


# ========================================================================
# FIGURE 4: Bidirectional Membership Functions (2x2 grid)
# ========================================================================
def fig_04_bidirectional_mfs():
    print("Generating Figure 4: Bidirectional Membership Functions...")
    vitals = [
        ("Heart rate (bpm)", "heart_rate_membership_functions.csv"),
        ("Systolic blood pressure (mmHg)", "systolic_blood_pressure_membership_functions.csv"),
        ("Temperature (°C)", "temperature_membership_functions.csv"),
        ("Respiratory rate (breaths/min)", "respiratory_rate_membership_functions.csv"),
    ]
    short_labels = [
        "Below severe", "Below moderate", "Below mild", "No concern",
        "Above mild", "Above moderate", "Above severe",
    ]
    linestyles = ["-", "--", "-.", "-", "-.", "--", "-"]
    panels = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
    axes = axes.ravel()

    handles_for_legend = None
    for idx, (xlabel, fname) in enumerate(vitals):
        df = pd.read_csv(SIGMOID_DIR / fname)
        x = df["Value"].values
        cols = [c for c in df.columns if c != "Value"]
        ax = axes[idx]

        lines = []
        for i, col in enumerate(cols):
            line, = ax.plot(x, df[col].values, color=SEVEN_COLOURS[i],
                            linewidth=1.3, linestyle=linestyles[i])
            lines.append(line)
        if handles_for_legend is None:
            handles_for_legend = lines

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Membership degree $\mu$")
        ax.set_ylim(-0.05, 1.08)
        style_axes(ax)
        panel_label(ax, panels[idx])

    fig.legend(handles_for_legend, short_labels,
               loc="lower center", ncol=7, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01),
               handlelength=2.4, columnspacing=1.4)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig_04_bidirectional_membership_functions")


# ========================================================================
# FIGURE 5: Unidirectional Membership Functions (SpO2 + FiO2)
# ========================================================================
def fig_05_unidirectional_mfs():
    print("Generating Figure 5: Unidirectional Membership Functions...")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    df = pd.read_csv(SIGMOID_DIR / "oxygen_saturation_membership_functions.csv")
    cols = [c for c in df.columns if c != "Value"]
    spo2_labels = ["Below severe", "Below moderate", "Below mild", "No concern"]
    ax = axes[0]
    for i, col in enumerate(cols):
        ax.plot(df["Value"], df[col], color=FOUR_COLOURS_DOWN[i],
                linewidth=1.5, label=spo2_labels[i])
    ax.set_xlabel(r"SpO$_2$ (%)")
    ax.set_ylabel(r"Membership degree $\mu$")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.58))
    style_axes(ax)
    panel_label(ax, "(a)")

    df = pd.read_csv(SIGMOID_DIR / "inspired_oxygen_concentration_membership_functions.csv")
    cols = [c for c in df.columns if c != "Value"]
    fio2_labels = ["No concern", "Above mild", "Above moderate", "Above severe"]
    ax = axes[1]
    for i, col in enumerate(cols):
        ax.plot(df["Value"], df[col], color=FOUR_COLOURS_UP[i],
                linewidth=1.5, label=fio2_labels[i])
    ax.set_xlabel(r"FiO$_2$ (%)")
    ax.set_ylabel(r"Membership degree $\mu$")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.58))
    style_axes(ax)
    panel_label(ax, "(b)")

    fig.tight_layout()
    save(fig, "fig_05_unidirectional_membership_functions")


# ========================================================================
# FIGURE 6: Output Trapezoidal Membership Functions
# ========================================================================
def fig_06_output_trapezoids():
    print("Generating Figure 6: Output Trapezoids...")
    defs = {
        r"No concern $C_0$ ($w=0$)": (-0.5, 0, 0, 0.75),
        r"Mild concern $C_1$ ($w=1$)": (0.25, 1, 1, 1.75),
        r"Moderate concern $C_2$ ($w=2$)": (1.25, 2, 2, 2.75),
        r"Severe concern $C_3$ ($w=3$)": (2.25, 3, 3, 3.5),
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

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for idx, (label, params) in enumerate(defs.items()):
        y = [trap(xi, *params) for xi in x]
        ax.plot(x, y, color=colours[idx], linewidth=1.6, label=label)
        ax.fill_between(x, y, alpha=0.10, color=colours[idx], linewidth=0)

    ax.set_xlabel("Concern score (per vital)")
    ax.set_ylabel("Membership degree")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.03, 1.10)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22),
              handlelength=2.4)
    style_axes(ax)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "fig_06_output_membership_functions")


# ========================================================================
# FIGURE 7: Centroid Defuzzification Worked Example
# ========================================================================
def fig_07_defuzzification_example():
    print("Generating Figure 7: Centroid Defuzzification Example...")
    defs = {
        "No concern": (-0.5, 0, 0, 0.75),
        "Mild concern": (0.25, 1, 1, 1.75),
        "Moderate concern": (1.25, 2, 2, 2.75),
        "Severe concern": (2.25, 3, 3, 3.5),
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

    firings = {
        "No concern": 0.15,
        "Mild concern": 0.70,
        "Moderate concern": 0.15,
        "Severe concern": 0.0,
    }
    colours = list(CONCERN_COLOURS.values())

    x = np.linspace(-0.5, 3.5, 401)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))

    ax = axes[0]
    labels = list(firings.keys())
    vals = list(firings.values())
    bars = ax.barh(labels, vals, color=colours, edgecolor="none",
                   linewidth=0, height=0.55, alpha=0.9)
    ax.set_xlabel(r"Activation level $\mu_{C_i}$")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=9, color="#333333")
    style_axes(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    aggregated = np.zeros_like(x)
    for idx, (label, params) in enumerate(defs.items()):
        firing = firings[label]
        if firing <= 0:
            continue
        y_full = np.array([trap(xi, *params) for xi in x])
        y_clipped = np.minimum(y_full, firing)
        ax.plot(x, y_full, color=colours[idx], linewidth=0.7,
                alpha=0.45, linestyle="--")
        ax.fill_between(x, y_clipped, alpha=0.18,
                        color=colours[idx], linewidth=0)
        aggregated = np.maximum(aggregated, y_clipped)

    ax.plot(x, aggregated, color="#222222", linewidth=1.5)
    ax.fill_between(x, aggregated, alpha=0.05, color="#222222", linewidth=0)
    ax.set_xlabel("Concern score")
    ax.set_ylabel("Membership")
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.03, 1.08)
    style_axes(ax)
    panel_label(ax, "(b)")

    ax = axes[2]
    ax.fill_between(x, aggregated, alpha=0.18, color=C_BLUE, linewidth=0)
    ax.plot(x, aggregated, color=C_BLUE, linewidth=1.5)
    numerator = np.sum(x * aggregated)
    denominator = np.sum(aggregated)
    centroid = numerator / denominator if denominator > 0 else 0
    ax.axvline(centroid, color=C_RED, linewidth=1.5)
    ax.scatter([centroid], [0], color=C_RED, s=40, zorder=5, marker="^",
               clip_on=False)
    ax.text(centroid + 0.05, 1.0,
            rf"$z^\ast = {centroid:.2f}$",
            fontsize=9.5, color=C_RED, va="top")
    ax.set_xlabel("Concern score")
    ax.set_ylabel("Aggregated membership")
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.03, 1.08)
    style_axes(ax)
    panel_label(ax, "(c)")

    fig.tight_layout()
    save(fig, "fig_07_centroid_defuzzification_example")


# ========================================================================
# FIGURE 8: Partition of Unity
# ========================================================================
def fig_08_partition_of_unity():
    print("Generating Figure 8: Partition of Unity...")
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.4),
                              gridspec_kw={"height_ratios": [3, 1]},
                              sharex=True)

    df = pd.read_csv(SIGMOID_DIR / "heart_rate_membership_functions.csv")
    x = df["Value"].values
    cols = [c for c in df.columns if c != "Value"]
    short_labels = [
        "Below severe", "Below moderate", "Below mild", "No concern",
        "Above mild", "Above moderate", "Above severe",
    ]
    linestyles = ["-", "--", "-.", "-", "-.", "--", "-"]

    ax = axes[0]
    total = np.zeros_like(x, dtype=float)
    for i, col in enumerate(cols):
        vals = df[col].values
        ax.plot(x, vals, color=SEVEN_COLOURS[i], linewidth=1.3,
                label=short_labels[i], linestyle=linestyles[i])
        total += vals
    ax.set_ylabel(r"Membership degree $\mu$")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.16),
              fontsize=8.5, columnspacing=1.4, handlelength=2.2)
    style_axes(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(x, total, color=C_BLUE, linewidth=1.5)
    ax.axhline(1.0, color=C_RED, linestyle="--", linewidth=0.9)
    ax.set_xlabel("Heart rate (bpm)")
    ax.set_ylabel(r"$\sum_i \mu_i$")
    ax.set_ylim(0.92, 1.08)
    ax.set_yticks([0.95, 1.0, 1.05])
    style_axes(ax)
    panel_label(ax, "(b)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_08_partition_of_unity")


# ========================================================================
# FIGURE 9: EWMA with Different Alpha Values
# ========================================================================
def fig_09_ewma_alpha():
    print("Generating Figure 9: EWMA Behaviour...")
    raw = np.concatenate([
        np.array([0.2, 0.3, 0.2, 0.4, 0.3, 0.5, 0.4, 0.6]),
        np.array([1.0, 1.4, 1.8, 2.0, 2.2, 2.5, 2.3, 2.6]),
        np.array([2.4, 2.1, 1.8, 1.5]),
    ])
    t = np.arange(len(raw))

    alphas = [0.3, 0.5, 0.7, 0.9]
    alpha_colours = [C_RED, C_ORANGE, C_BLUE, C_TEAL]

    fig, ax = plt.subplots(figsize=(9.5, 4.0))

    ax.axvspan(7.5, 15.5, alpha=0.06, color=C_RED, linewidth=0)
    ax.axvspan(15.5, 19.5, alpha=0.06, color=C_TEAL, linewidth=0)

    ax.plot(t, raw, marker="o", linestyle=":", color="#888888",
            markersize=3.5, linewidth=0.9, label="Raw observations",
            markerfacecolor="white", markeredgewidth=0.9)

    for alpha, colour in zip(alphas, alpha_colours):
        ewma = [raw[0]]
        for v in raw[1:]:
            ewma.append(alpha * v + (1 - alpha) * ewma[-1])
        clamped = [max(smoothed, r) for smoothed, r in zip(ewma, raw)]
        ax.plot(t, clamped, color=colour, linewidth=1.5,
                label=rf"$\alpha = {alpha}$")

    ax.set_xlabel(r"Observation index $t$")
    ax.set_ylabel("Score (0–3)")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.1, 3.3)

    ax.text(11.5, 3.10, "deterioration", fontsize=8.5,
            style="italic", color="#8a4040", ha="center")
    ax.text(17.5, 3.10, "improvement", fontsize=8.5,
            style="italic", color="#2a5e58", ha="center")

    style_axes(ax)
    fig.tight_layout()
    save(fig, "fig_09_ewma_alpha_comparison")


# ========================================================================
# FIGURE 10: Modified Sigmoid Trend Factor
# ========================================================================
def fig_10_trend_sigmoid():
    print("Generating Figure 10: Modified Sigmoid Trend Factor...")
    slopes = np.linspace(-2, 4, 500)
    betas = [0.5, 1.0, 2.0, 5.0]
    beta_colours = [C_TEAL, C_BLUE, C_ORANGE, C_RED]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=False)

    ax = axes[0]
    for beta, colour in zip(betas, beta_colours):
        T = 2.0 / (1.0 + np.exp(-beta * slopes)) - 1.0
        ax.plot(slopes, T, color=colour, linewidth=1.5,
                label=rf"$\beta = {beta}$")
    ax.axhline(0, color="#bbbbbb", linewidth=0.6)
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    ax.set_xlabel(r"Slope $m_j$")
    ax.set_ylabel(r"Trend factor $T_j(m_j)$")
    ax.legend()
    ax.set_ylim(-1.15, 1.15)
    style_axes(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    for beta, colour in zip(betas, beta_colours):
        T = 2.0 / (1.0 + np.exp(-beta * slopes)) - 1.0
        T_clamped = np.where(slopes > 0, T, 0.0)
        ax.plot(slopes, T_clamped, color=colour, linewidth=1.5,
                label=rf"$\beta = {beta}$")
    ax.axhline(0, color="#bbbbbb", linewidth=0.6)
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    ax.fill_between(slopes, -0.1, 0, where=(slopes <= 0),
                    alpha=0.08, color=C_TEAL, linewidth=0,
                    label="Clamped to 0")
    ax.set_xlabel(r"Slope $m_j$")
    ax.set_ylabel(r"Applied trend factor $T_j^{\ast}$")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.12, 1.12)
    style_axes(ax)
    panel_label(ax, "(b)")

    fig.tight_layout()
    save(fig, "fig_10_trend_sigmoid_beta")


# ========================================================================
# FIGURE 11: Gamma Blending Parameter Effect
# ========================================================================
def fig_11_gamma_blending():
    print("Generating Figure 11: Gamma Blending Effect...")
    worst_vital_scores = np.linspace(0, 3, 200)
    other_vital_score = 0.3
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    gamma_colours = [C_RED, C_ORANGE, C_YELLOW, C_BLUE, C_TEAL]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)

    ax = axes[0]
    for gamma, colour in zip(gammas, gamma_colours):
        additive = worst_vital_scores + 5 * other_vital_score
        max_based = 6.0 * worst_vital_scores
        final = (1 - gamma) * max_based + gamma * additive
        ax.plot(worst_vital_scores, final, color=colour, linewidth=1.5,
                label=rf"$\gamma = {gamma:.2f}$")
    ax.set_xlabel(r"Most abnormal vital score $\max_j y_j^\prime$")
    ax.set_ylabel(r"Final aggregated score $S_{\mathrm{final}}$")
    ax.axhline(7, color="#9a9a9a", linestyle=":", linewidth=0.8)
    ax.text(2.95, 7.5, "threshold", fontsize=8, color="#777777",
            style="italic", ha="right")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 20)
    style_axes(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    gamma_range = np.linspace(0, 1, 200)
    additive_A = 2.8 + 5 * 0.1
    max_based_A = 6.0 * 2.8
    scores_A = (1 - gamma_range) * max_based_A + gamma_range * additive_A
    additive_B = 6 * 1.2
    max_based_B = 6.0 * 1.2
    scores_B = (1 - gamma_range) * max_based_B + gamma_range * additive_B

    ax.plot(gamma_range, scores_A, color=C_RED, linewidth=1.6,
            label="Patient A: single critical (2.8, rest 0.1)")
    ax.plot(gamma_range, scores_B, color=C_BLUE, linewidth=1.6,
            label="Patient B: uniform moderate (all 1.2)")
    ax.axhline(7, color="#9a9a9a", linestyle=":", linewidth=0.8)
    ax.text(0.97, 7.5, "threshold", fontsize=8, color="#777777",
            style="italic", ha="right")
    ax.set_xlabel(r"Blending parameter $\gamma$")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 20)
    style_axes(ax)
    panel_label(ax, "(b)")

    fig.tight_layout()
    save(fig, "fig_11_gamma_blending_effect")


# ========================================================================
# FIGURE 12: All Membership Functions Overview (6 vitals)
# ========================================================================
def fig_12_all_membership_functions():
    print("Generating Figure 12: All Membership Functions Overview...")
    bidir_labels = [
        "Below severe", "Below moderate", "Below mild", "No concern",
        "Above mild", "Above moderate", "Above severe",
    ]
    bidir_styles = ["-", "--", "-.", "-", "-.", "--", "-"]

    bidir_vitals = [
        ("Heart rate (bpm)", "heart_rate_membership_functions.csv"),
        ("Systolic blood pressure (mmHg)", "systolic_blood_pressure_membership_functions.csv"),
        ("Temperature (°C)", "temperature_membership_functions.csv"),
        ("Respiratory rate (breaths/min)", "respiratory_rate_membership_functions.csv"),
    ]
    unidir_vitals = [
        (r"SpO$_2$ (%)", "oxygen_saturation_membership_functions.csv",
         FOUR_COLOURS_DOWN,
         ["Below severe", "Below moderate", "Below mild", "No concern"]),
        (r"FiO$_2$ (%)", "inspired_oxygen_concentration_membership_functions.csv",
         FOUR_COLOURS_UP,
         ["No concern", "Above mild", "Above moderate", "Above severe"]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.0))
    panels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    legend_handles = None
    for idx, (xlabel, fname) in enumerate(bidir_vitals):
        ax = axes.flat[idx]
        df = pd.read_csv(SIGMOID_DIR / fname)
        x = df["Value"].values
        cols = [c for c in df.columns if c != "Value"]
        lines = []
        for i, col in enumerate(cols):
            line, = ax.plot(x, df[col].values, color=SEVEN_COLOURS[i],
                            linewidth=1.3, linestyle=bidir_styles[i])
            lines.append(line)
        if legend_handles is None:
            legend_handles = lines
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Membership degree $\mu$")
        ax.set_ylim(-0.05, 1.08)
        style_axes(ax)
        panel_label(ax, panels[idx])

    for j, (xlabel, fname, colours, labels) in enumerate(unidir_vitals):
        ax = axes.flat[4 + j]
        df = pd.read_csv(SIGMOID_DIR / fname)
        cols = [c for c in df.columns if c != "Value"]
        for i, col in enumerate(cols):
            ax.plot(df["Value"], df[col], color=colours[i],
                    linewidth=1.5, label=labels[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Membership degree $\mu$")
        ax.set_ylim(-0.05, 1.08)
        loc = "center left" if j == 0 else "center right"
        anchor = (0.02, 0.58) if j == 0 else (0.98, 0.58)
        ax.legend(loc=loc, bbox_to_anchor=anchor, fontsize=8.5)
        style_axes(ax)
        panel_label(ax, panels[4 + j])

    fig.legend(legend_handles, bidir_labels,
               loc="lower center", ncol=7, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.005),
               handlelength=2.4, columnspacing=1.4)

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    save(fig, "fig_12_all_membership_functions")


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
    fig_12_all_membership_functions()
    print("=" * 60)
    print(f"All figures saved to: {OUT_DIR.resolve()}")
    print("=" * 60)
