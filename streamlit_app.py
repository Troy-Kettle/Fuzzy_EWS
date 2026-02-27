import altair as alt
import ast
import pandas as pd
import streamlit as st
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

DATA_DIR_DEFAULT = Path(__file__).parent / "data" / "membership_function_plots" / "csv_data"
DATA_DIR_SIGMOID = Path(__file__).parent / "generated_membership_data" / "sigmoid"
# NOTE: Generated trapezoidal outputs have been removed; only sigmoid generation is supported.
PART2_SURVEY_PATH = (
    Path(__file__).parent / "data" / "membership_function_plots" / "csv_data" / "part2_raw.xlsx"
)
PART3_SURVEY_PATH = (
    Path(__file__).parent / "data" / "membership_function_plots" / "part3_raw.xlsx"
)

PART2_COMBINATIONS = {
    1: {"hr": 110, "bp": 130, "resp": 18, "temp": 36.8, "ox": 98},
    2: {"hr": 75, "bp": 95, "resp": 18, "temp": 36.8, "ox": 98},
    3: {"hr": 75, "bp": 130, "resp": 24, "temp": 36.8, "ox": 98},
    4: {"hr": 75, "bp": 130, "resp": 18, "temp": 38.2, "ox": 98},
    5: {"hr": 75, "bp": 130, "resp": 18, "temp": 36.8, "ox": 92},
    6: {"hr": 110, "bp": 95, "resp": 18, "temp": 36.8, "ox": 98},
    7: {"hr": 110, "bp": 130, "resp": 24, "temp": 36.8, "ox": 98},
    8: {"hr": 110, "bp": 130, "resp": 18, "temp": 38.2, "ox": 98},
    9: {"hr": 110, "bp": 130, "resp": 18, "temp": 36.8, "ox": 92},
    10: {"hr": 75, "bp": 95, "resp": 24, "temp": 36.8, "ox": 98},
    11: {"hr": 75, "bp": 95, "resp": 18, "temp": 38.2, "ox": 98},
    12: {"hr": 75, "bp": 95, "resp": 18, "temp": 36.8, "ox": 92},
    13: {"hr": 75, "bp": 130, "resp": 24, "temp": 38.2, "ox": 98},
    14: {"hr": 75, "bp": 130, "resp": 24, "temp": 36.8, "ox": 92},
    15: {"hr": 75, "bp": 130, "resp": 18, "temp": 38.2, "ox": 92},
}


@dataclass(frozen=True)
class Observation:
    hr: int
    bp: int
    temp: float
    resp: float
    ox_sats: float
    insp_ox: float


def _interp_lookup(fs: dict, inp: float) -> float:
    """Lookup with linear interpolation for non-integer inputs."""
    if inp in fs:
        return float(fs[inp])
    keys = sorted(fs.keys())
    if not keys:
        return 0.0
    if inp <= keys[0]:
        return float(fs[keys[0]])
    if inp >= keys[-1]:
        return float(fs[keys[-1]])
    # Find bracketing keys
    for i in range(len(keys) - 1):
        if keys[i] <= inp < keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            t = (inp - lo) / (hi - lo)
            return float(fs[lo]) * (1 - t) + float(fs[hi]) * t
    return 0.0


class custom_mf_7_var:
    """Input membership function for vitals with 7 categories (e.g., HR, BP, Temp, Resp)."""

    def __init__(self, path: Path):
        self.df = pd.read_csv(path)
        keys = self.df.loc[:, "Value"].values
        self.B_SevC = dict(zip(keys, self.df.loc[:, "Below normal - severe concern"].values))
        self.B_ModC = dict(zip(keys, self.df.loc[:, "Below normal - moderate concern"].values))
        self.B_MildC = dict(zip(keys, self.df.loc[:, "Below normal - mild concern"].values))
        self.no_con = dict(zip(keys, self.df.loc[:, "No concern"].values))
        self.A_MildC = dict(zip(keys, self.df.loc[:, "Above normal - mild concern"].values))
        self.A_ModC = dict(zip(keys, self.df.loc[:, "Above normal - moderate concern"].values))
        self.A_SevC = dict(zip(keys, self.df.loc[:, "Above normal - severe concern"].values))
        self.fs = [
            self.B_SevC,
            self.B_ModC,
            self.B_MildC,
            self.no_con,
            self.A_MildC,
            self.A_ModC,
            self.A_SevC,
        ]
        self.labels = [
            "Below normal - severe concern",
            "Below normal - moderate concern",
            "Below normal - mild concern",
            "No concern",
            "Above normal - mild concern",
            "Above normal - moderate concern",
            "Above normal - severe concern",
        ]

    def __call__(self, inp: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for label, fs in zip(self.labels, self.fs):
            out[label] = _interp_lookup(fs, inp)
        return out


class custom_mf_3_var_up:
    """Input membership function for vitals with 3 upward categories (e.g., Inspired O2)."""

    def __init__(self, path: Path):
        self.df = pd.read_csv(path)
        keys = self.df.loc[:, "Value"].values
        self.no_con = dict(zip(keys, self.df.loc[:, "No concern"].values))
        self.A_MildC = dict(zip(keys, self.df.loc[:, "Above normal - mild concern"].values))
        self.A_ModC = dict(zip(keys, self.df.loc[:, "Above normal - moderate concern"].values))
        self.A_SevC = dict(zip(keys, self.df.loc[:, "Above normal - severe concern"].values))
        self.fs = [self.no_con, self.A_MildC, self.A_ModC, self.A_SevC]
        self.labels = [
            "No concern",
            "Above normal - mild concern",
            "Above normal - moderate concern",
            "Above normal - severe concern",
        ]

    def __call__(self, inp: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for label, fs in zip(self.labels, self.fs):
            out[label] = _interp_lookup(fs, inp)
        return out


class custom_mf_3_var_down:
    """Input membership function for vitals with 3 downward categories (e.g., O2 Saturation)."""

    def __init__(self, path: Path):
        self.df = pd.read_csv(path)
        keys = self.df.loc[:, "Value"].values
        self.B_SevC = dict(zip(keys, self.df.loc[:, "Below normal - severe concern"].values))
        self.B_ModC = dict(zip(keys, self.df.loc[:, "Below normal - moderate concern"].values))
        self.B_MildC = dict(zip(keys, self.df.loc[:, "Below normal - mild concern"].values))
        self.no_con = dict(zip(keys, self.df.loc[:, "No concern"].values))
        self.fs = [self.B_SevC, self.B_ModC, self.B_MildC, self.no_con]
        self.labels = [
            "Below normal - severe concern",
            "Below normal - moderate concern",
            "Below normal - mild concern",
            "No concern",
        ]

    def __call__(self, inp: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for label, fs in zip(self.labels, self.fs):
            out[label] = _interp_lookup(fs, inp)
        return out


@lru_cache(maxsize=1)
def load_part2_survey_means() -> Dict[int, float]:
    """Return mean clinician rating per combination from part2 survey data."""
    if not PART2_SURVEY_PATH.exists():
        return {}
    df = pd.read_excel(PART2_SURVEY_PATH)
    if "part2Data" not in df.columns:
        return {}
    ratings: Dict[int, list] = {}
    for raw in df["part2Data"].dropna():
        try:
            entries = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            try:
                combo = int(entry.get("Combination"))
                rating = float(entry.get("Rating"))
            except Exception:
                continue
            ratings.setdefault(combo, []).append(rating)
    return {combo: float(sum(vals) / len(vals)) for combo, vals in ratings.items() if vals}


def _part2_distance(obs: Observation, combo: Dict[str, float]) -> float:
    """Normalized distance between observation and a part2 combination."""
    ranges = {
        "hr": 110 - 75,
        "bp": 130 - 95,
        "resp": 24 - 18,
        "temp": 38.2 - 36.8,
        "ox": 98 - 92,
    }
    dh = (obs.hr - combo["hr"]) / ranges["hr"]
    db = (obs.bp - combo["bp"]) / ranges["bp"]
    dr = (obs.resp - combo["resp"]) / ranges["resp"]
    dt = (obs.temp - combo["temp"]) / ranges["temp"]
    do = (obs.ox_sats - combo["ox"]) / ranges["ox"]
    return float((dh * dh + db * db + dr * dr + dt * dt + do * do) ** 0.5)


def compute_part2_blend(obs: Observation, k: int = 3) -> Tuple[float | None, list]:
    """Return weighted mean rating using top-k nearest part2 combinations."""
    survey_means = load_part2_survey_means()
    if not survey_means:
        return None, []

    scored = []
    for combo_id, combo in PART2_COMBINATIONS.items():
        if combo_id not in survey_means:
            continue
        dist = _part2_distance(obs, combo)
        scored.append((combo_id, dist, survey_means[combo_id]))

    if not scored:
        return None, []

    scored.sort(key=lambda item: item[1])
    top = scored[: max(1, k)]
    weights = []
    for _, dist, _ in top:
        weights.append(1.0 / (dist + 1e-6))
    weight_sum = sum(weights)
    blended = sum(w * rating for w, (_, _, rating) in zip(weights, top)) / weight_sum
    details = [
        {
            "Combination": combo_id,
            "Distance": round(dist, 3),
            "Weight": round(w / weight_sum, 3),
            "Mean rating": round(rating, 2),
        }
        for (combo_id, dist, rating), w in zip(top, weights)
    ]
    return float(blended), details


class OutputMF:
    """Fuzzy output membership function for a SINGLE vital (0-3 scale)."""

    def __init__(self):
        self.definitions = {
            "No concern": (-0.5, 0, 0, 0.75),
            "Mild concern": (0.25, 1, 1, 1.75),
            "Moderate concern": (1.25, 2, 2, 2.75),
            "Severe concern": (2.25, 3, 3, 3.5),
        }

    def trapezoid(self, x: float, a: float, b: float, c: float, d: float) -> float:
        if b <= x <= c:
            return 1.0
        if x <= a or x >= d:
            return 0.0
        if a < x < b:
            return (x - a) / (b - a)
        if c < x < d:
            return (d - x) / (d - c)
        return 0.0

    def __call__(self, x: float) -> Dict[str, float]:
        return {label: self.trapezoid(x, *params) for label, params in self.definitions.items()}


@lru_cache(maxsize=4)
def _load_membership_functions_from(dir_str: str) -> Tuple[
    custom_mf_7_var,
    custom_mf_7_var,
    custom_mf_7_var,
    custom_mf_7_var,
    custom_mf_3_var_down,
    custom_mf_3_var_up,
]:
    base = Path(dir_str)
    return (
        custom_mf_7_var(base / "heart_rate_membership_functions.csv"),
        custom_mf_7_var(base / "systolic_blood_pressure_membership_functions.csv"),
        custom_mf_7_var(base / "temperature_membership_functions.csv"),
        custom_mf_7_var(base / "respiratory_rate_membership_functions.csv"),
        custom_mf_3_var_down(base / "oxygen_saturation_membership_functions.csv"),
        custom_mf_3_var_up(base / "inspired_oxygen_concentration_membership_functions.csv"),
    )


def load_membership_functions(base_dir: Path) -> Tuple[
    custom_mf_7_var,
    custom_mf_7_var,
    custom_mf_7_var,
    custom_mf_7_var,
    custom_mf_3_var_down,
    custom_mf_3_var_up,
]:
    return _load_membership_functions_from(str(base_dir.resolve()))


@st.cache_resource(show_spinner=False)
def output_cache() -> Tuple[OutputMF, Dict[float, Dict[str, float]]]:
    output = OutputMF()
    cache: Dict[float, Dict[str, float]] = {}
    for i in range(0, 301):
        x = i / 100.0
        cache[x] = output(x)
    return output, cache


def firings(hr: int, bp: int, temp: float, resp: float, ox: float, insp: float, base_dir: Path) -> Dict[str, Dict[str, float]]:
    heart_rate, blood_pressure, temperature, respiratory_rate, oxygen_saturation, inspired_oxygen = load_membership_functions(base_dir)
    return {
        "heart rate": heart_rate(hr),
        "blood pressure": blood_pressure(bp),
        "temperature": temperature(temp),
        "respiratory rate": respiratory_rate(resp),
        "oxygen saturation": oxygen_saturation(ox),
        "inspired oxygen": inspired_oxygen(insp),
    }


def map_to_concern_levels(vital_memberships: Dict[str, float]) -> Dict[str, float]:
    mapping = {"No concern": 0.0, "Mild concern": 0.0, "Moderate concern": 0.0, "Severe concern": 0.0}
    for key, value in vital_memberships.items():
        key_lower = key.lower()
        if "severe" in key_lower:
            mapping["Severe concern"] = max(mapping["Severe concern"], value)
        elif "moderate" in key_lower:
            mapping["Moderate concern"] = max(mapping["Moderate concern"], value)
        elif "mild" in key_lower:
            mapping["Mild concern"] = max(mapping["Mild concern"], value)
        elif "no concern" in key_lower:
            mapping["No concern"] = max(mapping["No concern"], value)
    return mapping


def defuzz_vital_centroid(concern_levels: Dict[str, float]) -> float:
    # Ignore very small firings caused by overlapping membership edges
    # which can otherwise produce a small non-zero centroid for normal inputs.
    MIN_FIRING = 0.05
    concern = {k: (v if v >= MIN_FIRING else 0.0) for k, v in concern_levels.items()}

    # If only the "No concern" set remains active, return exact zero.
    if concern.get("No concern", 0.0) > 0 and all(
        (level == "No concern") or (firing == 0.0) for level, firing in concern.items()
    ):
        return 0.0

    output, cache = output_cache()
    numerator = 0.0
    denominator = 0.0
    for i in range(0, 301):
        x = i / 100.0
        output_memberships = cache[x]
        aggregated = 0.0
        for level, firing in concern.items():
            if firing > 0:
                membership = output_memberships.get(level, 0)
                aggregated = max(aggregated, min(firing, membership))
        numerator += x * aggregated
        denominator += aggregated
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_fuzzy_ews_additive(all_firings: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    per_vital_scores: Dict[str, float] = {}
    total = 0.0
    for vital_name, vital_memberships in all_firings.items():
        concern_levels = map_to_concern_levels(vital_memberships)
        score = defuzz_vital_centroid(concern_levels)
        per_vital_scores[vital_name] = score
        total += score
    per_vital_scores["total"] = total
    return per_vital_scores


def aggregate_total(scores: Dict[str, float], method: str, power: float = 2.0) -> float:
    per_vital = [v for k, v in scores.items() if k != "total"]
    if not per_vital:
        return 0.0
    max_per_vital = 3.0
    n = float(len(per_vital))
    normalized = [min(max(v / max_per_vital, 0.0), 1.0) for v in per_vital]

    if method == "multiplicative":
        product = 1.0
        for val in normalized:
            product *= (1.0 - val)
        return (1.0 - product) * max_per_vital * n

    if method == "nonlinear":
        avg_power = sum(val ** power for val in normalized) / n
        return (avg_power ** (1.0 / power)) * max_per_vital * n

    return sum(per_vital)


def calculate_fuzzy_ews(all_firings: Dict[str, Dict[str, float]], method: str) -> Dict[str, float]:
    per_vital_scores: Dict[str, float] = {}
    for vital_name, vital_memberships in all_firings.items():
        concern_levels = map_to_concern_levels(vital_memberships)
        score = defuzz_vital_centroid(concern_levels)
        per_vital_scores[vital_name] = score
    per_vital_scores["total"] = aggregate_total(per_vital_scores, method)
    return per_vital_scores


def dominant_label(vital_memberships: Dict[str, float]) -> Tuple[str, float]:
    if not vital_memberships:
        return "No data", 0.0
    label, strength = max(vital_memberships.items(), key=lambda item: item[1])
    return label, strength


def calculate_news2(obs: Observation) -> Tuple[Dict[str, int], int]:
    """Compute NEWS-2 using Scale 1 thresholds; assumes inspired O2 > 21% means supplemental oxygen."""

    def score_resp(x: float) -> int:
        if x <= 8:
            return 3
        if 9 <= x <= 11:
            return 1
        if 12 <= x <= 20:
            return 0
        if 21 <= x <= 24:
            return 2
        return 3  # >=25

    def score_spo2(x: float) -> int:
        if x <= 91:
            return 3
        if 92 <= x <= 93:
            return 2
        if 94 <= x <= 95:
            return 1
        return 0  # >=96

    def score_temp(x: float) -> int:
        if x <= 35.0:
            return 3
        if 35.1 <= x <= 36.0:
            return 1
        if 36.1 <= x <= 38.0:
            return 0
        if 38.1 <= x <= 39.0:
            return 1
        return 2  # >=39.1

    def score_bp(x: float) -> int:
        if x <= 90:
            return 3
        if 91 <= x <= 100:
            return 2
        if 101 <= x <= 110:
            return 1
        if 111 <= x <= 219:
            return 0
        return 3  # >=220

    def score_hr(x: float) -> int:
        if x <= 40:
            return 3
        if 41 <= x <= 50:
            return 1
        if 51 <= x <= 90:
            return 0
        if 91 <= x <= 110:
            return 1
        if 111 <= x <= 130:
            return 2
        return 3  # >=131

    supplemental_o2 = obs.insp_ox > 21

    per_vital = {
        "respiratory rate": score_resp(obs.resp),
        "oxygen saturation": score_spo2(obs.ox_sats),
        "temperature": score_temp(obs.temp),
        "blood pressure": score_bp(obs.bp),
        "heart rate": score_hr(obs.hr),
        # Consciousness not captured; assumed 0.
    }

    total = sum(per_vital.values()) + (2 if supplemental_o2 else 0)
    per_vital["supplemental oxygen"] = 2 if supplemental_o2 else 0
    per_vital["consciousness"] = 0
    return per_vital, total


def clamp_observation(obs: Observation, base_dir: Path) -> Observation:
    """Clamp observation to the min/max of the selected membership grids."""
    hr_mf, bp_mf, temp_mf, resp_mf, ox_mf, insp_mf = load_membership_functions(base_dir)

    def clamp_val(val, series, *, round_1=False):
        lo, hi = float(series.min()), float(series.max())
        out = max(lo, min(hi, val))
        return round(out, 1) if round_1 else out

    hr = clamp_val(obs.hr, hr_mf.df["Value"])
    bp = clamp_val(obs.bp, bp_mf.df["Value"])
    temp = clamp_val(obs.temp, temp_mf.df["Value"], round_1=True)
    resp = clamp_val(obs.resp, resp_mf.df["Value"])
    ox = clamp_val(obs.ox_sats, ox_mf.df["Value"])
    insp = clamp_val(obs.insp_ox, insp_mf.df["Value"])
    return Observation(hr=int(hr), bp=int(bp), temp=float(temp), resp=float(resp), ox_sats=float(ox), insp_ox=float(insp))


def interpret_table(all_firings: Dict[str, Dict[str, float]], scores: Dict[str, float], news_scores: Dict[str, int], obs: Observation) -> pd.DataFrame:
    records = []
    vocab = {
        "heart rate": (obs.hr, "bpm"),
        "blood pressure": (obs.bp, "mmHg"),
        "temperature": (obs.temp, "\u00b0C"),
        "respiratory rate": (obs.resp, "breaths/min"),
        "oxygen saturation": (obs.ox_sats, "%"),
        "inspired oxygen": (obs.insp_ox, "% FiO2"),
    }
    for vital, memberships in all_firings.items():
        label, strength = dominant_label(memberships)
        score = scores.get(vital, 0.0)
        news_score = news_scores.get(vital, 0)
        value, unit = vocab.get(vital, ("", ""))
        description = f"{vital.title()} sits in '{label}' with strength {strength:.2f}, contributing {score:.2f} to the total risk." if strength > 0 else "No activation for this vital in the membership grid."
        records.append(
            {
                "Vital": vital.title(),
                "Input": f"{value} {unit}".strip(),
                "Dominant concern": label,
                "Strength (0-1)": round(strength, 2),
                "Per-vital score (0-3)": round(score, 2),
                "NEWS-2 score": news_score,
                "Explanation": description,
            }
        )
    return pd.DataFrame(records)


def membership_chart(df: pd.DataFrame, value: float, unit: str) -> alt.Chart:
    """Render a membership function chart with the input value highlighted."""
    long_df = df.melt(id_vars="Value", var_name="Membership", value_name="mu")
    base = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("Value:Q", title=f"Value ({unit})"),
            y=alt.Y("mu:Q", title="Membership"),
            color=alt.Color("Membership:N", legend=alt.Legend(title="Fuzzy set")),
            tooltip=["Membership:N", "Value:Q", "mu:Q"],
        )
    )
    rule = alt.Chart(pd.DataFrame({"Value": [value]})).mark_rule(color="red").encode(x="Value:Q")
    return (base + rule).properties(height=220)


def firing_table_df(labels, firing: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Membership": label, "Firing strength (0-1)": round(firing.get(label, 0.0), 3)} for label in labels]
    )


def risk_bucket(total: float) -> str:
    if total < 4:
        return "Low"
    if total < 8:
        return "Moderate"
    if total < 12:
        return "High"
    return "Critical"


# ---------------------------------------------------------------------------
# Interval Type-2 Fuzzy Logic System  (IT2FLS)
# ---------------------------------------------------------------------------

def _build_it2_bounds(fs_dict: dict, all_keys: list, spread: int) -> Tuple[dict, dict]:
    """Create upper (UMF) and lower (LMF) MF dicts with distance-weighted blending.

    Each neighbour's influence decays linearly with distance so that crisp
    transitions produce bounded FOUs rather than full-width 0-to-1 bands.
    """
    if spread == 0:
        vals = {k: float(fs_dict.get(k, 0.0)) for k in all_keys}
        return dict(vals), dict(vals)

    n = len(all_keys)
    t1 = [float(fs_dict.get(all_keys[i], 0.0)) for i in range(n)]

    umf, lmf = {}, {}
    for i in range(n):
        lo = max(0, i - spread)
        hi = min(n - 1, i + spread)

        u_val = t1[i]
        l_val = t1[i]
        for j in range(lo, hi + 1):
            if j == i:
                continue
            w = 1.0 - abs(j - i) / (spread + 1.0)
            blended = t1[i] + w * (t1[j] - t1[i])
            u_val = max(u_val, blended)
            l_val = min(l_val, blended)

        umf[all_keys[i]] = min(1.0, u_val)
        lmf[all_keys[i]] = max(0.0, l_val)
    return umf, lmf


def _spread_for_vital(n_points: int, pct: float) -> int:
    """Convert a percentage FOU width into an integer spread for a given vital.

    No artificial floor — small-range vitals can legitimately get spread 0
    (i.e. no FOU) when the percentage is low, avoiding disproportionately
    wide uncertainty bands on dense or narrow grids.
    """
    return round(n_points * pct / 100.0)


class IT2_mf_7_var:
    """IT2 input membership function with 7 categories and FOU."""

    def __init__(self, path: Path, spread_pct: float = 3.0):
        self.t1 = custom_mf_7_var(path)
        self.df = self.t1.df
        self.labels = self.t1.labels
        keys = list(self.df["Value"].values)
        self.spread = _spread_for_vital(len(keys), spread_pct)
        self.umf_fs: list[dict] = []
        self.lmf_fs: list[dict] = []
        for fs in self.t1.fs:
            umf, lmf = _build_it2_bounds(fs, keys, self.spread)
            self.umf_fs.append(umf)
            self.lmf_fs.append(lmf)

    def __call__(self, inp: float) -> Dict[str, Tuple[float, float]]:
        return {
            label: (_interp_lookup(lmf, inp), _interp_lookup(umf, inp))
            for label, umf, lmf in zip(self.labels, self.umf_fs, self.lmf_fs)
        }


class IT2_mf_3_var_up:
    """IT2 input membership function with 3 upward categories and FOU."""

    def __init__(self, path: Path, spread_pct: float = 3.0):
        self.t1 = custom_mf_3_var_up(path)
        self.df = self.t1.df
        self.labels = self.t1.labels
        keys = list(self.df["Value"].values)
        self.spread = _spread_for_vital(len(keys), spread_pct)
        self.umf_fs: list[dict] = []
        self.lmf_fs: list[dict] = []
        for fs in self.t1.fs:
            umf, lmf = _build_it2_bounds(fs, keys, self.spread)
            self.umf_fs.append(umf)
            self.lmf_fs.append(lmf)

    def __call__(self, inp: float) -> Dict[str, Tuple[float, float]]:
        return {
            label: (_interp_lookup(lmf, inp), _interp_lookup(umf, inp))
            for label, umf, lmf in zip(self.labels, self.umf_fs, self.lmf_fs)
        }


class IT2_mf_3_var_down:
    """IT2 input membership function with 3 downward categories and FOU."""

    def __init__(self, path: Path, spread_pct: float = 3.0):
        self.t1 = custom_mf_3_var_down(path)
        self.df = self.t1.df
        self.labels = self.t1.labels
        keys = list(self.df["Value"].values)
        self.spread = _spread_for_vital(len(keys), spread_pct)
        self.umf_fs: list[dict] = []
        self.lmf_fs: list[dict] = []
        for fs in self.t1.fs:
            umf, lmf = _build_it2_bounds(fs, keys, self.spread)
            self.umf_fs.append(umf)
            self.lmf_fs.append(lmf)

    def __call__(self, inp: float) -> Dict[str, Tuple[float, float]]:
        return {
            label: (_interp_lookup(lmf, inp), _interp_lookup(umf, inp))
            for label, umf, lmf in zip(self.labels, self.umf_fs, self.lmf_fs)
        }


class IT2OutputMF:
    """IT2 output membership function with trapezoidal UMF / LMF on the 0-3 scale."""

    _T1_DEFS = {
        "No concern": (-0.5, 0, 0, 0.75),
        "Mild concern": (0.25, 1, 1, 1.75),
        "Moderate concern": (1.25, 2, 2, 2.75),
        "Severe concern": (2.25, 3, 3, 3.5),
    }

    def __init__(self, delta: float = 0.15):
        self.umf_defs: Dict[str, Tuple[float, float, float, float]] = {}
        self.lmf_defs: Dict[str, Tuple[float, float, float, float]] = {}
        for k, (a, b, c, d) in self._T1_DEFS.items():
            self.umf_defs[k] = (a - delta, b - delta, c + delta, d + delta)
            nb, nc = b + delta, c - delta
            if nb > nc:
                nb = nc = (b + c) / 2.0
            self.lmf_defs[k] = (a + delta, nb, nc, d - delta)

    @staticmethod
    def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
        if b <= x <= c:
            return 1.0
        if x <= a or x >= d:
            return 0.0
        if a < x < b:
            return (x - a) / (b - a)
        if c < x < d:
            return (d - x) / (d - c)
        return 0.0

    def __call__(self, x: float) -> Dict[str, Tuple[float, float]]:
        return {
            label: (self._trap(x, *self.lmf_defs[label]), self._trap(x, *self.umf_defs[label]))
            for label in self.umf_defs
        }


@st.cache_resource(show_spinner=False)
def _it2_output_cache() -> Tuple[IT2OutputMF, Dict[float, Dict[str, Tuple[float, float]]]]:
    it2_out = IT2OutputMF()
    cache: Dict[float, Dict[str, Tuple[float, float]]] = {}
    for i in range(301):
        x = i / 100.0
        cache[x] = it2_out(x)
    return it2_out, cache


def km_type_reduce(
    x_points: list, lower_weights: list, upper_weights: list,
) -> Tuple[float, float]:
    """Karnik-Mendel centroid type-reduction.  Returns (y_l, y_r)."""
    n = len(x_points)
    if n == 0 or max(upper_weights) == 0:
        return 0.0, 0.0

    # y_l  –  minimise the centroid
    theta = [(l + u) / 2.0 for l, u in zip(lower_weights, upper_weights)]
    y_l = 0.0
    for _ in range(50):
        den = sum(theta)
        if den == 0:
            y_l = 0.0
            break
        y_l = sum(x * t for x, t in zip(x_points, theta)) / den
        new_theta = [
            upper_weights[i] if x_points[i] <= y_l else lower_weights[i]
            for i in range(n)
        ]
        if new_theta == theta:
            break
        theta = new_theta
    else:
        den = sum(theta)
        if den > 0:
            y_l = sum(x * t for x, t in zip(x_points, theta)) / den

    # y_r  –  maximise the centroid
    theta = [(l + u) / 2.0 for l, u in zip(lower_weights, upper_weights)]
    y_r = 0.0
    for _ in range(50):
        den = sum(theta)
        if den == 0:
            y_r = 0.0
            break
        y_r = sum(x * t for x, t in zip(x_points, theta)) / den
        new_theta = [
            lower_weights[i] if x_points[i] <= y_r else upper_weights[i]
            for i in range(n)
        ]
        if new_theta == theta:
            break
        theta = new_theta
    else:
        den = sum(theta)
        if den > 0:
            y_r = sum(x * t for x, t in zip(x_points, theta)) / den

    return y_l, y_r


@lru_cache(maxsize=8)
def _load_it2_membership_functions_from(dir_str: str, spread_pct: float):
    base = Path(dir_str)
    return (
        IT2_mf_7_var(base / "heart_rate_membership_functions.csv", spread_pct),
        IT2_mf_7_var(base / "systolic_blood_pressure_membership_functions.csv", spread_pct),
        IT2_mf_7_var(base / "temperature_membership_functions.csv", spread_pct),
        IT2_mf_7_var(base / "respiratory_rate_membership_functions.csv", spread_pct),
        IT2_mf_3_var_down(base / "oxygen_saturation_membership_functions.csv", spread_pct),
        IT2_mf_3_var_up(base / "inspired_oxygen_concentration_membership_functions.csv", spread_pct),
    )


def load_it2_membership_functions(base_dir: Path, spread_pct: float = 3.0):
    return _load_it2_membership_functions_from(str(base_dir.resolve()), spread_pct)


def firings_it2(
    hr: int, bp: int, temp: float, resp: float, ox: float, insp: float,
    base_dir: Path, spread_pct: float = 3.0,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    models = load_it2_membership_functions(base_dir, spread_pct)
    heart_rate, blood_pressure, temperature, respiratory_rate, oxygen_saturation, inspired_oxygen = models
    return {
        "heart rate": heart_rate(hr),
        "blood pressure": blood_pressure(bp),
        "temperature": temperature(temp),
        "respiratory rate": respiratory_rate(resp),
        "oxygen saturation": oxygen_saturation(ox),
        "inspired oxygen": inspired_oxygen(insp),
    }


def map_to_concern_intervals(
    vital_memberships: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """Map IT2 vital memberships to concern-level firing intervals."""
    mapping: Dict[str, Tuple[float, float]] = {
        "No concern": (0.0, 0.0),
        "Mild concern": (0.0, 0.0),
        "Moderate concern": (0.0, 0.0),
        "Severe concern": (0.0, 0.0),
    }
    for key, (lo, hi) in vital_memberships.items():
        kl = key.lower()
        if "severe" in kl:
            target = "Severe concern"
        elif "moderate" in kl:
            target = "Moderate concern"
        elif "mild" in kl:
            target = "Mild concern"
        elif "no concern" in kl:
            target = "No concern"
        else:
            continue
        cur_lo, cur_hi = mapping[target]
        mapping[target] = (max(cur_lo, lo), max(cur_hi, hi))
    return mapping


def it2_defuzz_vital_centroid(
    concern_intervals: Dict[str, Tuple[float, float]],
) -> Tuple[float, float, float]:
    """KM type-reduce one vital.  Returns (y_l, y_r, defuzzified_score)."""
    MIN_FIRING = 0.05
    intervals: Dict[str, Tuple[float, float]] = {}
    for level, (lo, hi) in concern_intervals.items():
        intervals[level] = (lo if lo >= MIN_FIRING else 0.0, hi if hi >= MIN_FIRING else 0.0)

    has_concern = any(
        level != "No concern" and hi > 0 for level, (_, hi) in intervals.items()
    )
    if not has_concern and intervals.get("No concern", (0, 0))[1] > 0:
        return 0.0, 0.0, 0.0

    _, cache = _it2_output_cache()
    x_points = [i / 100.0 for i in range(301)]
    lower_agg: list[float] = []
    upper_agg: list[float] = []
    for x in x_points:
        out_memberships = cache[x]
        lo_val, hi_val = 0.0, 0.0
        for level, (f_lo, f_hi) in intervals.items():
            if f_hi > 0:
                out_lo, out_hi = out_memberships[level]
                hi_val = max(hi_val, min(f_hi, out_hi))
                lo_val = max(lo_val, min(f_lo, out_lo))
        lower_agg.append(lo_val)
        upper_agg.append(hi_val)

    y_l, y_r = km_type_reduce(x_points, lower_agg, upper_agg)
    return y_l, y_r, (y_l + y_r) / 2.0


def calculate_it2_fuzzy_ews(
    all_firings: Dict[str, Dict[str, Tuple[float, float]]], method: str,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """IT2FLS scoring with per-vital type-reduction intervals."""
    per_vital_scores: Dict[str, float] = {}
    per_vital_intervals: Dict[str, Tuple[float, float]] = {}
    for vital_name, vital_memberships in all_firings.items():
        concern_intervals = map_to_concern_intervals(vital_memberships)
        y_l, y_r, score = it2_defuzz_vital_centroid(concern_intervals)
        per_vital_scores[vital_name] = score
        per_vital_intervals[vital_name] = (y_l, y_r)
    per_vital_scores["total"] = aggregate_total(per_vital_scores, method)
    return per_vital_scores, per_vital_intervals


def it2_interpret_table(
    all_firings: Dict[str, Dict[str, Tuple[float, float]]],
    scores: Dict[str, float],
    intervals: Dict[str, Tuple[float, float]],
    news_scores: Dict[str, int],
    obs: Observation,
) -> pd.DataFrame:
    records = []
    vocab = {
        "heart rate": (obs.hr, "bpm"),
        "blood pressure": (obs.bp, "mmHg"),
        "temperature": (obs.temp, "\u00b0C"),
        "respiratory rate": (obs.resp, "breaths/min"),
        "oxygen saturation": (obs.ox_sats, "%"),
        "inspired oxygen": (obs.insp_ox, "% FiO2"),
    }
    for vital, memberships in all_firings.items():
        upper_vals = {k: v[1] for k, v in memberships.items()}
        label, strength = dominant_label(upper_vals)
        score = scores.get(vital, 0.0)
        y_l, y_r = intervals.get(vital, (0.0, 0.0))
        news_score = news_scores.get(vital, 0)
        value, unit = vocab.get(vital, ("", ""))
        description = (
            f"{vital.title()} sits in '{label}' (UMF={strength:.2f}), "
            f"IT2 interval [{y_l:.2f}, {y_r:.2f}], defuzzified {score:.2f}."
            if strength > 0
            else "No activation for this vital."
        )
        records.append({
            "Vital": vital.title(),
            "Input": f"{value} {unit}".strip(),
            "Dominant concern": label,
            "UMF strength": round(strength, 2),
            "IT2 interval": f"[{y_l:.2f}, {y_r:.2f}]",
            "Per-vital score (0-3)": round(score, 2),
            "NEWS-2 score": news_score,
            "Explanation": description,
        })
    return pd.DataFrame(records)


def it2_membership_chart(model, value: float, unit: str) -> alt.Chart:
    """IT2 membership chart with FOU shown as shaded area between UMF and LMF."""
    keys = list(model.df["Value"].values)
    records = []
    for label, umf_fs, lmf_fs in zip(model.labels, model.umf_fs, model.lmf_fs):
        for k in keys:
            records.append({
                "Value": float(k),
                "Membership": label,
                "UMF": float(umf_fs.get(k, 0.0)),
                "LMF": float(lmf_fs.get(k, 0.0)),
            })
    df = pd.DataFrame(records)
    area = (
        alt.Chart(df)
        .mark_area(opacity=0.25)
        .encode(
            x=alt.X("Value:Q", title=f"Value ({unit})"),
            y="LMF:Q",
            y2="UMF:Q",
            color=alt.Color("Membership:N", legend=alt.Legend(title="Fuzzy set")),
        )
    )
    upper = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x="Value:Q",
            y=alt.Y("UMF:Q", title="Membership"),
            color="Membership:N",
            tooltip=["Membership:N", "Value:Q", "UMF:Q"],
        )
    )
    lower = (
        alt.Chart(df)
        .mark_line(strokeDash=[4, 2])
        .encode(
            x="Value:Q",
            y="LMF:Q",
            color="Membership:N",
            tooltip=["Membership:N", "Value:Q", "LMF:Q"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"Value": [value]}))
        .mark_rule(color="red")
        .encode(x="Value:Q")
    )
    return (area + upper + lower + rule).properties(height=220)


def it2_firing_table_df(
    labels, firing: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Membership": label,
            "Lower firing": round(firing.get(label, (0.0, 0.0))[0], 3),
            "Upper firing": round(firing.get(label, (0.0, 0.0))[1], 3),
        }
        for label in labels
    ])


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------

def _render_t1_tab(prefix: str) -> None:
    """Render the full Type-1 FLS interface inside a tab."""
    selected_dir = DATA_DIR_SIGMOID
    if not selected_dir.exists():
        st.error(f"Required sigmoid membership set not found at {selected_dir}.")
        return
    aggregation_method = "additive"
    st.caption("Configuration fixed: Generated sigmoid membership functions + additive aggregation.")

    use_part2_rules = st.checkbox(
        "Use clinician survey rules (Part 2)",
        value=False,
        help="Overrides the total score with the average clinician rating for matching combinations.",
        key=f"{prefix}_part2",
    )
    part2_k = st.slider(
        "Part 2 blend neighbors (k)",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of nearest survey combinations to blend.",
        disabled=not use_part2_rules,
        key=f"{prefix}_part2_k",
    )

    presets = {
        "Normal": Observation(hr=80, bp=120, temp=36.8, resp=16, ox_sats=98, insp_ox=21),
        "Mild concern": Observation(hr=105, bp=135, temp=37.8, resp=22, ox_sats=94, insp_ox=24),
        "Moderate concern": Observation(hr=120, bp=100, temp=38.5, resp=26, ox_sats=91, insp_ox=30),
        "Severe concern": Observation(hr=135, bp=88, temp=39.2, resp=30, ox_sats=86, insp_ox=60),
    }

    preset_name = st.radio("Quick examples", list(presets.keys()), horizontal=True, key=f"{prefix}_preset")
    default_obs = presets[preset_name]
    st.caption("Select a preset to pre-fill the form; adjust any field before running inference.")

    with st.form(f"{prefix}_inputs"):
        col1, col2, col3 = st.columns(3)
        with col1:
            hr = st.number_input("Heart rate (bpm)", min_value=30, max_value=200, value=default_obs.hr)
            bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=220, value=default_obs.bp)
        with col2:
            temp = st.number_input("Temperature (\u00b0C)", min_value=30.0, max_value=43.0, value=default_obs.temp, step=0.1, format="%.1f")
            resp = st.number_input("Respiratory rate (breaths/min)", min_value=4, max_value=50, value=default_obs.resp)
        with col3:
            ox = st.number_input("Oxygen saturation (%)", min_value=70, max_value=102, value=default_obs.ox_sats)
            insp = st.number_input("Inspired oxygen (% FiO2 or approximated)", min_value=21, max_value=100, value=default_obs.insp_ox)
        submitted = st.form_submit_button("Run inference", use_container_width=True)

    if submitted:
        raw_obs = Observation(hr=int(hr), bp=int(bp), temp=float(temp), resp=int(resp), ox_sats=int(ox), insp_ox=int(insp))
        obs = clamp_observation(raw_obs, selected_dir)
        if obs != raw_obs:
            st.info("Inputs were clamped to the membership function range used in the model.")

        all_firings = firings(obs.hr, obs.bp, obs.temp, obs.resp, obs.ox_sats, obs.insp_ox, selected_dir)
        scores = calculate_fuzzy_ews(all_firings, aggregation_method)
        total = scores.pop("total", 0.0)
        news_scores, news_total = calculate_news2(obs)

        survey_mean = None
        survey_multiplier = None
        survey_details: list = []
        if use_part2_rules:
            survey_mean, survey_details = compute_part2_blend(obs, k=part2_k)
            if survey_mean is not None:
                survey_multiplier = 1.0 + (survey_mean / 15.0)
            else:
                st.warning("Part 2 survey data unavailable for blending.")

        left, right = st.columns([1, 2])
        with left:
            display_total = min(total * survey_multiplier, 18.0) if survey_multiplier is not None else total
            st.metric(
                "Overall score (0-18)",
                f"{display_total:.2f}",
                help="Fuzzy total (optionally scaled by Part 2 survey multiplier).",
            )
            st.metric("Risk bucket", risk_bucket(display_total))
            st.metric(
                "NEWS-2 (0-20)",
                f"{news_total}",
                help="Computed with NEWS-2 Scale 1; supplemental O2 adds 2 if FiO2 > 21%. Consciousness assumed 0.",
            )
            if survey_mean is not None:
                st.metric("Part 2 clinician score (0-15)", f"{survey_mean:.2f}")
                st.metric("Part 2 multiplier", f"{survey_multiplier:.2f}")
                with st.expander("Part 2 nearest combinations", expanded=False):
                    st.dataframe(pd.DataFrame(survey_details), use_container_width=True)
        with right:
            st.write("Per-vital scores")
            combined = []
            for k, v in scores.items():
                combined.append({
                    "Vital": k.title(),
                    "Fuzzy (0-3)": round(v, 2),
                    "NEWS-2": news_scores.get(k, 0),
                })
            st.dataframe(pd.DataFrame(combined).set_index("Vital"))

        st.subheader("Interpretability table")
        st.caption("Dominant membership, strength, and contribution for each vital in plain language.")
        table = interpret_table(all_firings, scores, news_scores, obs)
        st.dataframe(table, use_container_width=True)

        st.subheader("Membership functions and firing")
        st.caption("Fuzzy sets for each vital from the CSVs, your input marked in red, and the firing strengths at that value.")

        mf_models = load_membership_functions(selected_dir)
        mf_lookup = {
            "heart rate": (mf_models[0], obs.hr, "bpm"),
            "blood pressure": (mf_models[1], obs.bp, "mmHg"),
            "temperature": (mf_models[2], obs.temp, "\u00b0C"),
            "respiratory rate": (mf_models[3], obs.resp, "breaths/min"),
            "oxygen saturation": (mf_models[4], obs.ox_sats, "%"),
            "inspired oxygen": (mf_models[5], obs.insp_ox, "% FiO2"),
        }

        for vital, (model, value, unit) in mf_lookup.items():
            with st.expander(f"{vital.title()} ({value} {unit})", expanded=False):
                st.altair_chart(membership_chart(model.df, value, unit), use_container_width=True)
                st.dataframe(
                    firing_table_df(model.labels, all_firings.get(vital, {})),
                    use_container_width=True,
                    height=220,
                )


def _render_it2_tab(prefix: str) -> None:
    """Render the full Interval Type-2 FLS interface inside a tab."""
    selected_dir = DATA_DIR_SIGMOID
    if not selected_dir.exists():
        st.error(f"Required sigmoid membership set not found at {selected_dir}.")
        return
    aggregation_method = "additive"
    st.caption("Configuration fixed: Generated sigmoid membership functions + additive aggregation.")

    use_part2_rules = st.checkbox(
        "Use clinician survey rules (Part 2)",
        value=False,
        help="Overrides the total score with the average clinician rating for matching combinations.",
        key=f"{prefix}_part2",
    )
    part2_k = st.slider(
        "Part 2 blend neighbors (k)",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of nearest survey combinations to blend.",
        disabled=not use_part2_rules,
        key=f"{prefix}_part2_k",
    )

    fou_pct = st.slider(
        "FOU width (% of range)",
        min_value=1.0,
        max_value=15.0,
        value=3.0,
        step=0.5,
        help=(
            "Footprint of Uncertainty width as a percentage of each vital's data range. "
            "The integer spread (in grid steps) is computed per-vital so that "
            "dense grids like temperature get a proportionally smaller spread."
        ),
        key=f"{prefix}_fou",
    )

    presets = {
        "Normal": Observation(hr=80, bp=120, temp=36.8, resp=16, ox_sats=98, insp_ox=21),
        "Mild concern": Observation(hr=105, bp=135, temp=37.8, resp=22, ox_sats=94, insp_ox=24),
        "Moderate concern": Observation(hr=120, bp=100, temp=38.5, resp=26, ox_sats=91, insp_ox=30),
        "Severe concern": Observation(hr=135, bp=88, temp=39.2, resp=30, ox_sats=86, insp_ox=60),
    }

    preset_name = st.radio("Quick examples", list(presets.keys()), horizontal=True, key=f"{prefix}_preset")
    default_obs = presets[preset_name]
    st.caption("Select a preset to pre-fill the form; adjust any field before running inference.")

    with st.form(f"{prefix}_inputs"):
        col1, col2, col3 = st.columns(3)
        with col1:
            hr = st.number_input("Heart rate (bpm)", min_value=30, max_value=200, value=default_obs.hr)
            bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=220, value=default_obs.bp)
        with col2:
            temp = st.number_input("Temperature (\u00b0C)", min_value=30.0, max_value=43.0, value=default_obs.temp, step=0.1, format="%.1f")
            resp = st.number_input("Respiratory rate (breaths/min)", min_value=4, max_value=50, value=default_obs.resp)
        with col3:
            ox = st.number_input("Oxygen saturation (%)", min_value=70, max_value=102, value=default_obs.ox_sats)
            insp = st.number_input("Inspired oxygen (% FiO2 or approximated)", min_value=21, max_value=100, value=default_obs.insp_ox)
        submitted = st.form_submit_button("Run inference", use_container_width=True)

    if submitted:
        raw_obs = Observation(hr=int(hr), bp=int(bp), temp=float(temp), resp=int(resp), ox_sats=int(ox), insp_ox=int(insp))
        obs = clamp_observation(raw_obs, selected_dir)
        if obs != raw_obs:
            st.info("Inputs were clamped to the membership function range used in the model.")

        it2_all = firings_it2(
            obs.hr, obs.bp, obs.temp, obs.resp, obs.ox_sats, obs.insp_ox,
            selected_dir, fou_pct,
        )
        it2_scores, it2_intervals = calculate_it2_fuzzy_ews(it2_all, aggregation_method)
        it2_total = it2_scores.pop("total", 0.0)
        news_scores, news_total = calculate_news2(obs)

        survey_mean = None
        survey_multiplier = None
        survey_details: list = []
        if use_part2_rules:
            survey_mean, survey_details = compute_part2_blend(obs, k=part2_k)
            if survey_mean is not None:
                survey_multiplier = 1.0 + (survey_mean / 15.0)
            else:
                st.warning("Part 2 survey data unavailable for blending.")

        left, right = st.columns([1, 2])
        with left:
            it2_display = min(it2_total * survey_multiplier, 18.0) if survey_multiplier is not None else it2_total
            st.metric(
                "Overall score (0-18)",
                f"{it2_display:.2f}",
                help="IT2FLS total (optionally scaled by Part 2 survey multiplier).",
            )
            st.metric("Risk bucket", risk_bucket(it2_display))
            st.metric(
                "NEWS-2 (0-20)",
                f"{news_total}",
                help="Computed with NEWS-2 Scale 1; supplemental O2 adds 2 if FiO2 > 21%. Consciousness assumed 0.",
            )
            if survey_mean is not None:
                st.metric("Part 2 clinician score (0-15)", f"{survey_mean:.2f}")
                st.metric("Part 2 multiplier", f"{survey_multiplier:.2f}")
                with st.expander("Part 2 nearest combinations", expanded=False):
                    st.dataframe(pd.DataFrame(survey_details), use_container_width=True)
        with right:
            st.write("Per-vital scores")
            combined = []
            for k, v in it2_scores.items():
                y_l, y_r = it2_intervals.get(k, (0.0, 0.0))
                combined.append({
                    "Vital": k.title(),
                    "IT2 Fuzzy (0-3)": round(v, 2),
                    "Interval": f"[{y_l:.2f}, {y_r:.2f}]",
                    "NEWS-2": news_scores.get(k, 0),
                })
            st.dataframe(pd.DataFrame(combined).set_index("Vital"))

        st.subheader("Interpretability table")
        st.caption("Dominant membership (UMF), type-reduced interval, and contribution for each vital.")
        it2_table = it2_interpret_table(it2_all, it2_scores, it2_intervals, news_scores, obs)
        st.dataframe(it2_table, use_container_width=True)

        st.subheader("Membership functions and firing (IT2)")
        st.caption(
            "Shaded regions show the Footprint of Uncertainty (FOU). "
            "Solid lines = UMF, dashed lines = LMF. Input marked in red."
        )

        it2_models = load_it2_membership_functions(selected_dir, fou_pct)
        it2_mf_lookup = {
            "heart rate": (it2_models[0], obs.hr, "bpm"),
            "blood pressure": (it2_models[1], obs.bp, "mmHg"),
            "temperature": (it2_models[2], obs.temp, "\u00b0C"),
            "respiratory rate": (it2_models[3], obs.resp, "breaths/min"),
            "oxygen saturation": (it2_models[4], obs.ox_sats, "%"),
            "inspired oxygen": (it2_models[5], obs.insp_ox, "% FiO2"),
        }

        for vital, (model, value, unit) in it2_mf_lookup.items():
            with st.expander(f"{vital.title()} ({value} {unit})", expanded=False):
                st.altair_chart(it2_membership_chart(model, value, unit), use_container_width=True)
                st.dataframe(
                    it2_firing_table_df(model.labels, it2_all.get(vital, {})),
                    use_container_width=True,
                    height=220,
                )


def _temporal_delta_memberships(delta: float) -> Dict[str, float]:
    """Fuzzify snapshot score delta into 5 linguistic terms.

    Input: delta = snapshot_now - snapshot_prev.
    Positive = worsening, negative = improving.

    MF shapes (triangular / trapezoidal shoulders):
      Big improvement:     full at d <= -3, ramp to 0 at d = -1
      Small improvement:   triangle(-3, -1, 0)
      Stable:              triangle(-1, 0, +1)
      Small deterioration: triangle(0, +1, +3)
      Big deterioration:   0 at d <= +1, ramp to full at d = +3
    """
    d = float(delta)
    big_improve = max(0.0, min(1.0, (-1.0 - d) / 2.0))
    small_improve = max(0.0, min((d + 3.0) / 2.0, -d / 1.0))
    stable = max(0.0, min((d + 1.0) / 1.0, (1.0 - d) / 1.0))
    small_deter = max(0.0, min(d / 1.0, (3.0 - d) / 2.0))
    big_deter = max(0.0, min(1.0, (d - 1.0) / 2.0))
    return {
        "Big improvement": big_improve,
        "Small improvement": small_improve,
        "Stable": stable,
        "Small deterioration": small_deter,
        "Big deterioration": big_deter,
    }


def _temporal_fuzzy_adjustment(delta: float) -> float:
    """5-rule fuzzy inference: map snapshot score delta to a temporal adjustment.

    Rules:
      IF delta is Big improvement     THEN adjustment is -2.0
      IF delta is Small improvement   THEN adjustment is -0.75
      IF delta is Stable              THEN adjustment is  0.0
      IF delta is Small deterioration THEN adjustment is +0.75
      IF delta is Big deterioration   THEN adjustment is +2.0

    Defuzzified via weighted-average of singleton consequents.
    """
    mf = _temporal_delta_memberships(delta)
    consequents = {
        "Big improvement": -2.0,
        "Small improvement": -0.75,
        "Stable": 0.0,
        "Small deterioration": 0.75,
        "Big deterioration": 2.0,
    }
    num = sum(mf[k] * consequents[k] for k in mf)
    den = sum(mf.values())
    if den == 0.0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Per-vital temporal trend nudge (Part 3 survey-driven)
# ---------------------------------------------------------------------------

# Part 3 scenario-to-vital mapping.
# Each scenario presented 5 readings over 72 h with ONE vital bolded as the
# focus.  Three trend types per vital (except temperature which has two):
#   "stable"       – vital stays at an abnormal level throughout
#   "deteriorating" – vital trends from normal to abnormal
#   "recovering"   – vital spikes to abnormal then returns toward normal
#
# Sc 1  HR  stable elevated   | Sc  8  RR  deteriorating
# Sc 2  HR  deteriorating     | Sc  9  RR  spike & recover
# Sc 3  HR  spike & recover   | Sc 10  Temp deteriorating
# Sc 4  BP  stable low        | Sc 11  Temp spike & recover
# Sc 5  BP  deteriorating     | Sc 12  SpO2 stable low
# Sc 6  BP  dip & recover     | Sc 13  SpO2 deteriorating
# Sc 7  RR  stable elevated   | Sc 14  SpO2 dip & recover

PART3_VITAL_SCENARIOS: Dict[str, Dict[str, int]] = {
    "heart rate":        {"stable": 1,  "deteriorating": 2,  "recovering": 3},
    "blood pressure":    {"stable": 4,  "deteriorating": 5,  "recovering": 6},
    "respiratory rate":  {"stable": 7,  "deteriorating": 8,  "recovering": 9},
    "temperature":       {              "deteriorating": 10, "recovering": 11},
    "oxygen saturation": {"stable": 12, "deteriorating": 13, "recovering": 14},
}
PART3_MAX_CONCERN = 15.0

# Per-vital score keys stored in each timeline entry
_PV_KEYS: Dict[str, str] = {
    "heart rate": "pv_heart_rate",
    "blood pressure": "pv_blood_pressure",
    "temperature": "pv_temperature",
    "respiratory rate": "pv_respiratory_rate",
    "oxygen saturation": "pv_oxygen_saturation",
    "inspired oxygen": "pv_inspired_oxygen",
}


@lru_cache(maxsize=1)
def _load_part3_mean_by_id() -> Dict[int, float]:
    """Return mean clinician concern per Part 3 scenario ID (0-15 scale)."""
    if not PART3_SURVEY_PATH.exists():
        return {}
    df = pd.read_excel(PART3_SURVEY_PATH)
    if "part3Data" not in df.columns:
        return {}
    scores_by_id: Dict[int, list] = {}
    for raw in df["part3Data"].dropna():
        try:
            entries = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            try:
                sid = int(entry.get("id"))
                concern = float(entry.get("concernLevel"))
            except Exception:
                continue
            scores_by_id.setdefault(sid, []).append(concern)
    return {
        sid: sum(vals) / len(vals)
        for sid, vals in scores_by_id.items()
        if vals
    }


def load_part3_concern_weights() -> Dict[str, Dict[str, float]]:
    """Per-vital, per-trend-type concern weights from Part 3 survey.

    Returns e.g.:
      {"heart rate": {"stable": 2.72, "deteriorating": 5.93, "recovering": 4.72},
       "blood pressure": {"stable": 3.12, "deteriorating": 8.90, "recovering": 6.25},
       ...}
    Inspired oxygen is not covered in Part 3 and will be absent.
    """
    mean_by_id = _load_part3_mean_by_id()
    if not mean_by_id:
        return {}
    weights: Dict[str, Dict[str, float]] = {}
    for vital, trend_ids in PART3_VITAL_SCENARIOS.items():
        vw: Dict[str, float] = {}
        for trend_type, sid in trend_ids.items():
            if sid in mean_by_id:
                vw[trend_type] = mean_by_id[sid]
        if vw:
            weights[vital] = vw
    return weights


@dataclass(frozen=True)
class TemporalConfig:
    window_hours: float = 24.0
    min_observations: int = 3
    min_gap_minutes: float = 30.0
    ewma_alpha: float = 0.3
    slope_stable_threshold: float = 0.02


def _ewma(values: list, alpha: float) -> list:
    """Exponentially weighted moving average."""
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return result


def _linear_slope(times_hours: list, values: list) -> float:
    """Ordinary least-squares slope (value change per hour)."""
    n = len(times_hours)
    if n < 2:
        return 0.0
    mean_t = sum(times_hours) / n
    mean_v = sum(values) / n
    ss_tt = sum((t - mean_t) ** 2 for t in times_hours)
    if ss_tt == 0:
        return 0.0
    ss_tv = sum((t - mean_t) * (v - mean_v) for t, v in zip(times_hours, values))
    return ss_tv / ss_tt


def compute_per_vital_trends(
    timeline: list,
    config: TemporalConfig,
) -> Dict[str, Dict]:
    """Compute per-vital trend over the lookback window.

    For each vital returns:
      trend        – 'deteriorating' | 'stable' | 'improving' | 'insufficient_data'
      slope        – raw slope (score units / hour)
      trend_factor – normalised to [-1, +1]  (+1 = strongly worsening)
      n_valid      – observations used
      ewma_scores  – EWMA-smoothed per-vital scores used
    """
    if len(timeline) < 2:
        return {v: {"trend": "insufficient_data", "slope": 0.0,
                     "trend_factor": 0.0, "n_valid": len(timeline),
                     "ewma_scores": []} for v in _PV_KEYS}

    latest_t = float(timeline[-1]["t_minutes"])
    window_min = config.window_hours * 60.0

    results: Dict[str, Dict] = {}
    for vital, pv_key in _PV_KEYS.items():
        valid: list[Tuple[float, float]] = []
        for entry in timeline:
            t = float(entry["t_minutes"])
            if latest_t - t > window_min:
                continue
            if pv_key not in entry:
                continue
            if valid and (t - valid[-1][0]) < config.min_gap_minutes:
                continue
            valid.append((t, float(entry[pv_key])))

        if len(valid) < config.min_observations:
            results[vital] = {
                "trend": "insufficient_data",
                "slope": 0.0,
                "trend_factor": 0.0,
                "n_valid": len(valid),
                "ewma_scores": [],
            }
            continue

        times_min = [v[0] for v in valid]
        scores = [v[1] for v in valid]
        smoothed = _ewma(scores, config.ewma_alpha)

        t0 = times_min[0]
        times_h = [(t - t0) / 60.0 for t in times_min]
        slope = _linear_slope(times_h, smoothed)

        max_slope = 3.0 / max(config.window_hours, 1.0)
        trend_factor = max(-1.0, min(1.0, slope / max_slope))

        if abs(trend_factor) < config.slope_stable_threshold:
            trend = "stable"
        elif trend_factor > 0:
            trend = "deteriorating"
        else:
            trend = "improving"

        results[vital] = {
            "trend": trend,
            "slope": round(slope, 4),
            "trend_factor": round(trend_factor, 4),
            "n_valid": len(valid),
            "ewma_scores": [round(s, 3) for s in smoothed],
        }
    return results


def _select_concern(
    vital: str,
    trend_label: str,
    concern_weights: Dict[str, Dict[str, float]],
) -> float:
    """Pick the trend-specific Part 3 concern weight for a vital.

    Maps the detected trend classification to the matching scenario type:
      deteriorating  -> "deteriorating" scenario concern
      improving      -> "recovering" scenario concern
      stable         -> "stable" scenario concern (if available, else mean of others)
      insufficient   -> 0 (no nudge)

    Vitals absent from Part 3 (e.g. inspired oxygen) return 0.
    """
    vw = concern_weights.get(vital, {})
    if not vw:
        return 0.0
    if trend_label == "deteriorating":
        return vw.get("deteriorating", 0.0)
    if trend_label == "improving":
        return vw.get("recovering", 0.0)
    if trend_label == "stable":
        if "stable" in vw:
            return vw["stable"]
        return sum(vw.values()) / len(vw) if vw else 0.0
    return 0.0


def apply_temporal_nudges(
    per_vital_scores: Dict[str, float],
    trends: Dict[str, Dict],
    concern_weights: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict], float]:
    """Apply per-vital temporal nudges and return adjusted scores + new total.

    The concern weight used depends on the detected trend:
      deteriorating -> Part 3 "deteriorating" scenario concern
      improving     -> Part 3 "recovering" scenario concern
      stable        -> Part 3 "stable" scenario concern

    Nudge formula:
      concern_pct  = selected concern weight / 15
      nudge        = trend_factor x concern_pct x current_score
      adjusted     = clamp(score + nudge, 0, 3)

    When the current score is 0 but the vital is deteriorating, a small floor
    (0.1) is used so the nudge can still inject a non-zero warning.
    """
    results: Dict[str, Dict] = {}
    adjusted_total = 0.0

    for vital, score in per_vital_scores.items():
        if vital == "total":
            continue
        trend_info = trends.get(vital, {})
        trend_factor = trend_info.get("trend_factor", 0.0)
        trend_label = trend_info.get("trend", "insufficient_data")

        concern = _select_concern(vital, trend_label, concern_weights)
        concern_pct = concern / PART3_MAX_CONCERN

        effective_score = max(score, 0.1) if trend_factor > 0 and score == 0.0 else score
        nudge = trend_factor * concern_pct * effective_score
        adjusted = max(0.0, min(3.0, score + nudge))
        adjusted_total += adjusted

        results[vital] = {
            "original_score": round(score, 3),
            "nudge": round(nudge, 3),
            "adjusted_score": round(adjusted, 3),
            "nudge_pct": round((nudge / score * 100) if score > 0 else 0.0, 1),
            "concern_weight": round(concern, 2),
            "concern_pct": round(concern_pct * 100, 1),
            "trend": trend_label,
            "trend_factor": round(trend_factor, 4),
        }

    return results, round(adjusted_total, 2)


def _render_temporal_tab(prefix: str) -> None:
    """Collect multiple observations over time and compute per-vital temporal nudges."""
    st.caption(
        "Add sequential observations at chosen intervals. Per-vital trends are "
        "computed over a configurable window with EWMA smoothing, and each vital's "
        "fuzzy score is nudged based on trend direction and Part 3 clinician concern weights."
    )

    selected_dir = DATA_DIR_SIGMOID
    if not selected_dir.exists():
        st.error(f"Required sigmoid membership set not found at {selected_dir}.")
        return
    aggregation_method = "additive"
    st.caption("Configuration fixed: Generated sigmoid membership functions + additive aggregation.")

    presets = {
        "Normal": Observation(hr=80, bp=120, temp=36.8, resp=16, ox_sats=98, insp_ox=21),
        "Mild concern": Observation(hr=105, bp=135, temp=37.8, resp=22, ox_sats=94, insp_ox=24),
        "Moderate concern": Observation(hr=120, bp=100, temp=38.5, resp=26, ox_sats=91, insp_ox=30),
        "Severe concern": Observation(hr=135, bp=88, temp=39.2, resp=30, ox_sats=86, insp_ox=60),
    }

    timeline_key = f"{prefix}_timeline"
    if timeline_key not in st.session_state:
        st.session_state[timeline_key] = []

    st.subheader("Add observation")
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        interval_unit = st.selectbox(
            "Interval unit",
            ["minutes", "hours"],
            index=1,
            key=f"{prefix}_unit",
        )
    with col_cfg2:
        interval_value = st.number_input(
            "Interval size",
            min_value=1,
            max_value=240,
            value=1,
            key=f"{prefix}_interval",
        )

    preset_name = st.radio(
        "Quick examples",
        list(presets.keys()),
        horizontal=True,
        key=f"{prefix}_preset",
    )
    default_obs = presets[preset_name]

    # Keep temporal input widgets in sync when user switches presets.
    hr_key = f"{prefix}_hr"
    bp_key = f"{prefix}_bp"
    temp_key = f"{prefix}_temp"
    resp_key = f"{prefix}_resp"
    ox_key = f"{prefix}_ox"
    insp_key = f"{prefix}_insp"
    last_preset_key = f"{prefix}_last_preset"

    if st.session_state.get(last_preset_key) != preset_name:
        st.session_state[hr_key] = int(default_obs.hr)
        st.session_state[bp_key] = int(default_obs.bp)
        st.session_state[temp_key] = float(default_obs.temp)
        st.session_state[resp_key] = int(default_obs.resp)
        st.session_state[ox_key] = int(default_obs.ox_sats)
        st.session_state[insp_key] = int(default_obs.insp_ox)
        st.session_state[last_preset_key] = preset_name

    with st.form(f"{prefix}_add_observation"):
        col1, col2, col3 = st.columns(3)
        with col1:
            hr = st.number_input(
                "Heart rate (bpm)",
                min_value=30,
                max_value=200,
                value=int(st.session_state.get(hr_key, default_obs.hr)),
                key=hr_key,
            )
            bp = st.number_input(
                "Systolic BP (mmHg)",
                min_value=50,
                max_value=220,
                value=int(st.session_state.get(bp_key, default_obs.bp)),
                key=bp_key,
            )
        with col2:
            temp = st.number_input(
                "Temperature (°C)",
                min_value=30.0,
                max_value=43.0,
                value=float(st.session_state.get(temp_key, default_obs.temp)),
                step=0.1,
                format="%.1f",
                key=temp_key,
            )
            resp = st.number_input(
                "Respiratory rate (breaths/min)",
                min_value=4,
                max_value=50,
                value=int(st.session_state.get(resp_key, default_obs.resp)),
                key=resp_key,
            )
        with col3:
            ox = st.number_input(
                "Oxygen saturation (%)",
                min_value=70,
                max_value=102,
                value=int(st.session_state.get(ox_key, default_obs.ox_sats)),
                key=ox_key,
            )
            insp = st.number_input(
                "Inspired oxygen (% FiO2 or approximated)",
                min_value=21,
                max_value=100,
                value=int(st.session_state.get(insp_key, default_obs.insp_ox)),
                key=insp_key,
            )

        add_clicked = st.form_submit_button("Add observation", use_container_width=True)

    if add_clicked:
        raw_obs = Observation(hr=int(hr), bp=int(bp), temp=float(temp), resp=int(resp), ox_sats=int(ox), insp_ox=int(insp))
        clamped = clamp_observation(raw_obs, selected_dir)
        all_f = firings(
            clamped.hr, clamped.bp, clamped.temp, clamped.resp,
            clamped.ox_sats, clamped.insp_ox, selected_dir,
        )
        pv_scores = calculate_fuzzy_ews(all_f, aggregation_method)
        fuzzy_total = float(pv_scores.pop("total", 0.0))
        _, news_total = calculate_news2(raw_obs)

        unit_to_minutes = 60 if interval_unit == "hours" else 1
        step_minutes = int(interval_value) * unit_to_minutes
        sequence = st.session_state[timeline_key]
        next_t = 0 if not sequence else int(sequence[-1]["t_minutes"]) + step_minutes

        entry: dict = {
            "idx": len(sequence) + 1,
            "t_minutes": next_t,
            "interval_unit": interval_unit,
            "interval_value": int(interval_value),
            "heart_rate": int(raw_obs.hr),
            "blood_pressure": int(raw_obs.bp),
            "temperature": float(raw_obs.temp),
            "respiratory_rate": int(raw_obs.resp),
            "oxygen_saturation": int(raw_obs.ox_sats),
            "inspired_oxygen": int(raw_obs.insp_ox),
            "model_heart_rate": int(clamped.hr),
            "model_blood_pressure": int(clamped.bp),
            "model_temperature": float(clamped.temp),
            "model_respiratory_rate": int(clamped.resp),
            "model_oxygen_saturation": int(clamped.ox_sats),
            "model_inspired_oxygen": int(clamped.insp_ox),
            "snapshot_fuzzy_ews": round(fuzzy_total, 2),
            "snapshot_news2": int(news_total),
        }
        for vital, pv_key in _PV_KEYS.items():
            entry[pv_key] = round(pv_scores.get(vital, 0.0), 3)
        sequence.append(entry)
        if clamped != raw_obs:
            st.info("Observation clamped to selected membership function range.")

    controls = st.columns(2)
    with controls[0]:
        if st.button("Remove last observation", use_container_width=True, key=f"{prefix}_remove_last"):
            if st.session_state[timeline_key]:
                st.session_state[timeline_key].pop()
    with controls[1]:
        if st.button("Clear timeline", use_container_width=True, key=f"{prefix}_clear_all"):
            st.session_state[timeline_key] = []

    timeline = st.session_state[timeline_key]
    if not timeline:
        st.info("No temporal observations yet. Add at least one to build a timeline.")
        return

    # ------------------------------------------------------------------
    # Backfill legacy entries missing per-vital or snapshot scores
    # ------------------------------------------------------------------
    for entry in timeline:
        needs_pv = any(pv_key not in entry for pv_key in _PV_KEYS.values())
        needs_snap = "snapshot_fuzzy_ews" not in entry or "snapshot_news2" not in entry
        if not needs_pv and not needs_snap:
            continue
        obs_raw = Observation(
            hr=int(entry["heart_rate"]),
            bp=int(entry["blood_pressure"]),
            temp=float(entry["temperature"]),
            resp=float(entry["respiratory_rate"]),
            ox_sats=float(entry["oxygen_saturation"]),
            insp_ox=float(entry["inspired_oxygen"]),
        )
        obs_model = clamp_observation(obs_raw, selected_dir)
        all_f = firings(
            obs_model.hr, obs_model.bp, obs_model.temp, obs_model.resp,
            obs_model.ox_sats, obs_model.insp_ox, selected_dir,
        )
        pv_scores = calculate_fuzzy_ews(all_f, aggregation_method)
        if needs_snap:
            entry["snapshot_fuzzy_ews"] = round(float(pv_scores.get("total", 0.0)), 2)
            _, news_total = calculate_news2(obs_raw)
            entry["snapshot_news2"] = int(news_total)
        if needs_pv:
            for vital, pv_key in _PV_KEYS.items():
                if pv_key not in entry:
                    entry[pv_key] = round(pv_scores.get(vital, 0.0), 3)

    # ------------------------------------------------------------------
    # Temporal trend configuration
    # ------------------------------------------------------------------
    with st.expander("Temporal trend settings", expanded=False):
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            cfg_window = st.number_input(
                "Look-back window (hours)", min_value=1.0, max_value=72.0,
                value=24.0, step=1.0, key=f"{prefix}_cfg_window",
            )
        with tc2:
            cfg_min_obs = st.number_input(
                "Min observations for trend", min_value=2, max_value=20,
                value=3, key=f"{prefix}_cfg_min_obs",
            )
        with tc3:
            cfg_min_gap = st.number_input(
                "Min gap between obs (min)", min_value=0.0, max_value=120.0,
                value=30.0, step=5.0, key=f"{prefix}_cfg_min_gap",
            )
        with tc4:
            cfg_alpha = st.slider(
                "EWMA smoothing α", min_value=0.1, max_value=1.0,
                value=0.3, step=0.05, key=f"{prefix}_cfg_alpha",
                help="Higher = more weight on recent observations.",
            )
    t_config = TemporalConfig(
        window_hours=cfg_window,
        min_observations=cfg_min_obs,
        min_gap_minutes=cfg_min_gap,
        ewma_alpha=cfg_alpha,
    )

    # ------------------------------------------------------------------
    # Load Part 3 concern weights (per-vital, per-trend-type)
    # ------------------------------------------------------------------
    concern_weights = load_part3_concern_weights()
    if not concern_weights:
        st.warning(
            "Part 3 survey data not found - using equal concern weights for all vitals. "
            f"Expected at: {PART3_SURVEY_PATH}"
        )
        default_w = PART3_MAX_CONCERN / 2.0
        concern_weights = {
            v: {"stable": default_w, "deteriorating": default_w, "recovering": default_w}
            for v in _PV_KEYS
        }

    # ------------------------------------------------------------------
    # Compute per-vital trends
    # ------------------------------------------------------------------
    trends = compute_per_vital_trends(timeline, t_config)

    # Latest per-vital snapshot scores
    latest_entry = timeline[-1]
    latest_pv: Dict[str, float] = {}
    for vital, pv_key in _PV_KEYS.items():
        latest_pv[vital] = float(latest_entry.get(pv_key, 0.0))

    # ------------------------------------------------------------------
    # Apply nudges
    # ------------------------------------------------------------------
    nudge_results, adjusted_total = apply_temporal_nudges(
        latest_pv, trends, concern_weights,
    )

    # ------------------------------------------------------------------
    # Headline metrics
    # ------------------------------------------------------------------
    raw_total = float(latest_entry.get("snapshot_fuzzy_ews", 0.0))
    st.subheader("Temporal-adjusted scores")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Snapshot Fuzzy EWS (0-18)",
            f"{raw_total:.2f}",
            help="Sum of per-vital fuzzy scores at the latest observation (no temporal adjustment).",
        )
    with m2:
        st.metric(
            "Trend-adjusted Fuzzy EWS (0-18)",
            f"{min(adjusted_total, 18.0):.2f}",
            delta=f"{adjusted_total - raw_total:+.2f}",
            help="Per-vital scores nudged by trend direction and Part 3 clinician concern weights, then summed.",
        )
    with m3:
        st.metric("Risk bucket (adjusted)", risk_bucket(min(adjusted_total, 18.0)))

    # ------------------------------------------------------------------
    # Per-vital nudge detail table
    # ------------------------------------------------------------------
    st.subheader("Per-vital trend nudges")
    st.caption(
        "Each vital's 0-3 fuzzy score is shifted by: trend_factor × (Part 3 concern / 15) × score. "
        "Trend computed over the look-back window using EWMA-smoothed scores and linear regression."
    )
    nudge_rows = []
    trend_symbols = {
        "deteriorating": "▲ Deteriorating",
        "stable": "— Stable",
        "improving": "▼ Improving",
        "insufficient_data": "⋯ Insufficient data",
    }
    for vital in _PV_KEYS:
        info = nudge_results.get(vital, {})
        t_info = trends.get(vital, {})
        nudge_rows.append({
            "Vital": vital.title(),
            "Score (0-3)": info.get("original_score", 0.0),
            "Trend": trend_symbols.get(info.get("trend", ""), "?"),
            "Trend factor": info.get("trend_factor", 0.0),
            "Concern wt": info.get("concern_weight", 0.0),
            "Concern %": info.get("concern_pct", 0.0),
            "Nudge": info.get("nudge", 0.0),
            "Nudge %": info.get("nudge_pct", 0.0),
            "Adjusted (0-3)": info.get("adjusted_score", 0.0),
            "Obs used": t_info.get("n_valid", 0),
        })
    st.dataframe(pd.DataFrame(nudge_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Part 3 concern weights reference (per-vital, per-trend-type)
    # ------------------------------------------------------------------
    with st.expander("Part 3 concern weights (survey-derived)", expanded=False):
        cw_rows = []
        for v, trend_dict in concern_weights.items():
            row: dict = {"Vital": v.title()}
            for tt in ("stable", "deteriorating", "recovering"):
                val = trend_dict.get(tt)
                row[tt.title()] = f"{val:.2f}" if val is not None else "-"
            cw_rows.append(row)
        st.dataframe(pd.DataFrame(cw_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Scenario-to-vital mapping: "
            + ", ".join(
                f"{v.title()}: {', '.join(f'{t}=Sc{sid}' for t, sid in ids.items())}"
                for v, ids in PART3_VITAL_SCENARIOS.items()
            )
        )

    # ------------------------------------------------------------------
    # Temporal snapshots table (compact)
    # ------------------------------------------------------------------
    df = pd.DataFrame(timeline)
    df = df.sort_values("idx").reset_index(drop=True)

    pv_display_cols = list(_PV_KEYS.values())
    base_cols = ["idx", "t_minutes", "heart_rate", "blood_pressure", "temperature",
                 "respiratory_rate", "oxygen_saturation", "inspired_oxygen",
                 "snapshot_fuzzy_ews", "snapshot_news2"]
    available_cols = [c for c in base_cols + pv_display_cols if c in df.columns]
    compact_df = df[available_cols].copy()
    compact_df = compact_df.rename(columns={
        "idx": "Obs #", "t_minutes": "Time (min)",
        "heart_rate": "HR", "blood_pressure": "BP", "temperature": "Temp",
        "respiratory_rate": "Resp", "oxygen_saturation": "SpO2",
        "inspired_oxygen": "FiO2",
        "snapshot_fuzzy_ews": "Fuzzy EWS", "snapshot_news2": "NEWS-2",
        "pv_heart_rate": "PV HR", "pv_blood_pressure": "PV BP",
        "pv_temperature": "PV Temp", "pv_respiratory_rate": "PV Resp",
        "pv_oxygen_saturation": "PV SpO2", "pv_inspired_oxygen": "PV FiO2",
    })

    st.subheader("Temporal snapshots")
    st.dataframe(compact_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Legacy 5-rule overall delta adjustment (kept for comparison)
    # ------------------------------------------------------------------
    df["snapshot_delta"] = df["snapshot_fuzzy_ews"].astype(float).diff()
    df["temporal_adjustment"] = df["snapshot_delta"].apply(
        lambda d: _temporal_fuzzy_adjustment(d) if pd.notna(d) else 0.0
    )

    trend_summary = df[
        [c for c in [
            "idx", "t_minutes", "heart_rate", "blood_pressure", "temperature",
            "respiratory_rate", "oxygen_saturation", "inspired_oxygen",
            "snapshot_fuzzy_ews", "snapshot_news2", "snapshot_delta", "temporal_adjustment",
        ] if c in df.columns]
    ].copy()
    trend_summary = trend_summary.sort_values("idx").reset_index(drop=True)
    trend_summary["delta_news2"] = trend_summary["snapshot_news2"].diff()
    trend_summary["dt_minutes"] = trend_summary["t_minutes"].diff()
    trend_summary["fuzzy_rate_per_hour"] = trend_summary["snapshot_delta"] / (trend_summary["dt_minutes"] / 60.0)
    trend_summary = trend_summary.rename(columns={
        "idx": "Obs #", "t_minutes": "Time (min)",
        "heart_rate": "HR", "blood_pressure": "BP", "temperature": "Temp",
        "respiratory_rate": "Resp", "oxygen_saturation": "SpO2",
        "inspired_oxygen": "FiO2",
        "snapshot_fuzzy_ews": "Snapshot Fuzzy EWS", "snapshot_news2": "Snapshot NEWS-2",
        "snapshot_delta": "Score Δ", "temporal_adjustment": "Legacy adj.",
        "delta_news2": "ΔNEWS-2", "fuzzy_rate_per_hour": "Fuzzy EWS Δ/hour",
    })
    for col in ["Temp", "Snapshot Fuzzy EWS", "Score Δ", "Legacy adj.", "Fuzzy EWS Δ/hour"]:
        if col in trend_summary.columns:
            trend_summary[col] = pd.to_numeric(trend_summary[col], errors="coerce").round(2)

    with st.expander("Trend summary (legacy 5-rule delta system)", expanded=False):
        st.caption(
            "Score Δ = change in total Fuzzy EWS between consecutive snapshots. "
            "Legacy adj. = output of the original 5-rule fuzzy system applied to that delta."
        )
        display_cols = [c for c in [
            "Obs #", "Time (min)", "HR", "BP", "Temp", "Resp", "SpO2", "FiO2",
            "Snapshot Fuzzy EWS", "Snapshot NEWS-2", "Score Δ", "Legacy adj.",
            "ΔNEWS-2", "Fuzzy EWS Δ/hour",
        ] if c in trend_summary.columns]
        st.dataframe(trend_summary[display_cols], use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download timeline CSV",
        data=csv_bytes,
        file_name="temporal_observations.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{prefix}_download_csv",
    )


def main() -> None:
    if not has_streamlit_context():
        print("This file is meant to run with: streamlit run streamlit_app.py")
        return

    st.set_page_config(page_title="Fuzzy EWS", page_icon="\U0001fa7a", layout="wide")
    st.title("Fuzzy Early Warning Score (EWS)")

    tab_t1, tab_it2, tab_temporal = st.tabs(
        ["Type-1 FLS", "Interval Type-2 FLS", "Temporal Context Builder"]
    )

    with tab_t1:
        _render_t1_tab("t1")

    with tab_it2:
        _render_it2_tab("it2")

    with tab_temporal:
        _render_temporal_tab("temporal")


try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # Streamlit < 1.27 fallback
    get_script_run_ctx = lambda: None  # type: ignore


def has_streamlit_context() -> bool:
    """Return True when running under `streamlit run`, False in bare Python."""
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    # Avoid ScriptRunContext warnings when run via `python streamlit_app.py`.
    if not has_streamlit_context():
        print("This file is meant to run with: streamlit run streamlit_app.py")
    else:
        main()
