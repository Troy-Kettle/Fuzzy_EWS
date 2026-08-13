import altair as alt
import math
import pandas as pd
import streamlit as st
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_SIGMOID = REPO_ROOT / "membership_functions" / "sigmoid"   # the only set used

AVPU_OPTIONS = [
    "Alert",
    "Responds to voice",
    "Newly confused / agitated",
    "Responds to pain",
    "Unresponsive",
]
# Retained for reference only — ACVPU is no longer given a fuzzy concern score.
# The engine's canonical map is engine_scoring.ACVPU_MAP.
AVPU_FUZZY_SCORE = {
    "Alert": 0.0,
    "Responds to voice": 1.0,
    "Newly confused / agitated": 2.0,
    "Responds to pain": 3.0,
    "Unresponsive": 3.0,
}
# All zero: the NEWS-2 baseline gets no consciousness sub-score here either. ACVPU is
# out-of-band for EVERY system — a non-Alert reading is a positive deterioration flag, not
# points in any total. Real NEWS-2 awards 3, but including it would give NEWS-2 a
# consciousness input the fuzzy scores lack and make the comparison unmatched.
# Matches engine_scoring.news2_consciousness_score. See ACVPU_BONUS below.
AVPU_NEWS2_SCORE = {
    "Alert": 0,
    "Responds to voice": 0,
    "Newly confused / agitated": 0,
    "Responds to pain": 0,
    "Unresponsive": 0,
}

# Any ACVPU reading other than Alert is an automatic deterioration signal: flagged
# as positive regardless of the aggregated score, AND a flat bonus added to the
# fuzzy total on top of the vital's own defuzzified contribution.
# ACVPU contributes NOTHING to the fuzzy score: it is not a scored vital and there is no
# bonus. Any reading other than Alert makes the whole row a positive flag of
# deterioration on its own. Kept at 0.0 (not deleted) to match engine_scoring.ACVPU_BONUS
# and so any reintroduced reference is inert. This applies to EVERY system including
# the NEWS-2 baseline, whose consciousness sub-score is also 0 (AVPU_NEWS2_SCORE).
ACVPU_BONUS = 0.0

# Inspired oxygen is recorded in ONE of two units, exactly as the source datasets do
# (INSPIRED_O2 + INSPIRED_O2_UNITS = "%" / "litres"). The two are never
# interconverted — each is scored on its own membership function. A pseudo-FiO2
# estimate (the old "21 + 4×L/min" rule of thumb) is deliberately NOT offered.
O2_UNIT_PCT  = "% FiO2"
O2_UNIT_LMIN = "L/min supplementary flow"
O2_UNITS = [O2_UNIT_PCT, O2_UNIT_LMIN]
# widget bounds per unit: (min, max, room-air/default value)
O2_INPUT_RANGE = {O2_UNIT_PCT: (21, 100, 21), O2_UNIT_LMIN: (0, 15, 2)}


def acvpu_deterioration_flag(avpu: str) -> bool:
    """True for any ACVPU value other than Alert — independent of the aggregated
    fuzzy score, not a substitute for it."""
    return avpu != "Alert"


@dataclass(frozen=True)
class Observation:
    hr: int
    bp: int
    temp: float
    resp: float
    ox_sats: float
    # Inspired oxygen VALUE, in the unit named by ``insp_ox_unit``: a FiO2 percentage
    # (O2_UNIT_PCT) or a supplementary flow rate in L/min (O2_UNIT_LMIN).
    insp_ox: float
    avpu: str = "Alert"
    # NEWS-2 Scale 2: patients with known chronic hypercapnic respiratory failure
    # (e.g. COPD) are scored against an 88-92% target SpO2 range instead of the
    # standard 96-98% (Scale 1). 0 = Scale 1 (default), 1 = Scale 2.
    chronic_resp: int = 0
    insp_ox_unit: str = O2_UNIT_PCT


def on_supplemental_oxygen(obs: "Observation") -> bool:
    """NEWS-2's binary "on supplemental oxygen" test, evaluated in the observation's
    own unit: FiO2 above room air (>21%), or any positive supplementary flow.
    Mirrors engine_scoring.on_supplemental_oxygen."""
    if obs.insp_ox_unit == O2_UNIT_LMIN:
        return obs.insp_ox > 0
    return obs.insp_ox > 21


def o2_display(obs: "Observation") -> Tuple[float, str]:
    """(value, unit label) for showing this observation's inspired oxygen."""
    return obs.insp_ox, ("L/min" if obs.insp_ox_unit == O2_UNIT_LMIN else "% FiO2")


def render_o2_unit(prefix: str, default_obs: "Observation", preset_name: str = "") -> str:
    """Inspired-oxygen unit picker. MUST be rendered OUTSIDE st.form.

    Streamlit does not rerun the script when a widget inside a form changes — values are
    only sent on submit — so a unit picker inside the form would leave the value box
    labelled and bounded for the previous unit until the user pressed submit. Outside the
    form it takes effect immediately and the matching box appears.

    Also applies the preset here: the widgets are keyed, so Streamlit reads them from
    session state and ignores ``value=`` after the first render. Writing session state
    directly is how the temporal tabs already reset their fields; without it the oxygen
    input stays behind while every other vital follows the preset.
    """
    pct_key, lmin_key = f"{prefix}_insp_value_pct", f"{prefix}_insp_value_lmin"
    unit_key, last_key = f"{prefix}_insp_unit", f"{prefix}_insp_last_preset"

    if st.session_state.get(last_key) != preset_name:
        preset_unit = default_obs.insp_ox_unit if default_obs.insp_ox_unit in O2_UNITS else O2_UNIT_PCT
        st.session_state[unit_key] = preset_unit
        # the unit the preset uses gets its value; the other resets to its own default.
        # Both the widget key and its shadow (see render_o2_value) are written, because
        # only one of the two boxes is rendered and the other's widget state is dropped.
        pct = float(default_obs.insp_ox if preset_unit == O2_UNIT_PCT
                    else O2_INPUT_RANGE[O2_UNIT_PCT][2])
        lmin = float(default_obs.insp_ox if preset_unit == O2_UNIT_LMIN
                     else O2_INPUT_RANGE[O2_UNIT_LMIN][2])
        st.session_state[pct_key] = st.session_state[pct_key + "_remembered"] = pct
        st.session_state[lmin_key] = st.session_state[lmin_key + "_remembered"] = lmin
        st.session_state[last_key] = preset_name

    return st.radio(
        "Inspired oxygen recorded as", O2_UNITS, horizontal=True,
        help="Each unit is scored on its own membership function — % FiO2 on the "
             "concentration sets, L/min on the supplementary-flow sets. They are never "
             "interconverted.",
        key=unit_key,
    )


def render_o2_value(prefix: str, unit: str) -> float:
    """Value box for inspired oxygen in ``unit``. Safe inside st.form.

    One box per unit, keyed separately, so switching units brings up that unit's own
    value instead of carrying a nonsensical number across — a "60" meaning 60% FiO2 must
    not silently become 60 L/min.

    Streamlit discards session state for widgets it did not render in a given run, so the
    hidden unit's box would come back reset to its minimum. A plain (non-widget) shadow
    key holds the value and seeds the widget when it is recreated. Because this box lives
    inside a form, the shadow tracks the last *submitted* value — an edit that was typed
    but not submitted before switching units is not carried over, which is ordinary form
    behaviour. ``value=`` is only passed when the widget key is absent: passing both a
    default and existing session state triggers a Streamlit warning.
    """
    lo, hi, unit_default = O2_INPUT_RANGE[unit]
    key = f"{prefix}_insp_value_{'lmin' if unit == O2_UNIT_LMIN else 'pct'}"
    shadow = key + "_remembered"

    kwargs = {}
    if key not in st.session_state:
        remembered = float(st.session_state.get(shadow, unit_default))
        kwargs["value"] = min(max(remembered, float(lo)), float(hi))
    value = float(st.number_input(
        f"Inspired oxygen ({unit})", min_value=float(lo), max_value=float(hi),
        step=0.5 if unit == O2_UNIT_LMIN else 1.0, key=key, **kwargs,
    ))
    st.session_state[shadow] = value
    return value


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


class custom_mf_sbp_merged:
    """Main-system systolic BP input MF, aligned with NEWS-2 and clinical judgement.

    NEWS-2 gives 0 points anywhere across SBP 111-219 mmHg, with no intermediate
    step before the severe (>=220 or <=90) band. Reads the standard 7-set SBP CSV
    but folds ``No concern``, ``Above normal - mild concern`` and ``Above normal -
    moderate concern`` into a single, wider ``No concern`` set (sum, clipped to
    1.0 — partition of unity is preserved since these three sets already summed
    to the remaining mass). ``Above normal - severe concern`` is left completely
    unchanged, so it now overlaps the upper tail of the widened No concern set
    directly instead of sitting behind a mild/moderate buffer: the fuzzy
    transition goes straight from no concern into severe hypertension. Hypotension
    (the three "Below normal" sets) is untouched.

    Resulting label order: severe, moderate, mild (below) -> no concern (the
    merged no-concern + mild-above + moderate-above data) -> severe (above).
    """

    LABEL_TO_CONCERN = {
        "Below normal - severe concern": "Severe concern",
        "Below normal - moderate concern": "Moderate concern",
        "Below normal - mild concern": "Mild concern",
        "No concern": "No concern",
        "Above normal - severe concern": "Severe concern",
    }

    def __init__(self, path: Path):
        self.df = pd.read_csv(path)
        keys = self.df.loc[:, "Value"].values
        self.B_SevC = dict(zip(keys, self.df.loc[:, "Below normal - severe concern"].values))
        self.B_ModC = dict(zip(keys, self.df.loc[:, "Below normal - moderate concern"].values))
        self.B_MildC = dict(zip(keys, self.df.loc[:, "Below normal - mild concern"].values))
        no_con = self.df.loc[:, "No concern"].values
        a_mild = self.df.loc[:, "Above normal - mild concern"].values
        a_mod = self.df.loc[:, "Above normal - moderate concern"].values
        merged_no_concern = [min(1.0, max(0.0, float(n) + float(m) + float(md)))
                              for n, m, md in zip(no_con, a_mild, a_mod)]
        self.no_con = dict(zip(keys, merged_no_concern))
        self.A_SevC = dict(zip(keys, self.df.loc[:, "Above normal - severe concern"].values))
        self.fs = [self.B_SevC, self.B_ModC, self.B_MildC, self.no_con, self.A_SevC]
        self.labels = list(self.LABEL_TO_CONCERN.keys())

    def __call__(self, inp: float) -> Dict[str, float]:
        return {label: _interp_lookup(fs, inp) for label, fs in zip(self.labels, self.fs)}

    def chart_df(self) -> pd.DataFrame:
        df = self.df[["Value"]].copy()
        df["Below normal - severe concern"] = self.df["Below normal - severe concern"].values
        df["Below normal - moderate concern"] = self.df["Below normal - moderate concern"].values
        df["Below normal - mild concern"] = self.df["Below normal - mild concern"].values
        df["No concern"] = [self.no_con[k] for k in self.df["Value"].values]
        df["Above normal - severe concern"] = self.df["Above normal - severe concern"].values
        return df


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
    custom_mf_sbp_merged,
    custom_mf_7_var,
    custom_mf_7_var,
    custom_mf_3_var_down,
    custom_mf_3_var_up,
]:
    base = Path(dir_str)
    return (
        custom_mf_7_var(base / "heart_rate_membership_functions.csv"),
        custom_mf_sbp_merged(base / "systolic_blood_pressure_membership_functions.csv"),
        custom_mf_7_var(base / "temperature_membership_functions.csv"),
        custom_mf_7_var(base / "respiratory_rate_membership_functions.csv"),
        custom_mf_3_var_down(base / "oxygen_saturation_membership_functions.csv"),
        custom_mf_3_var_up(base / "inspired_oxygen_concentration_membership_functions.csv"),
    )


def load_membership_functions(base_dir: Path) -> Tuple[
    custom_mf_7_var,
    custom_mf_sbp_merged,
    custom_mf_7_var,
    custom_mf_7_var,
    custom_mf_3_var_down,
    custom_mf_3_var_up,
]:
    return _load_membership_functions_from(str(base_dir.resolve()))


@lru_cache(maxsize=4)
def _load_supp_o2_mf_from(dir_str: str) -> custom_mf_3_var_up:
    return custom_mf_3_var_up(Path(dir_str) / "supplementary_oxygen_lmin_membership_functions.csv")


def load_supp_o2_mf(base_dir: Path) -> custom_mf_3_var_up:
    """Membership function for supplementary oxygen FLOW in L/min — a separate set
    from the FiO2 concentration one, loaded separately so the six-vital tuple above
    keeps its shape."""
    return _load_supp_o2_mf_from(str(base_dir.resolve()))


def inspired_oxygen_mf(base_dir: Path, unit: str):
    """The membership function that matches ``unit`` — the whole point of keeping the
    two oxygen units apart."""
    if unit == O2_UNIT_LMIN:
        return load_supp_o2_mf(base_dir)
    return load_membership_functions(base_dir)[5]


def resolve_o2_scoring(base_dir: Path, unit: str, value: float) -> Tuple[object, float]:
    """(membership function, value) actually used to score inspired oxygen.

    0 L/min IS room air, so it is scored as 21% on the concentration sets rather than at
    the bottom of the flow sets — otherwise the identical patient would score 0.43 or
    0.00 purely according to which unit someone wrote down. Mirrors
    engine_scoring.inspired_oxygen_concern.
    """
    if unit == O2_UNIT_LMIN and value > 0:
        return load_supp_o2_mf(base_dir), float(value)
    fio2 = float(value) if unit == O2_UNIT_PCT else 21.0
    return load_membership_functions(base_dir)[5], fio2


@st.cache_resource(show_spinner=False)
def output_cache() -> Tuple[OutputMF, Dict[float, Dict[str, float]]]:
    output = OutputMF()
    cache: Dict[float, Dict[str, float]] = {}
    for i in range(0, 301):
        x = i / 100.0
        cache[x] = output(x)
    return output, cache


def firings(
    hr: int,
    bp: int,
    temp: float,
    resp: float,
    ox: float,
    insp: float,
    base_dir: Path,
    avpu: str = "Alert",          # accepted for call-site compatibility; NOT scored
    insp_unit: str = O2_UNIT_PCT,
) -> Dict[str, Dict[str, float]]:
    heart_rate, blood_pressure, temperature, respiratory_rate, oxygen_saturation, _ = load_membership_functions(base_dir)
    inspired_oxygen, insp = resolve_o2_scoring(base_dir, insp_unit, insp)
    return {
        "heart rate": heart_rate(hr),
        "blood pressure": blood_pressure(bp),
        "temperature": temperature(temp),
        "respiratory rate": respiratory_rate(resp),
        "oxygen saturation": oxygen_saturation(ox),
        "inspired oxygen": inspired_oxygen(insp),
        # ACVPU deliberately absent: it is not a scored vital. Any non-Alert reading is a
        # deterioration flag (acvpu_deterioration_flag) and adds nothing to the total.
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
    # Must match engine_scoring.MIN_FIRING — the engine reproduces this function's
    # output via per-vital LUTs, and the tests assert the two agree.
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


def aggregate_total(
    scores: Dict[str, float],
    method: str,
    power: float = 2.0,
    gamma: float = 1.0,
) -> float:
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
        base_total = (1.0 - product) * max_per_vital * n
    elif method == "nonlinear":
        avg_power = sum(val ** power for val in normalized) / n
        base_total = (avg_power ** (1.0 / power)) * max_per_vital * n
    else:
        base_total = sum(per_vital)

    # Gamma controls how much a single extremely abnormal vital sign can dominate.
    # gamma = 1.0 -> original aggregation (approximately additive).
    # gamma = 0.0 -> total driven purely by the worst vital, scaled to the same range.
    gamma_clamped = max(0.0, min(1.0, float(gamma)))
    if gamma_clamped == 1.0:
        return base_total

    max_vital = max(per_vital)
    # Scale worst per-vital score (0-3) into the global 0-18 range, independent
    # of how many vitals are present, so at gamma = 0 a single vital near 3/3
    # can in principle drive the overall total toward 18 with no additivity.
    max_based_total = n * max_vital
    return (1.0 - gamma_clamped) * max_based_total + gamma_clamped * base_total


def calculate_fuzzy_ews(
    all_firings: Dict[str, Dict[str, float]],
    method: str,
    gamma: float = 1.0,
    avpu: str = "Alert",
) -> Dict[str, float]:
    per_vital_scores: Dict[str, float] = {}
    for vital_name, vital_memberships in all_firings.items():
        concern_levels = map_to_concern_levels(vital_memberships)
        score = defuzz_vital_centroid(concern_levels)
        per_vital_scores[vital_name] = score
    total = aggregate_total(per_vital_scores, method, power=2.0, gamma=gamma)
    per_vital_scores["total"] = total
    return per_vital_scores


def dominant_label(vital_memberships: Dict[str, float]) -> Tuple[str, float]:
    if not vital_memberships:
        return "No data", 0.0
    label, strength = max(vital_memberships.items(), key=lambda item: item[1])
    return label, strength


def calculate_news2(obs: Observation) -> Tuple[Dict[str, int], int]:
    """Compute NEWS-2. "On supplemental oxygen" is read in the observation's own unit
    (FiO2 > 21%, or any positive L/min flow) — see ``on_supplemental_oxygen``.

    SpO2 scoring uses Scale 1 (target 96-98%) unless ``obs.chronic_resp`` is set,
    in which case Scale 2 (target 88-92%, for known chronic hypercapnic
    respiratory failure e.g. COPD) is used instead — official NEWS-2 (RCP)
    thresholds, which also depend on whether the patient is on supplemental O2.
    """

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

    def score_spo2_scale1(x: float) -> int:
        if x <= 91:
            return 3
        if 92 <= x <= 93:
            return 2
        if 94 <= x <= 95:
            return 1
        return 0  # >=96

    def score_spo2_scale2(x: float, on_oxygen: bool) -> int:
        if x <= 83:
            return 3
        if 84 <= x <= 85:
            return 2
        if 86 <= x <= 87:
            return 1
        if 88 <= x <= 92:
            return 0  # target range
        if 93 <= x <= 94:
            return 1 if on_oxygen else 0
        if 95 <= x <= 96:
            return 2 if on_oxygen else 0
        return 3 if on_oxygen else 0  # >=97

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

    supplemental_o2 = on_supplemental_oxygen(obs)
    spo2_score = (
        score_spo2_scale2(obs.ox_sats, supplemental_o2)
        if obs.chronic_resp
        else score_spo2_scale1(obs.ox_sats)
    )

    per_vital = {
        "respiratory rate": score_resp(obs.resp),
        "oxygen saturation": spo2_score,
        "temperature": score_temp(obs.temp),
        "blood pressure": score_bp(obs.bp),
        "heart rate": score_hr(obs.hr),
    }

    consciousness = AVPU_NEWS2_SCORE.get(obs.avpu, 0)

    total = sum(per_vital.values()) + (2 if supplemental_o2 else 0) + consciousness
    per_vital["supplemental oxygen"] = 2 if supplemental_o2 else 0
    per_vital["consciousness"] = consciousness
    return per_vital, total


def clamp_observation(obs: Observation, base_dir: Path) -> Observation:
    """Clamp observation to the min/max of the selected membership grids."""
    hr_mf, bp_mf, temp_mf, resp_mf, ox_mf, _ = load_membership_functions(base_dir)
    # clamp inspired oxygen against the grid for ITS OWN unit
    insp_mf = inspired_oxygen_mf(base_dir, obs.insp_ox_unit)

    def clamp_val(val, series, *, round_1=False):
        lo, hi = float(series.min()), float(series.max())
        out = max(lo, min(hi, val))
        return round(out, 1) if round_1 else out

    hr = clamp_val(obs.hr, hr_mf.df["Value"])
    bp = clamp_val(obs.bp, bp_mf.df["Value"])
    temp = clamp_val(obs.temp, temp_mf.df["Value"], round_1=True)
    resp = clamp_val(obs.resp, resp_mf.df["Value"])
    ox = clamp_val(obs.ox_sats, ox_mf.df["Value"])
    insp = clamp_val(obs.insp_ox, insp_mf.df["Value"], round_1=True)
    return Observation(hr=int(hr), bp=int(bp), temp=float(temp), resp=float(resp), ox_sats=float(ox), insp_ox=float(insp),
                       avpu=obs.avpu, chronic_resp=obs.chronic_resp, insp_ox_unit=obs.insp_ox_unit)


def interpret_table(all_firings: Dict[str, Dict[str, float]], scores: Dict[str, float], news_scores: Dict[str, int], obs: Observation) -> pd.DataFrame:
    records = []
    vocab = {
        "heart rate": (obs.hr, "bpm"),
        "blood pressure": (obs.bp, "mmHg"),
        "temperature": (obs.temp, "\u00b0C"),
        "respiratory rate": (obs.resp, "breaths/min"),
        "oxygen saturation": (obs.ox_sats, "%"),
        "inspired oxygen": o2_display(obs),
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


def render_membership_functions(selected_dir: Path, obs: Observation,
                                all_firings: Dict[str, Dict[str, float]]) -> None:
    """Membership-function charts + firing strengths for the six scored vitals.

    Shared by both tabs: the Snapshot tab shows the observation just scored, the Temporal
    tab shows the most recent observation on the timeline. Inspired oxygen is charted
    against the set it was actually scored on, which is the flow set when the unit is
    L/min — not always the FiO2 concentration set.
    """
    st.subheader("Membership functions and firing")
    st.caption("Your input marked in red; firing strengths at that value below each chart.")

    mf = load_membership_functions(selected_dir)
    lookup = {
        "heart rate": (mf[0], obs.hr, "bpm"),
        "blood pressure": (mf[1], obs.bp, "mmHg"),
        "temperature": (mf[2], obs.temp, "°C"),
        "respiratory rate": (mf[3], obs.resp, "breaths/min"),
        "oxygen saturation": (mf[4], obs.ox_sats, "%"),
        "inspired oxygen": (resolve_o2_scoring(selected_dir, obs.insp_ox_unit, obs.insp_ox)[0],
                            *o2_display(obs)),
    }
    for vital, (model, value, unit) in lookup.items():
        with st.expander(f"{vital.title()} ({value} {unit})", expanded=False):
            if vital == "blood pressure":
                st.caption("NEWS-2-aligned: No concern spans the mild/moderate-elevated "
                           "range and overlaps Above-severe near the top.")
            source = model.chart_df() if hasattr(model, "chart_df") else model.df
            st.altair_chart(membership_chart(source, value, unit), use_container_width=True)
            st.dataframe(firing_table_df(model.labels, all_firings.get(vital, {})),
                         use_container_width=True, height=220)


def risk_bucket(total: float) -> str:
    if total < 4:
        return "Low"
    if total < 8:
        return "Moderate"
    if total < 12:
        return "High"
    return "Critical"


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------

def _render_snapshot_tab(prefix: str) -> None:
    """Single-observation ("snapshot") scoring: no timeline, no temporal adjustment."""
    selected_dir = DATA_DIR_SIGMOID
    if not selected_dir.exists():
        st.error(f"Required sigmoid membership set not found at {selected_dir}.")
        return
    aggregation_method = "additive"
    st.caption("Sigmoid membership functions, additive aggregation. Six scored vitals; "
               "ACVPU flags the row but adds no points.")

    presets = {
        "Normal": Observation(hr=80, bp=120, temp=36.8, resp=16, ox_sats=98, insp_ox=21),
        "Mild concern": Observation(hr=105, bp=135, temp=37.8, resp=22, ox_sats=94, insp_ox=28),
        "Moderate concern": Observation(hr=120, bp=100, temp=38.5, resp=26, ox_sats=91, insp_ox=30),
        "Severe concern": Observation(hr=135, bp=88, temp=39.2, resp=30, ox_sats=86, insp_ox=60),
    }

    preset_name = st.radio("Quick examples", list(presets.keys()), horizontal=True, key=f"{prefix}_preset")
    default_obs = presets[preset_name]
    st.caption("Presets pre-fill the form; edit any field before running.")

    # outside the form on purpose: form-internal changes do not rerun the script
    insp_unit = render_o2_unit(prefix, default_obs, preset_name)
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
            insp = render_o2_value(prefix, insp_unit)
            avpu = st.selectbox("AVPU / ACVPU", AVPU_OPTIONS, index=AVPU_OPTIONS.index(default_obs.avpu))
            chronic_resp = st.checkbox(
                "Chronic respiratory disease (NEWS-2 Scale 2)",
                value=bool(default_obs.chronic_resp),
                help="Uses the 88-92% SpO2 target range (e.g. COPD) instead of the standard 96-98%.",
            )
        submitted = st.form_submit_button("Run inference", use_container_width=True)

    if submitted:
        raw_obs = Observation(
            hr=int(hr), bp=int(bp), temp=float(temp), resp=int(resp),
            ox_sats=int(ox), insp_ox=float(insp), avpu=avpu, chronic_resp=int(chronic_resp),
            insp_ox_unit=insp_unit,
        )
        obs = clamp_observation(raw_obs, selected_dir)
        if obs != raw_obs:
            st.info("Inputs were clamped to the membership function range used in the model.")

        all_firings = firings(
            obs.hr, obs.bp, obs.temp, obs.resp, obs.ox_sats, obs.insp_ox, selected_dir,
            avpu=obs.avpu, insp_unit=obs.insp_ox_unit,
        )
        scores = calculate_fuzzy_ews(all_firings, aggregation_method, avpu=obs.avpu)
        total = scores.pop("total", 0.0)
        news_scores, news_total = calculate_news2(obs)

        left, right = st.columns([1, 2])
        with left:
            st.metric("Overall score (0-18)", f"{total:.2f}",
                      help="Sum of the six per-vital concern scores (0-3 each).")
            st.metric("Risk bucket", risk_bucket(total))
            st.metric(
                "NEWS-2 (0-17)",
                f"{news_total}",
                help=(
                    f"Computed with NEWS-2 {'Scale 2 (chronic resp)' if obs.chronic_resp else 'Scale 1'}; "
                    "supplemental O2 adds 2 for FiO2 > 21% or any positive L/min flow; "
                    "ACVPU contributes no consciousness points to either system."
                ),
            )
            if acvpu_deterioration_flag(obs.avpu):
                st.metric("ACVPU deterioration flag", "POSITIVE", help="Non-Alert ACVPU: the row is flagged as deteriorating. Nothing is added to the fuzzy total.")
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
        st.caption("Strongest fuzzy set per vital and what it contributes.")
        table = interpret_table(all_firings, scores, news_scores, obs)
        st.dataframe(table, use_container_width=True)

        render_membership_functions(selected_dir, obs, all_firings)


# ---------------------------------------------------------------------------
# Per-vital temporal adjustment (two-step: EWMA + worsening-trend factor)
# ---------------------------------------------------------------------------

_PV_KEYS: Dict[str, str] = {
    "heart rate": "pv_heart_rate",
    "blood pressure": "pv_blood_pressure",
    "temperature": "pv_temperature",
    "respiratory rate": "pv_respiratory_rate",
    "oxygen saturation": "pv_oxygen_saturation",
    "inspired oxygen": "pv_inspired_oxygen",
}


@dataclass(frozen=True)
class TemporalConfig:
    ewma_alpha: float = 0.7
    trend_beta: float = 2.0
    window_hours: float = 24.0
    ewma_ref_minutes: float = 60.0
    # Dead zone + persistence gate on the worsening-trend factor (mirrors
    # engine_scoring.TREND_MIN_SLOPE_DEFAULT / TREND_REQUIRE_CONSECUTIVE_DEFAULT).
    # Without a floor, a single noisy uptick already saturates the sigmoid at high
    # beta, pushing even normal-looking patients toward the ceiling. min_slope is a
    # placeholder threshold (score-points/hour), not yet tuned against data.
    min_slope: float = 0.05
    require_consecutive: bool = True


def _ewma_alpha_eff(dt_minutes: float, alpha: float, ref_minutes: float) -> float:
    if dt_minutes <= 0.0 or ref_minutes <= 0.0:
        return alpha
    return 1.0 - (1.0 - alpha) ** (dt_minutes / ref_minutes)


def _ewma(values: list, times_min: list, alpha: float, ref_minutes: float = 60.0) -> list:
    """Exponentially weighted moving average with irregular time gaps."""
    if not values:
        return []
    result = [values[0]]
    for i in range(1, len(values)):
        dt = max(float(times_min[i] - times_min[i - 1]), 0.0)
        alpha_eff = _ewma_alpha_eff(dt, alpha, ref_minutes)
        result.append(alpha_eff * values[i] + (1.0 - alpha_eff) * result[-1])
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


def _slope_ending_at(times_min: list, raw_scores: list, end_idx: int, window_min: float) -> float:
    """OLS slope of raw scores within the look-back window ending at
    times_min[end_idx], using only observations up to and including end_idx."""
    end_t = times_min[end_idx]
    window_raw: list[float] = []
    window_times: list[float] = []
    for t, s in zip(times_min[:end_idx + 1], raw_scores[:end_idx + 1]):
        if end_t - t <= window_min:
            window_raw.append(s)
            window_times.append(t)
    if len(window_raw) < 2:
        return 0.0
    t0 = window_times[0]
    window_times_h = [(t - t0) / 60.0 for t in window_times]
    return _linear_slope(window_times_h, window_raw)


def _compute_temporal_adjusted_scores(
    timeline: list,
    config: TemporalConfig,
) -> Dict[str, Dict]:
    """Two-step per-vital temporal adjustment.

    Step 1: EWMA of per-vital concern scores (0-3) over the full timeline,
            with smoothing parameter alpha.  Captures the "memory" of the
            vital sign, including any recent instability.

    Step 2: Linear trend in RAW (non-EWMA) concern scores over the look-back
            window.  If the trend is positive (worsening), a sigmoid factor
            controlled by beta pushes the score upward.  Negative (improving)
            or zero (stable) trends produce no adjustment.

    The adjusted score is guaranteed to remain in [0, 3].
    """
    if not timeline:
        return {}

    results: Dict[str, Dict] = {}
    latest_t = float(timeline[-1]["t_minutes"])
    window_min = config.window_hours * 60.0

    for vital, pv_key in _PV_KEYS.items():
        raw_scores: list[float] = []
        times_min: list[float] = []
        for entry in timeline:
            if pv_key not in entry:
                continue
            raw_scores.append(float(entry[pv_key]))
            times_min.append(float(entry["t_minutes"]))

        if not raw_scores:
            results[vital] = {
                "raw_scores": [], "ewma_scores": [],
                "ewma_current": 0.0, "trend_slope": 0.0,
                "trend_factor": 0.0, "adjusted_score": 0.0,
                "n_obs": 0, "n_trend_obs": 0,
            }
            continue

        # Step 1: EWMA of concern scores over the full timeline
        ewma_scores = _ewma(raw_scores, times_min, config.ewma_alpha, config.ewma_ref_minutes)
        # Alpha must not reduce concern: clamp EWMA up to the raw score at each time.
        clamped_ewma_scores = [
            max(ewma, raw) for ewma, raw in zip(ewma_scores, raw_scores)
        ]
        latest_raw = raw_scores[-1]
        ewma_current = clamped_ewma_scores[-1]

        # Step 2: linear trend in RAW concern scores within the look-back window
        n_obs = len(raw_scores)
        slope = _slope_ending_at(times_min, raw_scores, n_obs - 1, window_min)
        window_raw = [s for t, s in zip(times_min, raw_scores) if latest_t - t <= window_min]

        # Dead zone: a single noisy uptick shouldn't saturate the sigmoid — the
        # slope must clear min_slope before it counts as "worsening" at all.
        fires = slope > config.min_slope
        # Persistence gate: also require the PREVIOUS observation's own
        # window-ending slope to have cleared the dead zone, so one blip isn't
        # enough — the rise has to hold across two consecutive readings.
        if fires and config.require_consecutive:
            if n_obs < 2:
                fires = False
            else:
                prev_slope = _slope_ending_at(times_min, raw_scores, n_obs - 2, window_min)
                fires = prev_slope > config.min_slope

        # Sigmoid trend factor: only when the dead zone / persistence gate fires
        if fires:
            trend_factor = 2.0 / (1.0 + math.exp(-config.trend_beta * (slope - config.min_slope))) - 1.0
        else:
            trend_factor = 0.0

        # Push EWMA toward 3 proportionally — guarantees result in [0, 3]
        adjusted = ewma_current + trend_factor * (3.0 - ewma_current)
        adjusted = max(0.0, min(3.0, adjusted))

        results[vital] = {
            "raw_scores": [round(s, 3) for s in raw_scores],
            "ewma_scores": [round(s, 3) for s in clamped_ewma_scores],
            "ewma_current": round(ewma_current, 3),
            "trend_slope": round(slope, 4),
            "trend_factor": round(trend_factor, 4),
            "adjusted_score": round(adjusted, 3),
            "n_obs": len(raw_scores),
            "n_trend_obs": len(window_raw),
        }

    return results


def _render_temporal_tab(prefix: str) -> None:
    """EWMA smoothing + worsening-trend factor over a timeline of observations."""
    firings_fn, calc_fn = firings, calculate_fuzzy_ews
    st.caption(
        "Build a timeline of observations. Each vital's concern score is smoothed (EWMA), "
        "then pushed up if it is on a worsening trend \u2014 never down. All six vitals, "
        "oxygen included; ACVPU only flags the row."
    )

    selected_dir = DATA_DIR_SIGMOID
    if not selected_dir.exists():
        st.error(f"Required sigmoid membership set not found at {selected_dir}.")
        return
    aggregation_method = "additive"

    presets = {
        "Normal": Observation(hr=80, bp=120, temp=36.8, resp=16, ox_sats=98, insp_ox=21),
        "Mild concern": Observation(hr=105, bp=135, temp=37.8, resp=22, ox_sats=94, insp_ox=28),
        "Moderate concern": Observation(hr=120, bp=100, temp=38.5, resp=26, ox_sats=91, insp_ox=30),
        "Severe concern": Observation(hr=135, bp=88, temp=39.2, resp=30, ox_sats=86, insp_ox=60),
    }

    timeline_key = f"{prefix}_timeline"
    if timeline_key not in st.session_state:
        st.session_state[timeline_key] = []

    # ------------------------------------------------------------------
    # Temporal parameters
    # ------------------------------------------------------------------
    st.subheader("Temporal parameters")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        cfg_alpha = st.slider(
            "\u03b1 (EWMA smoothing)",
            min_value=0.05, max_value=1.0, value=0.7, step=0.05,
            key=f"{prefix}_cfg_alpha",
            help=(
                "Weight given to the most recent observation in the EWMA. "
                "Higher \u03b1 = more responsive to recent values; lower \u03b1 = longer memory."
            ),
        )
    with p2:
        cfg_beta = st.slider(
            "\u03b2 (trend strength)",
            min_value=0.1, max_value=10.0, value=2.0, step=0.1,
            key=f"{prefix}_cfg_beta",
            help=(
                "Controls how strongly a worsening trend in raw concern scores "
                "pushes the adjusted score upward via the sigmoid. "
                "Higher \u03b2 = stronger response to even small positive trends."
            ),
        )
    with p3:
        cfg_window = st.number_input(
            "Look-back window (hours)",
            min_value=1.0, max_value=72.0, value=24.0, step=1.0,
            key=f"{prefix}_cfg_window",
            help="The trend slope is computed over raw concern scores within this window.",
        )
    with p4:
        cfg_gamma = st.slider(
            "\u03b3 (single-vital dominance)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key=f"{prefix}_cfg_gamma",
            help=(
                "Controls how much a single extremely abnormal vital can dominate the total fuzzy total. "
                "\u03b3 = 1.0 \u2192 approximately additive (original behaviour); "
                "\u03b3 = 0.0 \u2192 total driven purely by the worst vital (no additivity)."
            ),
        )
    p5, p6 = st.columns(2)
    with p5:
        cfg_min_slope = st.slider(
            "Trend dead zone (min slope, pts/hr)",
            min_value=0.0, max_value=1.0, value=0.05, step=0.01,
            key=f"{prefix}_cfg_min_slope",
            help=(
                "A raw slope must exceed this rate (concern-points per hour) before "
                "it counts as 'worsening' at all. Filters single noisy upticks that "
                "would otherwise saturate the sigmoid and inflate normal-looking "
                "patients. Placeholder value, not yet tuned against data."
            ),
        )
    with p6:
        cfg_require_consecutive = st.checkbox(
            "Require 2 consecutive rising readings",
            value=True,
            key=f"{prefix}_cfg_require_consecutive",
            help=(
                "In addition to the dead zone, also require the PREVIOUS observation's "
                "own trend slope to have cleared it — one blip isn't enough, the rise "
                "must persist across two consecutive readings before it fires."
            ),
        )
    t_config = TemporalConfig(
        ewma_alpha=cfg_alpha,
        trend_beta=cfg_beta,
        window_hours=cfg_window,
        min_slope=cfg_min_slope,
        require_consecutive=cfg_require_consecutive,
    )

    # ------------------------------------------------------------------
    # Method explanation
    # ------------------------------------------------------------------
    with st.expander("How the temporal adjustment works", expanded=False):
        st.markdown(
            "**1 \u2014 EWMA** smooths each vital's concern score, keeping memory of worse "
            "past readings:\n\n"
            r"$$\text{EWMA}_t = \alpha x_t + (1-\alpha)\text{EWMA}_{t-1}$$"
            "\n\n**2 \u2014 Trend** fits a slope *s* to the **raw** scores over the look-back "
            "window. It only counts as worsening if *s* clears the dead zone, and (if enabled) "
            "did so on the previous reading too \u2014 one blip is not a trend:\n\n"
            r"$$f = \frac{2}{1+e^{-\beta(s-s_{\min})}} - 1$$"
            "\n\n"
            r"$$\text{adjusted} = b + f\,(3-b), \quad b = \max(\text{EWMA},\, x_{\text{latest}})$$"
            "\n\nSo the result stays in [0, 3] and never drops below the latest snapshot. "
            "Improving or stable trends give *f* = 0.\n\n"
            "**\u03b3** mixes additive and worst-vital totals; \u03b3 = 1 is purely additive, "
            "\u03b3 = 0 is driven by the single worst vital:\n\n"
            r"$$\text{total} = (1-\gamma)\,n\max_i v_i + \gamma \sum_i v_i$$"
        )

    # ------------------------------------------------------------------
    # Add observation
    # ------------------------------------------------------------------
    st.subheader("Add observation")
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        interval_unit = st.selectbox(
            "Interval unit", ["minutes", "hours"], index=1,
            key=f"{prefix}_unit",
        )
    with col_cfg2:
        interval_value = st.number_input(
            "Interval size", min_value=1, max_value=240, value=1,
            key=f"{prefix}_interval",
        )

    preset_name = st.radio(
        "Quick examples", list(presets.keys()), horizontal=True,
        key=f"{prefix}_preset",
    )
    default_obs = presets[preset_name]

    hr_key = f"{prefix}_hr"
    bp_key = f"{prefix}_bp"
    temp_key = f"{prefix}_temp"
    resp_key = f"{prefix}_resp"
    ox_key = f"{prefix}_ox"
    avpu_key = f"{prefix}_avpu"
    chronic_key = f"{prefix}_chronic_resp"
    last_preset_key = f"{prefix}_last_preset"

    if st.session_state.get(last_preset_key) != preset_name:
        st.session_state[hr_key] = int(default_obs.hr)
        st.session_state[bp_key] = int(default_obs.bp)
        st.session_state[temp_key] = float(default_obs.temp)
        st.session_state[resp_key] = int(default_obs.resp)
        st.session_state[ox_key] = int(default_obs.ox_sats)
        st.session_state[avpu_key] = default_obs.avpu
        st.session_state[chronic_key] = bool(default_obs.chronic_resp)
        st.session_state[last_preset_key] = preset_name

    # outside the form on purpose: form-internal changes do not rerun the script
    insp_unit = render_o2_unit(prefix, default_obs, preset_name)
    with st.form(f"{prefix}_add_observation"):
        col1, col2, col3 = st.columns(3)
        with col1:
            hr = st.number_input(
                "Heart rate (bpm)", min_value=30, max_value=200,
                value=int(st.session_state.get(hr_key, default_obs.hr)), key=hr_key,
            )
            bp = st.number_input(
                "Systolic BP (mmHg)", min_value=50, max_value=220,
                value=int(st.session_state.get(bp_key, default_obs.bp)), key=bp_key,
            )
        with col2:
            temp = st.number_input(
                "Temperature (\u00b0C)", min_value=30.0, max_value=43.0,
                value=float(st.session_state.get(temp_key, default_obs.temp)),
                step=0.1, format="%.1f", key=temp_key,
            )
            resp = st.number_input(
                "Respiratory rate (breaths/min)", min_value=4, max_value=50,
                value=int(st.session_state.get(resp_key, default_obs.resp)), key=resp_key,
            )
        with col3:
            ox = st.number_input(
                "Oxygen saturation (%)", min_value=70, max_value=102,
                value=int(st.session_state.get(ox_key, default_obs.ox_sats)), key=ox_key,
            )
            insp = render_o2_value(prefix, insp_unit)
            avpu = st.selectbox(
                "AVPU / ACVPU", AVPU_OPTIONS,
                index=AVPU_OPTIONS.index(st.session_state.get(avpu_key, default_obs.avpu)),
                key=avpu_key,
            )
            chronic_resp = st.checkbox(
                "Chronic resp. disease (NEWS-2 Scale 2)",
                value=bool(st.session_state.get(chronic_key, default_obs.chronic_resp)),
                key=chronic_key,
            )
        add_clicked = st.form_submit_button("Add observation", use_container_width=True)

    if add_clicked:
        raw_obs = Observation(hr=int(hr), bp=int(bp), temp=float(temp), resp=int(resp), ox_sats=int(ox),
                              insp_ox=float(insp), avpu=avpu, chronic_resp=int(chronic_resp),
                              insp_ox_unit=insp_unit)
        clamped = clamp_observation(raw_obs, selected_dir)
        all_f = firings_fn(
            clamped.hr, clamped.bp, clamped.temp, clamped.resp,
            clamped.ox_sats, clamped.insp_ox, selected_dir, avpu=clamped.avpu,
            insp_unit=clamped.insp_ox_unit,
        )
        pv_scores = calc_fn(all_f, aggregation_method, gamma=cfg_gamma, avpu=clamped.avpu)
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
            "inspired_oxygen": float(raw_obs.insp_ox),
            "inspired_oxygen_unit": raw_obs.insp_ox_unit,
            "avpu": raw_obs.avpu,
            "chronic_resp": raw_obs.chronic_resp,
            "model_heart_rate": int(clamped.hr),
            "model_blood_pressure": int(clamped.bp),
            "model_temperature": float(clamped.temp),
            "model_respiratory_rate": int(clamped.resp),
            "model_oxygen_saturation": int(clamped.ox_sats),
            "model_inspired_oxygen": float(clamped.insp_ox),
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
            avpu=entry.get("avpu", "Alert"),
            chronic_resp=entry.get("chronic_resp", 0),
            # entries saved before the unit split were all FiO2%
            insp_ox_unit=entry.get("inspired_oxygen_unit", O2_UNIT_PCT),
        )
        obs_model = clamp_observation(obs_raw, selected_dir)
        all_f = firings_fn(
            obs_model.hr, obs_model.bp, obs_model.temp, obs_model.resp,
            obs_model.ox_sats, obs_model.insp_ox, selected_dir, avpu=obs_model.avpu,
            insp_unit=obs_model.insp_ox_unit,
        )
        pv_scores = calc_fn(all_f, aggregation_method, gamma=cfg_gamma, avpu=obs_model.avpu)
        if needs_snap:
            entry["snapshot_fuzzy_ews"] = round(float(pv_scores.get("total", 0.0)), 2)
            _, news_total = calculate_news2(obs_raw)
            entry["snapshot_news2"] = int(news_total)
        if needs_pv:
            for vital, pv_key in _PV_KEYS.items():
                if pv_key not in entry:
                    entry[pv_key] = round(pv_scores.get(vital, 0.0), 3)

    # ------------------------------------------------------------------
    # Compute two-step temporal adjustments
    # ------------------------------------------------------------------
    temporal_results = _compute_temporal_adjusted_scores(timeline, t_config)

    snapshot_total = float(timeline[-1].get("snapshot_fuzzy_ews", 0.0))

    # Aggregate EWMA and trend-adjusted per-vital scores using the same gamma rule
    # used for snapshot totals, so a single extremely abnormal (and worsening)
    # vital can, in principle, drive the overall temporal concern score toward 18
    # when gamma is near 0, while gamma near 1.0 recovers approximately additive behaviour.
    ewma_per_vital = {vital: r.get("ewma_current", 0.0) for vital, r in temporal_results.items()}
    adjusted_per_vital = {vital: r.get("adjusted_score", 0.0) for vital, r in temporal_results.items()}
    ewma_total = aggregate_total(ewma_per_vital, method="additive", power=2.0, gamma=cfg_gamma)
    adjusted_total = aggregate_total(adjusted_per_vital, method="additive", power=2.0, gamma=cfg_gamma)
    # Temporal context (alpha / EWMA) must not lower the overall concern score.
    ewma_total = max(ewma_total, snapshot_total)
    adjusted_total = max(adjusted_total, snapshot_total)

    # ------------------------------------------------------------------
    # Headline metrics
    # ------------------------------------------------------------------
    st.subheader("Temporal-adjusted scores")
    latest_avpu = timeline[-1].get("avpu", "Alert")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "Snapshot total (0\u201318)",
            f"{snapshot_total:.2f}",
            help="Sum of per-vital concern scores at the latest observation (no temporal adjustment).",
        )
    with m2:
        st.metric(
            "EWMA total (0\u201318)",
            f"{ewma_total:.2f}",
            delta=f"{ewma_total - snapshot_total:+.2f}",
            help="Sum of EWMA-smoothed per-vital scores (Step 1). Clamped so it cannot fall below the snapshot total.",
        )
    with m3:
        adj_display = min(adjusted_total, 18.0)
        st.metric(
            "Trend-adjusted total (0\u201318)",
            f"{adj_display:.2f}",
            delta=f"{adjusted_total - ewma_total:+.2f}",
            help="EWMA scores adjusted upward for worsening trends (Step 2: trend factor).",
        )
    with m4:
        st.metric("Risk bucket", risk_bucket(adj_display))
    with m5:
        if acvpu_deterioration_flag(latest_avpu):
            st.metric("ACVPU flag", "POSITIVE", help="Latest ACVPU non-Alert: the row is flagged as deteriorating. Nothing is added to the fuzzy total.")
        else:
            st.metric("ACVPU flag", "—")

    # ------------------------------------------------------------------
    # Per-vital detail table
    # ------------------------------------------------------------------
    st.subheader("Per-vital temporal detail")
    st.caption("EWMA = memory of past scores. Trend factor = how hard a worsening slope pushes up.")
    detail_rows = []
    for vital in _PV_KEYS:
        r = temporal_results.get(vital, {})
        raw_latest = r["raw_scores"][-1] if r.get("raw_scores") else 0.0
        slope = r.get("trend_slope", 0.0)
        trend_factor = r.get("trend_factor", 0.0)
        if slope > 0:
            trend_label = "\u25b2 Worsening"
        elif slope < 0:
            trend_label = "\u25bc Improving"
        else:
            trend_label = "\u2014 Stable"
        detail_rows.append({
            "Vital": vital.title(),
            "Latest raw (0\u20133)": round(raw_latest, 2),
            "EWMA (0\u20133)": r.get("ewma_current", 0.0),
            "Trend slope (/hr)": round(slope, 4),
            "Trend direction": trend_label,
            "Trend factor f": round(trend_factor, 4),
            "Adjusted (0\u20133)": r.get("adjusted_score", 0.0),
            "Obs (total)": r.get("n_obs", 0),
            "Obs (window)": r.get("n_trend_obs", 0),
        })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Membership functions for the most recent observation
    # ------------------------------------------------------------------
    latest = timeline[-1]
    latest_obs = Observation(
        hr=int(latest["model_heart_rate"]),
        bp=int(latest["model_blood_pressure"]),
        temp=float(latest["model_temperature"]),
        resp=float(latest["model_respiratory_rate"]),
        ox_sats=float(latest["model_oxygen_saturation"]),
        insp_ox=float(latest["model_inspired_oxygen"]),
        avpu=latest.get("avpu", "Alert"),
        chronic_resp=latest.get("chronic_resp", 0),
        insp_ox_unit=latest.get("inspired_oxygen_unit", O2_UNIT_PCT),
    )
    render_membership_functions(
        selected_dir, latest_obs,
        firings_fn(latest_obs.hr, latest_obs.bp, latest_obs.temp, latest_obs.resp,
                   latest_obs.ox_sats, latest_obs.insp_ox, selected_dir,
                   avpu=latest_obs.avpu, insp_unit=latest_obs.insp_ox_unit),
    )
    st.caption(f"Showing observation {latest['idx']} (the most recent).")

    # ------------------------------------------------------------------
    # EWMA trend visualisation
    # ------------------------------------------------------------------
    if len(timeline) >= 2:
        st.subheader("EWMA trend visualisation")
        chart_records: list[dict] = []
        for vital in _PV_KEYS:
            r = temporal_results.get(vital, {})
            raw_scores = r.get("raw_scores", [])
            ewma_scores = r.get("ewma_scores", [])
            for i, entry in enumerate(timeline):
                t_h = float(entry["t_minutes"]) / 60.0
                if i < len(raw_scores):
                    chart_records.append({
                        "Time (hours)": round(t_h, 2),
                        "Vital": vital.title(),
                        "Series": "Raw concern",
                        "Score": raw_scores[i],
                    })
                if i < len(ewma_scores):
                    chart_records.append({
                        "Time (hours)": round(t_h, 2),
                        "Vital": vital.title(),
                        "Series": "EWMA",
                        "Score": ewma_scores[i],
                    })

        if chart_records:
            chart_df = pd.DataFrame(chart_records)
            vital_options = [v.title() for v in _PV_KEYS]
            vital_selector = alt.selection_point(
                fields=["Vital"],
                bind=alt.binding_select(options=vital_options, name="Vital: "),
                value=vital_options[0],
            )
            ewma_chart = (
                alt.Chart(chart_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Time (hours):Q"),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 3]),
                            title="Concern score (0\u20133)"),
                    color=alt.Color("Series:N"),
                    strokeDash=alt.StrokeDash("Series:N"),
                    tooltip=["Vital:N", "Series:N", "Time (hours):Q", "Score:Q"],
                )
                .add_params(vital_selector)
                .transform_filter(vital_selector)
                .properties(height=300)
            )
            st.altair_chart(ewma_chart, use_container_width=True)

    # ------------------------------------------------------------------
    # Timeline snapshots table
    # ------------------------------------------------------------------
    df = pd.DataFrame(timeline)
    df = df.sort_values("idx").reset_index(drop=True)

    pv_display_cols = list(_PV_KEYS.values())
    base_cols = [
        "idx", "t_minutes", "heart_rate", "blood_pressure", "temperature",
        "respiratory_rate", "oxygen_saturation", "inspired_oxygen", "inspired_oxygen_unit",
        "snapshot_fuzzy_ews", "snapshot_news2",
    ]
    available_cols = [c for c in base_cols + pv_display_cols if c in df.columns]
    compact_df = df[available_cols].copy()
    compact_df = compact_df.rename(columns={
        "idx": "Obs #", "t_minutes": "Time (min)",
        "heart_rate": "HR", "blood_pressure": "BP", "temperature": "Temp",
        "respiratory_rate": "Resp", "oxygen_saturation": "SpO2",
        # NOT "FiO2" — the value may be a supplementary flow in L/min
        "inspired_oxygen": "Insp O2", "inspired_oxygen_unit": "O2 unit",
        "snapshot_fuzzy_ews": "Fuzzy EWS", "snapshot_news2": "NEWS-2",
        "pv_heart_rate": "PV HR", "pv_blood_pressure": "PV BP",
        "pv_temperature": "PV Temp", "pv_respiratory_rate": "PV Resp",
        "pv_oxygen_saturation": "PV SpO2", "pv_inspired_oxygen": "PV Insp O2",
    })

    st.subheader("Timeline snapshots")
    st.dataframe(compact_df, use_container_width=True, hide_index=True)

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

    tab_snapshot, tab_temporal = st.tabs(["Snapshot", "Temporal"])

    with tab_snapshot:
        _render_snapshot_tab("t1")

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
