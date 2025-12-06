import altair as alt
import pandas as pd
import streamlit as st
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

DATA_DIR_DEFAULT = Path(__file__).parent / "data" / "membership_function_plots" / "csv_data"
DATA_DIR_SIGMOID = Path(__file__).parent / "generated_membership_data" / "sigmoid"
DATA_DIR_TRAPEZOIDAL = Path(__file__).parent / "generated_membership_data" / "trapezoidal"


@dataclass(frozen=True)
class Observation:
    hr: int
    bp: int
    temp: float
    resp: float
    ox_sats: float
    insp_ox: float


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
            out[label] = fs.get(inp, 0.0)
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
            out[label] = fs.get(inp, 0.0)
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
            out[label] = fs.get(inp, 0.0)
        return out


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


def firings(hr: int, bp: int, temp: float, resp: int, ox: int, insp: int, base_dir: Path) -> Dict[str, Dict[str, float]]:
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
    # Avoid a non-zero baseline when only the "No concern" set is active
    if concern_levels.get("No concern", 0.0) > 0 and all(
        level == "No concern" or firing == 0 for level, firing in concern_levels.items()
    ):
        return 0.0

    output, cache = output_cache()
    numerator = 0.0
    denominator = 0.0
    for i in range(0, 301):
        x = i / 100.0
        output_memberships = cache[x]
        aggregated = 0.0
        for level, firing in concern_levels.items():
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
        "temperature": (obs.temp, "°C"),
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


def main() -> None:
    if not has_streamlit_context():
        print("This file is meant to run with: streamlit run streamlit_app.py")
        return

    st.set_page_config(page_title="Fuzzy EWS", page_icon="🩺", layout="wide")
    st.title("Fuzzy Early Warning Score (EWS)")

    membership_options = {
        "Oringal": DATA_DIR_DEFAULT,
        "Generated sigmoid": DATA_DIR_SIGMOID,
        "Generated trapezoidal": DATA_DIR_TRAPEZOIDAL,
    }

    st.write("Select membership function set")
    choice = st.radio(
        "Membership shape",
        list(membership_options.keys()),
        horizontal=True,
        help="Toggle which membership CSVs to use for fuzzification.",
    )
    selected_dir = membership_options[choice]
    if not selected_dir.exists():
        st.warning(f"Selected membership set not found at {selected_dir}. Falling back to default.")
        selected_dir = DATA_DIR_DEFAULT

    presets = {
        "Normal": Observation(hr=80, bp=120, temp=36.8, resp=16, ox_sats=98, insp_ox=21),
        "Mild concern": Observation(hr=105, bp=135, temp=37.8, resp=22, ox_sats=94, insp_ox=24),
        "Moderate concern": Observation(hr=120, bp=100, temp=38.5, resp=26, ox_sats=91, insp_ox=30),
        "Severe concern": Observation(hr=135, bp=88, temp=39.2, resp=30, ox_sats=86, insp_ox=60),
    }

    preset_name = st.radio("Quick examples", list(presets.keys()), horizontal=True)
    default_obs = presets[preset_name]
    st.caption("Select a preset to pre-fill the form; adjust any field before running inference.")

    with st.form("inputs"):
        col1, col2, col3 = st.columns(3)
        with col1:
            hr = st.number_input("Heart rate (bpm)", min_value=30, max_value=200, value=default_obs.hr)
            bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=220, value=default_obs.bp)
        with col2:
            temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=43.0, value=default_obs.temp, step=0.1, format="%.1f")
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
        scores = calculate_fuzzy_ews_additive(all_firings)
        total = scores.pop("total", 0.0)
        news_scores, news_total = calculate_news2(obs)

        left, right = st.columns([1, 2])
        with left:
            st.metric("Fuzzy EWS (0-18)", f"{total:.2f}", help="Sum of per-vital fuzzy scores (0-3 each).")
            st.metric("Risk bucket", risk_bucket(total))
            st.metric("NEWS-2 (0-20)", f"{news_total}", help="Computed with NEWS-2 Scale 1; supplemental O2 adds 2 if FiO2 > 21%. Consciousness assumed 0.")
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
            "temperature": (mf_models[2], obs.temp, "°C"),
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
