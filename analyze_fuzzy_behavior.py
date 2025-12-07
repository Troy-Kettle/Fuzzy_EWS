import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streamlit_app import (
    Observation,
    clamp_observation,
    firings,
    calculate_fuzzy_ews_additive,
    load_membership_functions,
    DATA_DIR_DEFAULT,
    DATA_DIR_SIGMOID,
    DATA_DIR_TRAPEZOIDAL,
)

plt.switch_backend("Agg")

NORMAL_OBS = Observation(hr=80, bp=120, temp=36.8, resp=16, ox_sats=98, insp_ox=21)


def compute_total(obs: Observation, base_dir: Path) -> float:
    all_firings = firings(obs.hr, obs.bp, obs.temp, obs.resp, obs.ox_sats, obs.insp_ox, base_dir)
    scores = calculate_fuzzy_ews_additive(all_firings)
    return scores.get("total", 0.0)


def get_ranges(base_dir: Path) -> Dict[str, Tuple[float, float]]:
    hr_mf, bp_mf, temp_mf, resp_mf, ox_mf, insp_mf = load_membership_functions(base_dir)
    return {
        "hr": (float(hr_mf.df["Value"].min()), float(hr_mf.df["Value"].max())),
        "bp": (float(bp_mf.df["Value"].min()), float(bp_mf.df["Value"].max())),
        "temp": (float(temp_mf.df["Value"].min()), float(temp_mf.df["Value"].max())),
        "resp": (float(resp_mf.df["Value"].min()), float(resp_mf.df["Value"].max())),
        "ox_sats": (float(ox_mf.df["Value"].min()), float(ox_mf.df["Value"].max())),
        "insp_ox": (float(insp_mf.df["Value"].min()), float(insp_mf.df["Value"].max())),
    }


def replace_obs(base: Observation, **kwargs) -> Observation:
    data = base.__dict__.copy()
    data.update(kwargs)
    return Observation(
        hr=int(data["hr"]),
        bp=int(data["bp"]),
        temp=float(data["temp"]),
        resp=float(data["resp"]),
        ox_sats=float(data["ox_sats"]),
        insp_ox=float(data["insp_ox"]),
    )


def sweep_single(vital_key: str, values: np.ndarray, base_obs: Observation, base_dir: Path) -> pd.DataFrame:
    rows = []
    for v in values:
        obs = replace_obs(base_obs, **{vital_key: v})
        obs = clamp_observation(obs, base_dir)
        score = compute_total(obs, base_dir)
        rows.append({vital_key: float(v), "fuzzy_score": score})
    return pd.DataFrame(rows)


def sweep_pair(v1: str, v2: str, vals1: np.ndarray, vals2: np.ndarray, base_obs: Observation, base_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Z = np.meshgrid(vals1, vals2)
    Y = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            obs = replace_obs(base_obs, **{v1: X[i, j], v2: Z[i, j]})
            obs = clamp_observation(obs, base_dir)
            Y[i, j] = compute_total(obs, base_dir)
    return X, Z, Y


def plot_single(df: pd.DataFrame, vital_key: str, label: str, unit: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[vital_key], df["fuzzy_score"], lw=2)
    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel("Fuzzy score (0-18)")
    ax.set_title(f"Fuzzy score vs {label}")
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pair(X: np.ndarray, Z: np.ndarray, Y: np.ndarray, v1: str, v2: str, label1: str, label2: str, unit1: str, unit2: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Z, Y, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel(f"{label1} ({unit1})")
    ax.set_ylabel(f"{label2} ({unit2})")
    ax.set_zlabel("Fuzzy score (0-18)")
    ax.set_title(f"Fuzzy score vs {label1} & {label2}")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Fuzzy score")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Explore fuzzy model behaviour across vital ranges.")
    parser.add_argument(
        "--membership-set",
        choices=["default", "sigmoid", "trapezoidal"],
        default="default",
        help="Which membership CSV set to use.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("behavior_analysis"), help="Directory for plots and CSVs.")
    parser.add_argument("--single-points", type=int, default=100, help="Points per single-vital sweep.")
    parser.add_argument("--pair-grid", type=int, default=30, help="Grid size per axis for pair sweeps.")
    args = parser.parse_args()

    base_dir = {
        "default": DATA_DIR_DEFAULT,
        "sigmoid": DATA_DIR_SIGMOID,
        "trapezoidal": DATA_DIR_TRAPEZOIDAL,
    }[args.membership_set]

    ranges = get_ranges(base_dir)
    out_dir = args.output_dir

    # Single-vital sweeps
    single_defs = [
        ("hr", "Heart rate", "bpm", np.linspace(*ranges["hr"], args.single_points)),
        ("bp", "Systolic BP", "mmHg", np.linspace(*ranges["bp"], args.single_points)),
        ("resp", "Respiratory rate", "breaths/min", np.linspace(*ranges["resp"], args.single_points)),
        ("ox_sats", "Oxygen saturation", "%", np.linspace(*ranges["ox_sats"], args.single_points)),
        ("insp_ox", "Inspired O2 (FiO2 %)", "%", np.linspace(*ranges["insp_ox"], args.single_points)),
    ]

    # Inspired O2 in L/min (converted to % using ~4% per L/min, capped by range)
    fio2_lo, fio2_hi = ranges["insp_ox"]
    lpm_hi = max(0.0, (fio2_hi - 21.0) / 4.0)
    single_defs.append(("insp_lpm", "Inspired O2", "L/min", np.linspace(0.0, lpm_hi, args.single_points)))

    single_dir = out_dir / "single_vital"
    for key, label, unit, values in single_defs:
        if key == "insp_lpm":
            fio2_values = 21.0 + 4.0 * values
            df = sweep_single("insp_ox", fio2_values, NORMAL_OBS, base_dir)
            df.insert(0, "insp_lpm", values)
            csv_path = single_dir / f"{key}.csv"
            plot_single(df, "insp_lpm", label, unit, single_dir / f"{key}.png")
        else:
            df = sweep_single(key, values, NORMAL_OBS, base_dir)
            csv_path = single_dir / f"{key}.csv"
            plot_single(df, key, label, unit, single_dir / f"{key}.png")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    # Pair sweeps (20 illustrative combos)
    pair_candidates: List[Tuple[str, str, str, str, str, str]] = [
        ("resp", "insp_ox", "Respiratory rate", "Inspired O2 (FiO2 %)", "breaths/min", "%"),
        ("resp", "ox_sats", "Respiratory rate", "Oxygen saturation", "breaths/min", "%"),
        ("resp", "temp", "Respiratory rate", "Temperature", "breaths/min", "°C"),
        ("resp", "hr", "Respiratory rate", "Heart rate", "breaths/min", "bpm"),
        ("resp", "bp", "Respiratory rate", "Systolic BP", "breaths/min", "mmHg"),
        ("hr", "bp", "Heart rate", "Systolic BP", "bpm", "mmHg"),
        ("hr", "ox_sats", "Heart rate", "Oxygen saturation", "bpm", "%"),
        ("hr", "temp", "Heart rate", "Temperature", "bpm", "°C"),
        ("hr", "insp_ox", "Heart rate", "Inspired O2 (FiO2 %)", "bpm", "%"),
        ("bp", "ox_sats", "Systolic BP", "Oxygen saturation", "mmHg", "%"),
        ("bp", "insp_ox", "Systolic BP", "Inspired O2 (FiO2 %)", "mmHg", "%"),
        ("bp", "temp", "Systolic BP", "Temperature", "mmHg", "°C"),
        ("ox_sats", "insp_ox", "Oxygen saturation", "Inspired O2 (FiO2 %)", "%", "%"),
        ("ox_sats", "temp", "Oxygen saturation", "Temperature", "%", "°C"),
        ("ox_sats", "resp", "Oxygen saturation", "Respiratory rate", "%", "breaths/min"),
        ("temp", "insp_ox", "Temperature", "Inspired O2 (FiO2 %)", "°C", "%"),
        ("temp", "resp", "Temperature", "Respiratory rate", "°C", "breaths/min"),
        ("temp", "hr", "Temperature", "Heart rate", "°C", "bpm"),
        ("insp_ox", "resp", "Inspired O2 (FiO2 %)", "Respiratory rate", "%", "breaths/min"),
        ("insp_ox", "ox_sats", "Inspired O2 (FiO2 %)", "Oxygen saturation", "%", "%"),
    ]

    pair_dir = out_dir / "pair_vitals"
    vals_cache: Dict[str, np.ndarray] = {
        k: np.linspace(*ranges[k], args.pair_grid) for k in ["hr", "bp", "resp", "ox_sats", "insp_ox", "temp"]
    }

    for v1, v2, label1, label2, unit1, unit2 in pair_candidates:
        X_vals = vals_cache[v1]
        Z_vals = vals_cache[v2]
        X, Z, Y = sweep_pair(v1, v2, X_vals, Z_vals, NORMAL_OBS, base_dir)
        plot_pair(X, Z, Y, v1, v2, label1, label2, unit1, unit2, pair_dir / f"{v1}__{v2}.png")
        # Save underlying grid
        pair_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pair_dir / f"{v1}__{v2}.npz", X=X, Z=Z, Y=Y, v1=v1, v2=v2, label1=label1, label2=label2, unit1=unit1, unit2=unit2)


if __name__ == "__main__":
    main()
