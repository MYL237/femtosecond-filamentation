import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "\u5f1b\u8c6b\u8fc7\u7a0b\u6e29\u5ea6\u53d8\u5316.py"
SOURCE_PATH = (
    BASE_DIR.parent.parent
    / "\u98de\u79d2\u6fc0\u5149\u7a0b\u5e8f\u53ca\u7ed3\u679c"
    / "\u4ee3\u7801"
    / "\u5f1b\u8c6b\u8fc7\u7a0b\u6e29\u5ea6\u53d8\u5316.py"
)
OUTPUT_DIR = (
    BASE_DIR
    / "validation_outputs"
    / "input_only_corrected"
    / "chapter3_same_initial_as_sharma_full_radial_pde_20260728"
)
INITIAL_CSV = OUTPUT_DIR / "sharma_figure4_initial_condition.csv"
COMPARISON_CSV = OUTPUT_DIR / "sharma_figure4_comparison.csv"
NORMALIZED_CSV = OUTPUT_DIR / "sharma_figure4_normalized_decay.csv"
MANIFEST_CSV = OUTPUT_DIR / "sharma_figure4_manifest.csv"
THREE_TEMPERATURES_CSV = OUTPUT_DIR / "sharma_figure4_three_temperatures_K.csv"

I0_W_CM2 = 2.70e13
TG0_K = 300.0
TV0_K = 300.0
TE0_EV = 1.0
EV_PER_K = 8.617333262e-5
TE0_K = TE0_EV / EV_PER_K
T_END_NS = 12.0
VALIDATION_DT_NS = 0.0001
SAVE_EVERY = 1
VALIDATION_DATE = "2018-02-13 01:00"
VALIDATION_MAX_HEIGHT_KM = 0.0

SIGMA8_SI = 3.3e-130
O2_DENSITY_M3 = 5.13e24
PULSE_TAU_S = 98.6e-15
PAPER_CENTER_NE_M3 = 3.0e21

N2_TOTAL_M3 = 1.9117e25
O2_TOTAL_M3 = 5.13e24
AR_TOTAL_M3 = 2.2778e23

ATM_PA = 101325.0
R_GAS = 8.31446261815324
N_A = 6.02214076e23
MOLAR_MASS_KG_MOL = {
    "N2": 28.0134e-3,
    "O2": 31.9988e-3,
    "Ar": 39.948e-3,
}

SHARMA_EXPERIMENT_HALF_LIFE_NS = 2.5
SHARMA_SIMULATION_HALF_LIFE_NS = 3.2


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paper_center_density_m3():
    return PAPER_CENTER_NE_M3


def build_initial_condition():
    ne0_m3 = paper_center_density_m3()

    ion_m3 = {
        "N2": 0.0,
        "O2": ne0_m3,
        "Ar": 0.0,
    }
    neutral_m3 = {
        "N2": N2_TOTAL_M3,
        "O2": O2_TOTAL_M3 - ion_m3["O2"],
        "Ar": AR_TOTAL_M3,
    }
    neutral_total_m3 = sum(neutral_m3.values())
    fractions = {
        gas: neutral_m3[gas] / neutral_total_m3
        for gas in ("N2", "O2", "Ar")
    }
    molar_mass = sum(
        fractions[gas] * MOLAR_MASS_KG_MOL[gas]
        for gas in ("N2", "O2", "Ar")
    )
    rho = ATM_PA * molar_mass / (R_GAS * TG0_K)

    initial = pd.DataFrame(
        [
            {
                "case_id": "sharma_figure4_same_initial",
                "case_type": "paper_macroscopic_and_primary_plasma_initials",
                "I0_W_cm2": I0_W_CM2,
                "date": "2018-02-13 01:00",
                "H_km": 0.0,
                "P_atm": 1.0,
                "Tg0_K": TG0_K,
                "Tv0_K": TV0_K,
                "Te0_K": TE0_K,
                "Te0_eV": TE0_EV,
                "Te0_source": "Sharma_Figure_4_graphical_initial_value",
                "ne0_m3": ne0_m3,
                "ne0_cm3": ne0_m3 / 1.0e6,
                "n_N2_plus0_m3": ion_m3["N2"],
                "n_N2_plus0_cm3": ion_m3["N2"] / 1.0e6,
                "n_O2_plus0_m3": ion_m3["O2"],
                "n_O2_plus0_cm3": ion_m3["O2"] / 1.0e6,
                "n_Ar_plus0_m3": ion_m3["Ar"],
                "n_Ar_plus0_cm3": ion_m3["Ar"] / 1.0e6,
                "n_O4_plus0_m3": 0.0,
                "n_O4_plus0_cm3": 0.0,
                "n_O2_minus0_m3": 0.0,
                "n_O2_minus0_cm3": 0.0,
                "n_O0_m3": 0.0,
                "n_O0_cm3": 0.0,
                "neutral_N2_0_m3": neutral_m3["N2"],
                "neutral_N2_0_cm3": neutral_m3["N2"] / 1.0e6,
                "neutral_O2_0_m3": neutral_m3["O2"],
                "neutral_O2_0_cm3": neutral_m3["O2"] / 1.0e6,
                "neutral_Ar_0_m3": neutral_m3["Ar"],
                "neutral_Ar_0_cm3": neutral_m3["Ar"] / 1.0e6,
                "neutral_total_0_m3": neutral_total_m3,
                "neutral_total_0_cm3": neutral_total_m3 / 1.0e6,
                "x_N2_0": fractions["N2"],
                "x_O2_0": fractions["O2"],
                "x_Ar_0": fractions["Ar"],
                "Mmix_kg_mol": molar_mass,
                "MWMSA_kg": molar_mass / N_A,
                "rho0_kg_m3": rho,
            }
        ]
    )
    initial.to_csv(INITIAL_CSV, index=False, encoding="utf-8-sig")
    return initial


def half_life_ns(time_ns, values):
    time_ns = np.asarray(time_ns, dtype=float)
    values = np.asarray(values, dtype=float)
    target = 0.5 * values[0]
    below = np.flatnonzero(values <= target)
    if len(below) == 0:
        return math.nan
    index = int(below[0])
    if index == 0:
        return float(time_ns[0])
    t0, t1 = time_ns[index - 1], time_ns[index]
    y0, y1 = values[index - 1], values[index]
    return float(t0 + (target - y0) * (t1 - t0) / (y1 - y0))


def crossing_time_ns(time_ns, values, threshold):
    time_ns = np.asarray(time_ns, dtype=float)
    values = np.asarray(values, dtype=float)
    indices = np.flatnonzero(values <= float(threshold))
    if len(indices) == 0:
        return math.nan
    index = int(indices[0])
    if index == 0:
        return float(time_ns[0])
    t0, t1 = time_ns[index - 1], time_ns[index]
    y0, y1 = values[index - 1], values[index]
    if y1 == y0:
        return float(t1)
    return float(t0 + (threshold - y0) * (t1 - t0) / (y1 - y0))


def output_paths(relaxation, row):
    prefix = relaxation.output_case_prefix(row["date"], row["H_km"])
    return {
        "particle": OUTPUT_DIR / "particle_density" / f"{prefix}-n.csv",
        "te": OUTPUT_DIR / "temperature" / f"{prefix}-Te.csv",
        "tv": OUTPUT_DIR / "temperature" / f"{prefix}-Tv.csv",
        "gas": OUTPUT_DIR / "temperature" / f"{prefix}-T.csv",
    }


def summarize(relaxation, initial):
    row = initial.iloc[0]
    paths = output_paths(relaxation, row)
    particle = pd.read_csv(paths["particle"])
    te = pd.read_csv(paths["te"])
    tv = pd.read_csv(paths["tv"])
    gas = pd.read_csv(paths["gas"])

    time_ns = particle["t_ns"].to_numpy(dtype=float)
    ne = particle["ne_cm3"].to_numpy(dtype=float)
    te_ev = te["Te_eV"].to_numpy(dtype=float)
    tv_k = tv["Tv_K"].to_numpy(dtype=float)
    gas_k = gas["T_K"].to_numpy(dtype=float)

    three_temperatures = pd.DataFrame(
        {
            "t_ns": time_ns,
            "Te_K": te["Te_K"].to_numpy(dtype=float),
            "Tv_K": tv_k,
            "T_K": gas_k,
        }
    )
    three_temperatures.to_csv(
        THREE_TEMPERATURES_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    comparison = pd.DataFrame(
        [
            {
                "comparison_type": "same_identifiable_initial_conditions",
                "I0_W_cm2": I0_W_CM2,
                "model_ne0_cm3": ne[0],
                "model_ne_12ns_cm3": ne[-1],
                "model_ne_retention_12ns": ne[-1] / ne[0],
                "model_half_life_ns": half_life_ns(time_ns, ne),
                "Sharma_experiment_half_life_ns": (
                    SHARMA_EXPERIMENT_HALF_LIFE_NS
                ),
                "Sharma_simulation_half_life_ns": (
                    SHARMA_SIMULATION_HALF_LIFE_NS
                ),
                "model_Te0_eV": te_ev[0],
                "model_Te_at_0p5ns_eV": float(
                    np.interp(0.5, time_ns, te_ev)
                ),
                "model_Te_at_1ns_eV": float(np.interp(1.0, time_ns, te_ev)),
                "model_time_to_Te_0p4eV_ns": crossing_time_ns(
                    time_ns,
                    te_ev,
                    0.4,
                ),
                "model_Te_at_12ns_eV": te_ev[-1],
                "model_Tv0_K": tv_k[0],
                "model_Tv_peak_K": float(np.max(tv_k)),
                "model_Tv_peak_time_ns": float(time_ns[int(np.argmax(tv_k))]),
                "model_Tv_at_12ns_K": tv_k[-1],
                "model_T0_K": gas_k[0],
                "model_T_peak_K": float(np.max(gas_k)),
                "model_T_at_12ns_K": gas_k[-1],
            }
        ]
    )
    comparison.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")

    normalized = pd.DataFrame(
        {
            "t_ns": time_ns,
            "model_ne_over_ne0": ne / ne[0],
            "Sharma_experiment_2p5ns_reference": (
                2.0 ** (-time_ns / SHARMA_EXPERIMENT_HALF_LIFE_NS)
            ),
            "Sharma_simulation_3p2ns_reference": (
                2.0 ** (-time_ns / SHARMA_SIMULATION_HALF_LIFE_NS)
            ),
            "model_Te_eV": te_ev,
            "model_Tv_eV": tv_k * EV_PER_K,
            "model_T_eV": gas_k * EV_PER_K,
        }
    )
    normalized.to_csv(NORMALIZED_CSV, index=False, encoding="utf-8-sig")
    return comparison


def write_manifest(relaxation, initial):
    model_hash = sha256(MODEL_PATH)
    source_hash = sha256(SOURCE_PATH)
    row = initial.iloc[0]
    records = [
        {
            "record_type": "model_hash",
            "name": "chapter3_model",
            "value": model_hash,
            "source": str(MODEL_PATH),
        },
        {
            "record_type": "source_hash",
            "name": "original_chapter3_model",
            "value": source_hash,
            "source": str(SOURCE_PATH),
        },
        {
            "record_type": "model_unchanged",
            "name": "hashes_identical",
            "value": model_hash == source_hash,
            "source": "SHA256",
        },
    ]
    inputs = {
        "I0_W_cm2": I0_W_CM2,
        "Tg0_K": TG0_K,
        "Tv0_K": TV0_K,
        "Te0_eV": TE0_EV,
        "Te0_K": TE0_K,
        "ne0_cm3": row["ne0_cm3"],
        "O2_plus0_cm3": row["n_O2_plus0_cm3"],
        "N2_plus0_cm3": row["n_N2_plus0_cm3"],
        "Ar_plus0_cm3": row["n_Ar_plus0_cm3"],
        "time_end_ns": T_END_NS,
        "time_step_ns_runtime_input": relaxation.DT_NS,
        "run_date": VALIDATION_DATE,
        "max_height_km": VALIDATION_MAX_HEIGHT_KM,
    }
    for name, value in inputs.items():
        records.append(
            {
                "record_type": "runtime_input",
                "name": name,
                "value": value,
                "source": "Sharma Figure 4 or current model",
            }
        )
    pd.DataFrame(records).to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if sha256(MODEL_PATH) != sha256(SOURCE_PATH):
        raise ValueError("The validation model differs from the original source model")

    relaxation = load_module("chapter3_same_initial_model", MODEL_PATH)
    initial = build_initial_condition()

    relaxation.INIT_CSV = INITIAL_CSV
    relaxation.OUT_DIR = OUTPUT_DIR
    relaxation.T_END_NS = T_END_NS
    relaxation.DT_NS = VALIDATION_DT_NS
    relaxation.SAVE_EVERY = SAVE_EVERY
    relaxation.RUN_DATE = VALIDATION_DATE
    relaxation.RUN_MAX_HEIGHT_KM = VALIDATION_MAX_HEIGHT_KM

    relaxation.run_all_cases()
    comparison = summarize(relaxation, initial)
    write_manifest(relaxation, initial)

    print(comparison.to_string(index=False))
    print(f"Initial condition: {INITIAL_CSV}")
    print(f"Comparison: {COMPARISON_CSV}")
    print(f"Normalized decay: {NORMALIZED_CSV}")
    print(f"Three temperatures (K): {THREE_TEMPERATURES_CSV}")
    print(f"Manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
