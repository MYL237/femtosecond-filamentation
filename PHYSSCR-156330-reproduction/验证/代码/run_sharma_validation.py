import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR.parent.parent / "飞秒激光程序及结果" / "代码"
OUTPUT_DIR = BASE_DIR / "validation_outputs" / "input_only_corrected"
CHAPTER2_DIR = OUTPUT_DIR / "chapter2_critical_condition"
CHAPTER3_DIR = OUTPUT_DIR / "chapter3_relaxation"

CRITICAL_SUMMARY_CSV = CHAPTER2_DIR / "chapter2_critical_summary.csv"
CRITICAL_TIME_CSV = CHAPTER2_DIR / "chapter2_critical_timeseries.csv"
ORDER_COMPARISON_CSV = CHAPTER2_DIR / "critical_intensity_order_comparison.csv"
CHAPTER3_INIT_CSV = OUTPUT_DIR / "chapter3_initial_condition_from_Icrit.csv"
CHAPTER3_COMPARE_CSV = OUTPUT_DIR / "chapter3_Icrit_relaxation_comparison.csv"
DECAY_CSV = OUTPUT_DIR / "chapter3_Icrit_normalized_decay.csv"
MANIFEST_CSV = OUTPUT_DIR / "validation_manifest.csv"

PAPER_FWHM_S = 164e-15
PAPER_TG_K = 300.0
PAPER_TE0_K = 11000.0
PAPER_TAU_S = 98.6e-15
PAPER_SIGMA8_SI = 3.3e-130
PAPER_O2_DENSITY_M3 = 5.13e24
PAPER_REFERENCE_INTENSITY_W_CM2 = 2.70e13
PAPER_DENSITY_REFERENCE_INTENSITY_W_CM2 = 2.68e13
PAPER_EXPERIMENT_HALF_LIFE_NS = 2.5
PAPER_SIMULATION_HALF_LIFE_NS = 3.2

N_A = 6.02214076e23
R_GAS = 8.31446261815324
ATM_PA = 101325.0
MOLAR_MASS_KG_MOL = {
    "N2": 28.0134e-3,
    "O2": 31.9988e-3,
    "Ar": 39.948e-3,
}


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


def verify_model_copies_unchanged():
    rows = []
    for filename in ("光强计算.py", "弛豫过程温度变化.py"):
        source = SOURCE_DIR / filename
        validation_copy = BASE_DIR / filename
        source_hash = sha256(source)
        copy_hash = sha256(validation_copy)
        if source_hash != copy_hash:
            raise ValueError(f"{filename} differs from the original source code")
        rows.append(
            {
                "model_file": filename,
                "source_path": str(source),
                "validation_copy_path": str(validation_copy),
                "source_sha256": source_hash,
                "validation_copy_sha256": copy_hash,
                "identical": True,
            }
        )
    return rows


def sharma_air_case():
    return {
        "date": "2018-02-13 01:00",
        "H_km": 0.0,
        "T_K": PAPER_TG_K,
        "P_atm_table": 1.0,
        "N2_m3": 1.9117e25,
        "O2_m3": 5.13e24,
        "Ar_m3": 2.2778e23,
    }


def apply_chapter2_inputs(ionization):
    """Apply experimental inputs without editing the model source file."""
    ionization.PULSE_FWHM_S = PAPER_FWHM_S
    ionization.PULSE_CENTER_S = (
        0.5 * ionization.PULSE_WINDOW_FWHM * ionization.PULSE_FWHM_S
    )
    ionization.T_END_PULSE_S = (
        ionization.PULSE_WINDOW_FWHM * ionization.PULSE_FWHM_S
    )
    ionization.T_GRID_S, ionization.PULSE_SHAPE = ionization.build_time_grid()
    ionization.IDX_PEAK = int(np.argmax(ionization.PULSE_SHAPE))


def experimental_center_density_reference_m3():
    intensity_si = PAPER_DENSITY_REFERENCE_INTENSITY_W_CM2 * 1.0e4
    rate = PAPER_SIGMA8_SI * intensity_si**8
    return (
        PAPER_O2_DENSITY_M3
        * rate
        * PAPER_TAU_S
        * math.sqrt(math.pi / 8.0)
    )


def run_chapter2_critical_condition(ionization, case):
    CHAPTER2_DIR.mkdir(parents=True, exist_ok=True)
    print("Chapter 2: solving the original critical-intensity equation", flush=True)
    critical_intensity, summary, rows = ionization.find_critical_intensity_trial(
        case=case,
        lambda_m=ionization.LAMBDA_M,
        I_min=ionization.I_SEARCH_MIN,
        I_max=ionization.I_SEARCH_MAX,
    )
    summary["timeseries_csv"] = str(CRITICAL_TIME_CSV)
    ionization.write_timeseries_csv(CRITICAL_TIME_CSV, rows, write_header=True)
    ionization.write_critical_summary_csv(CRITICAL_SUMMARY_CSV, [summary])

    model_i_cm2 = critical_intensity / 1.0e4
    reference_density_cm3 = experimental_center_density_reference_m3() / 1.0e6
    intensity_ratio = model_i_cm2 / PAPER_REFERENCE_INTENSITY_W_CM2
    comparison = pd.DataFrame(
        [
            {
                "comparison_type": "order_of_magnitude_only",
                "model_Icrit_W_m2": critical_intensity,
                "model_Icrit_W_cm2": model_i_cm2,
                "Sharma_reference_intensity_W_cm2": (
                    PAPER_REFERENCE_INTENSITY_W_CM2
                ),
                "model_over_reference_intensity": intensity_ratio,
                "absolute_log10_intensity_ratio": abs(math.log10(intensity_ratio)),
                "same_order_of_magnitude": abs(math.log10(intensity_ratio)) < 1.0,
                "model_ne_end_at_Icrit_cm3": summary["final_ne_total_cm-3"],
                "Sharma_center_ne_at_2p68e13_cm3": reference_density_cm3,
                "density_ratio_different_intensities": (
                    summary["final_ne_total_cm-3"] / reference_density_cm3
                ),
                "model_Te_end_at_Icrit_eV": summary["Te0_final_eV"],
                "model_Te_end_at_Icrit_K": summary["Te0_final_K"],
                "direct_Icrit_measurement_in_Sharmapaper": False,
            }
        ]
    )
    comparison.to_csv(ORDER_COMPARISON_CSV, index=False, encoding="utf-8-sig")
    return critical_intensity, summary


def build_chapter3_initial_condition(case, critical_summary):
    ion_m3 = {
        "N2": critical_summary["final_ne_N2_m-3"],
        "O2": critical_summary["final_ne_O2_m-3"],
        "Ar": critical_summary["final_ne_Ar_m-3"],
    }
    neutral_m3 = {
        gas: float(case[f"{gas}_m3"]) - ion_m3[gas]
        for gas in ("N2", "O2", "Ar")
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
    rho = ATM_PA * molar_mass / (R_GAS * PAPER_TG_K)
    ne0_m3 = sum(ion_m3.values())

    initial = pd.DataFrame(
        [
            {
                "case_id": "chapter2_critical_condition",
                "case_type": "original_chapter2_critical_output",
                "I0_W_cm2": critical_summary["I_peak_W_cm2"],
                "date": "2018-02-13 01:00",
                "H_km": 0.0,
                "P_atm": 1.0,
                "Tg0_K": PAPER_TG_K,
                "Tv0_K": PAPER_TG_K,
                "Te0_K": PAPER_TE0_K,
                "Te0_eV": PAPER_TE0_K / 11604.51812,
                "Te0_source": "original_model_fixed_11000_K",
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
    initial.to_csv(CHAPTER3_INIT_CSV, index=False, encoding="utf-8-sig")
    return initial


def apply_chapter3_inputs(relaxation):
    """Apply validation paths and time window without editing the model source."""
    relaxation.INIT_CSV = CHAPTER3_INIT_CSV
    relaxation.OUT_DIR = CHAPTER3_DIR
    relaxation.T_END_NS = 12.0
    relaxation.SAVE_EVERY = 10


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
    if y1 == y0:
        return float(t1)
    return float(t0 + (target - y0) * (t1 - t0) / (y1 - y0))


def first_crossing_time_ns(time_ns, values, threshold):
    below = np.flatnonzero(np.asarray(values, dtype=float) <= float(threshold))
    return float(time_ns[int(below[0])]) if len(below) else math.nan


def summarize_chapter3(relaxation, initial):
    row = next(initial.itertuples(index=False))
    prefix = relaxation.output_case_prefix(row.date, row.H_km)
    particle_path = CHAPTER3_DIR / "particle_density" / f"{prefix}-n.csv"
    te_path = CHAPTER3_DIR / "temperature" / f"{prefix}-Te.csv"
    tv_path = CHAPTER3_DIR / "temperature" / f"{prefix}-Tv.csv"
    gas_path = CHAPTER3_DIR / "temperature" / f"{prefix}-T.csv"

    particle = pd.read_csv(particle_path)
    te = pd.read_csv(te_path)
    tv = pd.read_csv(tv_path)
    gas = pd.read_csv(gas_path)
    time_ns = particle["t_ns"].to_numpy(dtype=float)
    ne = particle["ne_cm3"].to_numpy(dtype=float)
    normalized = ne / ne[0]
    model_half_life = half_life_ns(time_ns, ne)

    comparison = pd.DataFrame(
        [
            {
                "comparison_type": "trend_and_timescale_at_model_Icrit",
                "model_Icrit_W_cm2": float(row.I0_W_cm2),
                "Sharma_reference_intensity_W_cm2": (
                    PAPER_REFERENCE_INTENSITY_W_CM2
                ),
                "model_ne0_cm3": ne[0],
                "model_ne_12ns_cm3": ne[-1],
                "model_retention_12ns": normalized[-1],
                "model_half_life_ns": model_half_life,
                "Sharma_experiment_half_life_ns_at_2p68e13": (
                    PAPER_EXPERIMENT_HALF_LIFE_NS
                ),
                "Sharma_simulation_half_life_ns_at_2p70e13": (
                    PAPER_SIMULATION_HALF_LIFE_NS
                ),
                "model_Te0_eV": float(te.iloc[0]["Te_eV"]),
                "model_Te_at_1ns_eV": float(
                    np.interp(1.0, te["t_ns"], te["Te_eV"])
                ),
                "model_time_to_Te_0p4eV_ns": first_crossing_time_ns(
                    te["t_ns"],
                    te["Te_eV"],
                    0.4,
                ),
                "model_Te_at_12ns_eV": float(te.iloc[-1]["Te_eV"]),
                "model_Tv_at_12ns_K": float(tv.iloc[-1]["Tv_K"]),
                "model_T_at_12ns_K": float(gas.iloc[-1]["T_K"]),
                "direct_same_intensity_comparison": math.isclose(
                    float(row.I0_W_cm2),
                    PAPER_REFERENCE_INTENSITY_W_CM2,
                    rel_tol=0.05,
                ),
            }
        ]
    )
    comparison.to_csv(CHAPTER3_COMPARE_CSV, index=False, encoding="utf-8-sig")

    decay = pd.DataFrame(
        {
            "t_ns": time_ns,
            "model_at_Icrit_ne_over_ne0": normalized,
            "Sharma_experiment_2p5ns_reference": 2.0
            ** (-time_ns / PAPER_EXPERIMENT_HALF_LIFE_NS),
            "Sharma_simulation_3p2ns_reference": 2.0
            ** (-time_ns / PAPER_SIMULATION_HALF_LIFE_NS),
        }
    )
    decay.to_csv(DECAY_CSV, index=False, encoding="utf-8-sig")


def write_manifest(model_rows, ionization, critical_intensity):
    rows = []
    for row in model_rows:
        rows.append({"record_type": "model_hash", **row})
    inputs = [
        ("wavelength_m_unchanged", ionization.LAMBDA_M),
        ("pulse_fwhm_s_runtime_input", PAPER_FWHM_S),
        ("balance_tau_s_unchanged", ionization.TAU_BALANCE_S),
        ("gas_temperature_K", PAPER_TG_K),
        ("pressure_atm", 1.0),
        ("N2_m3", 1.9117e25),
        ("O2_m3", 5.13e24),
        ("Ar_m3", 2.2778e23),
        ("model_critical_intensity_W_m2", critical_intensity),
        ("chapter3_Te0_K_unchanged", PAPER_TE0_K),
        ("chapter3_end_ns_runtime_input", 12.0),
    ]
    for name, value in inputs:
        rows.append(
            {
                "record_type": "runtime_input",
                "model_file": "",
                "source_path": "",
                "validation_copy_path": "",
                "source_sha256": "",
                "validation_copy_sha256": "",
                "identical": "",
                "input_name": name,
                "input_value": value,
            }
        )
    pd.DataFrame(rows).to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_rows = verify_model_copies_unchanged()
    ionization = load_module("chapter2_model", BASE_DIR / "光强计算.py")
    relaxation = load_module("chapter3_model", BASE_DIR / "弛豫过程温度变化.py")

    apply_chapter2_inputs(ionization)
    case = sharma_air_case()
    critical_intensity, critical_summary = run_chapter2_critical_condition(
        ionization,
        case,
    )

    initial = build_chapter3_initial_condition(case, critical_summary)
    apply_chapter3_inputs(relaxation)
    print("Chapter 3: running from the original Chapter-2 Icrit output", flush=True)
    relaxation.run_all_cases()
    summarize_chapter3(relaxation, initial)
    write_manifest(model_rows, ionization, critical_intensity)

    print(f"Critical summary: {CRITICAL_SUMMARY_CSV}")
    print(f"Critical time series: {CRITICAL_TIME_CSV}")
    print(f"Order comparison: {ORDER_COMPARISON_CSV}")
    print(f"Chapter-3 initial condition: {CHAPTER3_INIT_CSV}")
    print(f"Chapter-3 comparison: {CHAPTER3_COMPARE_CSV}")
    print(f"Validation manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
