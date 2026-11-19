import argparse
import csv
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = BASE_DIR / "critical_outputs"
DEFAULT_TIME_DIR = DEFAULT_OUTPUTS_DIR / "time_height_csv"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUTS_DIR / "chapter3_initial_conditions_Te11000K.csv"

TE0_FIXED_K = 11000.0
CM3_PER_M3 = 1.0e6
EV_PER_K = 1.0 / 11604.51812
N_A = 6.02214076e23
R_GAS = 8.31446261815324
ATM_PA = 101325.0

MOLAR_MASS_KG_MOL = {
    "N2": 28.0134e-3,
    "O2": 31.9988e-3,
    "Ar": 39.948e-3,
}


# These are the only columns needed by the Chapter-3 relaxation code.
FIELDNAMES = [
    "date",
    "H_km",
    "P_atm",
    "Tg0_K",
    "Tv0_K",
    "Te0_K",
    "Te0_eV",
    "Te0_source",
    "ne0_m3",
    "ne0_cm3",
    "n_N2_plus0_m3",
    "n_N2_plus0_cm3",
    "n_O2_plus0_m3",
    "n_O2_plus0_cm3",
    "n_Ar_plus0_m3",
    "n_Ar_plus0_cm3",
    "n_O4_plus0_m3",
    "n_O4_plus0_cm3",
    "n_O2_minus0_m3",
    "n_O2_minus0_cm3",
    "n_O0_m3",
    "n_O0_cm3",
    "neutral_N2_0_m3",
    "neutral_N2_0_cm3",
    "neutral_O2_0_m3",
    "neutral_O2_0_cm3",
    "neutral_Ar_0_m3",
    "neutral_Ar_0_cm3",
    "neutral_total_0_m3",
    "neutral_total_0_cm3",
    "x_N2_0",
    "x_O2_0",
    "x_Ar_0",
    "Mmix_kg_mol",
    "MWMSA_kg",
    "rho0_kg_m3",
]


def parse_float(value, default=math.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def m3_to_cm3_density(value_m3):
    return value_m3 / CM3_PER_M3


def case_sort_key(csv_path):
    rows = read_csv_rows(csv_path)
    if not rows:
        return ("", float("inf"), csv_path.name)
    first = rows[0]
    return (
        first.get("date", ""),
        parse_float(first.get("H_km"), float("inf")),
        csv_path.name,
    )


def weighted_molar_mass(x_N2, x_O2, x_Ar):
    return (
        x_N2 * MOLAR_MASS_KG_MOL["N2"]
        + x_O2 * MOLAR_MASS_KG_MOL["O2"]
        + x_Ar * MOLAR_MASS_KG_MOL["Ar"]
    )


def build_initial_condition_row(csv_path):
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no rows")

    first = rows[0]
    final = rows[-1]

    ne0_m3 = parse_float(final["ne_total_m^-3"])
    n_N2_plus0_m3 = parse_float(final["ne_N2_m^-3"])
    n_O2_plus0_m3 = parse_float(final["ne_O2_m^-3"])
    n_Ar_plus0_m3 = parse_float(final["ne_Ar_m^-3"])

    neutral_N2_m3 = parse_float(final["neutral_N2_m^-3"])
    neutral_O2_m3 = parse_float(final["neutral_O2_m^-3"])
    neutral_Ar_m3 = parse_float(final["neutral_Ar_m^-3"])
    neutral_total_m3 = parse_float(final["neutral_total_m^-3"])

    if neutral_total_m3 > 0.0:
        x_N2 = neutral_N2_m3 / neutral_total_m3
        x_O2 = neutral_O2_m3 / neutral_total_m3
        x_Ar = neutral_Ar_m3 / neutral_total_m3
    else:
        x_N2 = x_O2 = x_Ar = math.nan

    Mmix = weighted_molar_mass(x_N2, x_O2, x_Ar)
    MWMSA = Mmix / N_A
    Tg0_K = parse_float(first["T_K"])
    P_atm = parse_float(first["P_atm"])
    rho0 = P_atm * ATM_PA * Mmix / (R_GAS * Tg0_K) if Tg0_K > 0.0 else math.nan

    return {
        "date": first["date"],
        "H_km": first["H_km"],
        "P_atm": P_atm,
        "Tg0_K": Tg0_K,
        "Tv0_K": Tg0_K,
        "Te0_K": TE0_FIXED_K,
        "Te0_eV": TE0_FIXED_K * EV_PER_K,
        "Te0_source": "fixed_11000_K",
        "ne0_m3": ne0_m3,
        "ne0_cm3": m3_to_cm3_density(ne0_m3),
        "n_N2_plus0_m3": n_N2_plus0_m3,
        "n_N2_plus0_cm3": m3_to_cm3_density(n_N2_plus0_m3),
        "n_O2_plus0_m3": n_O2_plus0_m3,
        "n_O2_plus0_cm3": m3_to_cm3_density(n_O2_plus0_m3),
        "n_Ar_plus0_m3": n_Ar_plus0_m3,
        "n_Ar_plus0_cm3": m3_to_cm3_density(n_Ar_plus0_m3),
        "n_O4_plus0_m3": 0.0,
        "n_O4_plus0_cm3": 0.0,
        "n_O2_minus0_m3": 0.0,
        "n_O2_minus0_cm3": 0.0,
        "n_O0_m3": 0.0,
        "n_O0_cm3": 0.0,
        "neutral_N2_0_m3": neutral_N2_m3,
        "neutral_N2_0_cm3": m3_to_cm3_density(neutral_N2_m3),
        "neutral_O2_0_m3": neutral_O2_m3,
        "neutral_O2_0_cm3": m3_to_cm3_density(neutral_O2_m3),
        "neutral_Ar_0_m3": neutral_Ar_m3,
        "neutral_Ar_0_cm3": m3_to_cm3_density(neutral_Ar_m3),
        "neutral_total_0_m3": neutral_total_m3,
        "neutral_total_0_cm3": m3_to_cm3_density(neutral_total_m3),
        "x_N2_0": x_N2,
        "x_O2_0": x_O2,
        "x_Ar_0": x_Ar,
        "Mmix_kg_mol": Mmix,
        "MWMSA_kg": MWMSA,
        "rho0_kg_m3": rho0,
    }


def write_initial_conditions(time_dir=DEFAULT_TIME_DIR, output_csv=DEFAULT_OUTPUT_CSV):
    time_dir = Path(time_dir)
    output_csv = Path(output_csv)

    case_files = sorted(time_dir.glob("*.csv"), key=case_sort_key)
    if not case_files:
        raise FileNotFoundError(f"No date-height CSV files found in {time_dir}")

    rows = [build_initial_condition_row(path) for path in case_files]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv, rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare fixed-Te0 Chapter-3 initial conditions only."
    )
    parser.add_argument("--time-dir", default=str(DEFAULT_TIME_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


def main():
    args = parse_args()
    output_csv, rows = write_initial_conditions(
        time_dir=args.time_dir,
        output_csv=args.output,
    )
    print(f"wrote {output_csv}")
    print(f"rows = {len(rows)}")
    print(f"columns = {len(FIELDNAMES)}")
    print(f"Te0 = {TE0_FIXED_K:.0f} K")


if __name__ == "__main__":
    main()
