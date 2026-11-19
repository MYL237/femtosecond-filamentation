import argparse
import csv
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = BASE_DIR / "critical_outputs" / "time_height_csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "critical_outputs" / "parameter_csv"


PARAMETER_ALIASES = {
    "I": "I_W_cm2",
    "I_cm2": "I_W_cm2",
    "I_m2": "I_W_m2",
    "W_N2": "W_N2_s^-1",
    "W_O2": "W_O2_s^-1",
    "W_Ar": "W_Ar_s^-1",
    "R_N2": "R_N2_m^-3_s^-1",
    "R_O2": "R_O2_m^-3_s^-1",
    "R_Ar": "R_Ar_m^-3_s^-1",
    "R_total": "R_total_m^-3_s^-1",
    "ne": "ne_total_m^-3",
    "ne_total": "ne_total_m^-3",
    "ne_N2": "ne_N2_m^-3",
    "ne_O2": "ne_O2_m^-3",
    "ne_Ar": "ne_Ar_m^-3",
    "neutral": "neutral_total_m^-3",
    "neutral_total": "neutral_total_m^-3",
    "n2": "n2_prime_m2_W",
    "n2I": "n2I",
    "Te": "Te_cum_eV",
    "Te_cum": "Te_cum_eV",
    "Te_inst": "Te_inst_eV",
    "Up": "Up_eV",
    "Ekin_N2": "Ekin_N2_eV",
    "Ekin_O2": "Ekin_O2_eV",
    "Ekin_Ar": "Ekin_Ar_eV",
}


AUTO_EXCLUDED_COLUMNS = {
    "date",
    "H_km",
    "T_K",
    "P_atm",
    "Icrit_W_m2",
    "Icrit_W_cm2",
    "t_s",
    "t_fs",
    "P_for_n2_atm",
    "n_e_cr_m^-3",
}


def safe_filename_token(text):
    text = str(text)
    text = text.replace("^", "")
    text = re.sub(r"[^0-9A-Za-z._+-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_.")


def resolve_parameter(name):
    return PARAMETER_ALIASES.get(name, name)


def case_column_name(first_row, csv_path):
    date = first_row.get("date", "").strip()
    height = first_row.get("H_km", "").strip()
    if date and height:
        date_token = date.replace(":", "_").replace(" ", "_")
        return f"{date_token}_H{height}km"
    return csv_path.stem


def case_sort_key(csv_path):
    """
    Sort by date first, then by numeric height.

    Filename sorting puts H10 before H5, so the order must come from
    the CSV metadata written by light-intensity calculation.
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)

    if not first_row:
        return (csv_path.stem, float("inf"), csv_path.name)

    date = first_row.get("date", "").strip()
    try:
        height = float(first_row.get("H_km", "inf"))
    except ValueError:
        height = float("inf")

    return (date, height, csv_path.name)


def sorted_case_csv_files(source_dir):
    return sorted(Path(source_dir).glob("*.csv"), key=case_sort_key)


def list_available_columns(source_dir):
    files = sorted_case_csv_files(source_dir)
    if not files:
        return []
    with files[0].open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def auto_parameter_columns(source_dir, varying_only=False):
    columns = [
        column
        for column in list_available_columns(source_dir)
        if column not in AUTO_EXCLUDED_COLUMNS
    ]
    if not varying_only:
        return columns

    source_dir = Path(source_dir)
    files = sorted_case_csv_files(source_dir)
    if not files:
        return []

    first_value = {column: None for column in columns}
    changed = {column: False for column in columns}

    for csv_path in files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for column in columns:
                    if changed[column]:
                        continue
                    value = row[column]
                    if first_value[column] is None:
                        first_value[column] = value
                    elif value != first_value[column]:
                        changed[column] = True
                if all(changed.values()):
                    return columns

    return [column for column in columns if changed[column]]


def read_case_series(csv_path, parameter):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header")
        if parameter not in reader.fieldnames:
            raise KeyError(
                f"{csv_path.name} does not contain column {parameter!r}"
            )
        if "t_fs" not in reader.fieldnames:
            raise KeyError(f"{csv_path.name} does not contain column 't_fs'")

        t_fs = []
        values = []
        first_row = None
        for row in reader:
            if first_row is None:
                first_row = row
            t_fs.append(row["t_fs"])
            values.append(row[parameter])

    if first_row is None:
        raise ValueError(f"{csv_path} has no data rows")

    return case_column_name(first_row, csv_path), t_fs, values


def write_parameter_csv(parameter_name, source_dir, output_dir):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter = resolve_parameter(parameter_name)
    files = sorted_case_csv_files(source_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    base_t_fs = None
    columns = []
    series_by_column = {}

    for csv_path in files:
        column_name, t_fs, values = read_case_series(csv_path, parameter)
        if base_t_fs is None:
            base_t_fs = t_fs
        elif len(t_fs) != len(base_t_fs) or t_fs != base_t_fs:
            raise ValueError(
                "Time grid mismatch between files. "
                f"First file has {len(base_t_fs)} rows; "
                f"{csv_path.name} has {len(t_fs)} rows."
            )
        columns.append(column_name)
        series_by_column[column_name] = values

    if parameter_name == parameter:
        output_name = f"{safe_filename_token(parameter)}_by_date_height.csv"
    else:
        output_name = (
            f"{safe_filename_token(parameter_name)}_"
            f"{safe_filename_token(parameter)}_by_date_height.csv"
        )
    output_path = output_dir / output_name

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_fs", *columns])
        for idx, t_value in enumerate(base_t_fs or []):
            writer.writerow(
                [t_value, *[series_by_column[column][idx] for column in columns]]
            )

    return output_path, len(columns), len(base_t_fs or [])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read parameters from all date-height time-series CSV files "
            "and merge each parameter into a new CSV whose columns are "
            "date-height cases. With no parameter arguments, all plottable "
            "time-series columns are exported."
        )
    )
    parser.add_argument(
        "parameters",
        nargs="*",
        default=[],
        help=(
            "Parameter aliases or exact CSV column names. "
            "Examples: ne, Te, Te_inst, n2, I, W_N2, R_total, ne_total_m^-3. "
            "If omitted, all plottable time-series columns are exported."
        ),
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Folder containing date-height CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder for merged parameter CSV files.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print available CSV columns and alias names, then exit.",
    )
    parser.add_argument(
        "--varying-only",
        action="store_true",
        help=(
            "When exporting all columns, keep only columns whose values "
            "actually change with time in the source files."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_columns:
        print("Available aliases:")
        for alias, column in sorted(PARAMETER_ALIASES.items()):
            print(f"  {alias} -> {column}")
        print("\nAvailable columns in source CSV:")
        for column in list_available_columns(args.source_dir):
            print(f"  {column}")
        print("\nAuto-export columns:")
        for column in auto_parameter_columns(args.source_dir, args.varying_only):
            print(f"  {column}")
        return

    parameters = args.parameters
    if not parameters:
        parameters = auto_parameter_columns(args.source_dir, args.varying_only)
        print(
            f"No parameter specified; exporting {len(parameters)} "
            "time-series columns."
        )

    for name in parameters:
        output_path, n_cases, n_rows = write_parameter_csv(
            name,
            args.source_dir,
            args.output_dir,
        )
        print(
            f"{name} -> {output_path} "
            f"({n_cases} date-height columns, {n_rows} time rows)"
        )


if __name__ == "__main__":
    main()
