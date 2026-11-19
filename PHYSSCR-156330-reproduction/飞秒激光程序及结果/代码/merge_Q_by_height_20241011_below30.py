import csv
import re
from contextlib import ExitStack
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = (
    BASE_DIR
    / "critical_outputs"
    / "relax_outputs_Te11000K_full_radial_pde_20260728"
    / "intermediate_Q"
)
OUTPUT_DIR = (
    BASE_DIR
    / "critical_outputs"
    / "relax_outputs_Te11000K_full_radial_pde_20260728"
    / "Q_by_height_2024-10-11-1_below30km"
)

DATE_PREFIX = "2024-10-11-1"
MAX_HEIGHT_KM = 30.0
PROGRESS_EVERY_ROWS = 100_000

Q_SPECS = {
    "QeT": {
        "source_column": "QeT_J_m3_s",
        "output_name": "2024-10-11-1-QeT_by_height_0-30km.csv",
    },
    "QVT": {
        "source_column": "QVT_J_m3_s",
        "output_name": "2024-10-11-1-QVT_by_height_0-30km.csv",
    },
    "QR": {
        "source_column": "QR_J_m3_s",
        "output_name": "2024-10-11-1-QR_by_height_0-30km.csv",
    },
    "Qind": {
        "source_column": "Qind_J_m3_s",
        "output_name": "2024-10-11-1-Qind_by_height_0-30km.csv",
    },
    "Qtotal": {
        "source_column": "Q_total_to_gas_J_m3_s",
        "output_name": "2024-10-11-1-Qtotal_by_height_0-30km.csv",
    },
}

FILE_PATTERN = re.compile(
    rf"^{re.escape(DATE_PREFIX)}-(?P<height>[0-9]+(?:p[0-9]+)?)km-Q\.csv$"
)


def parse_height_km(path):
    match = FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Cannot parse height from file name: {path.name}")
    return float(match.group("height").replace("p", "."))


def format_height_column(height_km):
    if abs(height_km - round(height_km)) < 1.0e-9:
        return f"{int(round(height_km))}km"
    return f"{height_km:g}km".replace(".", "p")


def selected_input_files():
    files = []
    for path in INPUT_DIR.glob(f"{DATE_PREFIX}-*km-Q.csv"):
        height_km = parse_height_km(path)
        if height_km <= MAX_HEIGHT_KM:
            files.append((height_km, path))

    files.sort(key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(
            f"No Q files found for {DATE_PREFIX} at or below "
            f"{MAX_HEIGHT_KM:g} km in {INPUT_DIR}"
        )
    return files


def validate_columns(reader, path):
    required = {"t_ns"}
    required.update(spec["source_column"] for spec in Q_SPECS.values())
    missing = sorted(required.difference(reader.fieldnames or []))
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {', '.join(missing)}")


def next_rows(readers, paths, row_number):
    rows = []
    ended = []
    for reader in readers:
        try:
            rows.append(next(reader))
            ended.append(False)
        except StopIteration:
            rows.append(None)
            ended.append(True)

    if all(ended):
        return None
    if any(ended):
        ended_names = [
            paths[index].name
            for index, is_ended in enumerate(ended)
            if is_ended
        ]
        raise ValueError(
            f"Input files have different row counts near data row {row_number}: "
            f"{', '.join(ended_names)} ended first"
        )
    return rows


def merge_q_files():
    selected = selected_input_files()
    heights = [height for height, _ in selected]
    paths = [path for _, path in selected]
    height_columns = [format_height_column(height) for height in heights]
    output_header = ["t_ns", *height_columns]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: OUTPUT_DIR / f".{spec['output_name']}.tmp"
        for name, spec in Q_SPECS.items()
    }
    output_paths = {
        name: OUTPUT_DIR / spec["output_name"]
        for name, spec in Q_SPECS.items()
    }

    try:
        with ExitStack() as stack:
            input_streams = [
                stack.enter_context(path.open("r", encoding="utf-8-sig", newline=""))
                for path in paths
            ]
            readers = [csv.DictReader(stream) for stream in input_streams]
            for reader, path in zip(readers, paths):
                validate_columns(reader, path)

            output_streams = {
                name: stack.enter_context(
                    temporary_paths[name].open(
                        "w",
                        encoding="utf-8-sig",
                        newline="",
                    )
                )
                for name in Q_SPECS
            }
            writers = {
                name: csv.writer(stream)
                for name, stream in output_streams.items()
            }
            for writer in writers.values():
                writer.writerow(output_header)

            row_number = 1
            while True:
                rows = next_rows(readers, paths, row_number)
                if rows is None:
                    break

                reference_time = rows[0]["t_ns"]
                for row, path in zip(rows[1:], paths[1:]):
                    if row["t_ns"] != reference_time:
                        raise ValueError(
                            f"Time grid mismatch at data row {row_number}: "
                            f"{paths[0].name} has {reference_time}, "
                            f"{path.name} has {row['t_ns']}"
                        )

                for name, spec in Q_SPECS.items():
                    value_column = spec["source_column"]
                    writers[name].writerow(
                        [reference_time, *[row[value_column] for row in rows]]
                    )

                if row_number % PROGRESS_EVERY_ROWS == 0:
                    print(
                        f"Merged {row_number:,} rows; "
                        f"current time = {reference_time} ns",
                        flush=True,
                    )
                row_number += 1

        for name in Q_SPECS:
            temporary_paths[name].replace(output_paths[name])
    except Exception:
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()
        raise

    return output_paths, height_columns, row_number - 1


def main():
    outputs, height_columns, row_count = merge_q_files()
    print(f"Rows written per file: {row_count:,}")
    print("Height columns:", ", ".join(height_columns))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
