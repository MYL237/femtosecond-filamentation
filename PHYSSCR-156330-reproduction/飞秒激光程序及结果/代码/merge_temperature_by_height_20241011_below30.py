from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = (
    BASE_DIR
    / "critical_outputs"
    / "relax_outputs_Te11000K_full_radial_pde_20260728"
    / "temperature"
)
OUTPUT_DIR = (
    BASE_DIR
    / "critical_outputs"
    / "relax_outputs_Te11000K_full_radial_pde_20260728"
    / "temperature_by_height_2024-10-11-1_below30km"
)

DATE_PREFIX = "2024-10-11-1"
MAX_HEIGHT_KM = 30.0
INCLUDE_MAX_HEIGHT = True

TEMPERATURE_SPECS = {
    "Te": {
        "suffix": "Te",
        "value_column": "Te_K",
        "output_name": "2024-10-11-1-Te_by_height_below30km.csv",
    },
    "Tv": {
        "suffix": "Tv",
        "value_column": "Tv_K",
        "output_name": "2024-10-11-1-Tv_by_height_below30km.csv",
    },
    "T": {
        "suffix": "T",
        "value_column": "T_K",
        "output_name": "2024-10-11-1-T_by_height_below30km.csv",
    },
}


def parse_height_km(path):
    stem = path.stem
    prefix = f"{DATE_PREFIX}-"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected file name: {path.name}")

    rest = stem[len(prefix) :]
    height_text = rest.split("-")[0]
    if not height_text.endswith("km"):
        raise ValueError(f"Cannot parse height from file name: {path.name}")
    return float(height_text[:-2].replace("p", "."))


def height_is_selected(height_km):
    if INCLUDE_MAX_HEIGHT:
        return height_km <= MAX_HEIGHT_KM
    return height_km < MAX_HEIGHT_KM


def format_height_column(height_km):
    if abs(height_km - round(height_km)) < 1.0e-9:
        return f"{int(round(height_km))}km"
    return f"{height_km:g}km".replace(".", "p")


def list_temperature_files(suffix):
    pattern = f"{DATE_PREFIX}-*km-{suffix}.csv"
    files = []
    for path in INPUT_DIR.glob(pattern):
        height_km = parse_height_km(path)
        if height_is_selected(height_km):
            files.append((height_km, path))
    return sorted(files, key=lambda item: item[0])


def read_temperature_series(path, value_column, height_km, reference_time=None):
    data = pd.read_csv(path, usecols=["t_ns", value_column])
    if reference_time is not None and not np.array_equal(
        data["t_ns"].to_numpy(), reference_time
    ):
        raise ValueError(f"Time grid does not match reference: {path.name}")

    height_column = format_height_column(height_km)
    return data.rename(columns={value_column: height_column})


def merge_one_temperature(spec):
    files = list_temperature_files(spec["suffix"])
    if not files:
        raise FileNotFoundError(
            f"No {DATE_PREFIX} files found for suffix {spec['suffix']} in {INPUT_DIR}"
        )

    merged = None
    reference_time = None
    for height_km, path in files:
        series = read_temperature_series(
            path,
            spec["value_column"],
            height_km,
            reference_time=reference_time,
        )
        if merged is None:
            merged = series
            reference_time = merged["t_ns"].to_numpy()
        else:
            merged[format_height_column(height_km)] = series.iloc[:, 1].to_numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / spec["output_name"]
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path, [format_height_column(height) for height, _ in files]


def main():
    outputs = []
    for name in ("Te", "Tv", "T"):
        output_path, height_columns = merge_one_temperature(TEMPERATURE_SPECS[name])
        outputs.append(output_path)
        print(f"{name}: {output_path}")
        print("heights:", ", ".join(height_columns))

    print("Done.")
    return outputs


if __name__ == "__main__":
    main()
