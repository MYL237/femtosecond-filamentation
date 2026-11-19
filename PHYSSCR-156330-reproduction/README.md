# Numerical calculations for PHYSSCR-156330

Scripts accompanying *Effects of Atmospheric Stratification on Intensity Clamping in Femtosecond-Laser Filamentation and on Plasma-Channel Relaxation Characteristics*. Python 3.12 was used to check the package. Run the commands below from the extracted archive folder.

```sh
python -m pip install -r requirements.txt
```

## Atmospheric inputs and filamentation (Section 2)

`飞秒激光程序及结果/代码/光强计算.py` contains the 22 atmospheric cases from Table 1, the PPT/MO-PPT ionization calculation, and the clamped intensity and critical power calculations. It generates the altitude summaries and pulse time series underlying Section 2.

```sh
python "飞秒激光程序及结果/代码/光强计算.py"
python "飞秒激光程序及结果/代码/extract_parameter_timeseries.py" --list-columns
```

`读取GRIB.py` interpolates log pressure from ERA5 geopotential. The included GRIB has 37 pressure levels for 11 October 2024 at 01:00 UTC and reproduces that day's Table 1 pressure column at 39° N, 116° E. For the 27 October case, the Table 1 pressures are included in `光强计算.py`. To check the upstream pressure interpolation for that case, retrieve the geopotential field on all 37 levels for 27 October 2024 at 01:00 UTC separately.

```sh
python "飞秒激光程序及结果/代码/读取GRIB.py" "飞秒激光程序及结果/代码/inputs/era5_geopotential_20241011_0100.grib"
```

The GRIB contains Copernicus Climate Change Service information (2024). Dataset credit goes to the Copernicus Climate Change Service and ECMWF. See the [ERA5 hourly pressure-level dataset, DOI 10.24381/cds.bd0915c6](https://doi.org/10.24381/cds.bd0915c6) for attribution and licence terms. The NRLMSIS temperatures and neutral number densities used by the scripts are the values listed in Table 1.

## Atmospheric plasma-channel relaxation (Section 3)

`prepare_chapter3_initial_conditions.py` builds the initial-condition table from the Section 2 time series. The archived table is included at `critical_outputs/chapter3_initial_conditions_Te11000K.csv` in the main code directory, so the relaxation solver can also start directly from those values. The solver's default case selection is 11 October 2024, 0–30 km. The default simulation extends to 1000 ns. The three-temperature comparison in the manuscript discusses the first 100 ns. The archived initial-condition table also lists 32, 35 and 40 km. The corresponding high-altitude files used in the final plots could not be traced to an archived run in the folders used to assemble this package.

```sh
# Optional if using the included initial-condition table
python "飞秒激光程序及结果/代码/prepare_chapter3_initial_conditions.py"
python "飞秒激光程序及结果/代码/弛豫过程温度变化.py"
```

The two `merge_*_20241011_below30.py` scripts combine heat-transfer rates and temperature time series after the relaxation run. The full-resolution simulation CSVs and the final figure layout files are outside the scope of this code archive.

## Sharma comparison (Section 4)

`run_sharma_validation.py` computes the dry-air clamped-intensity case and also runs an auxiliary relaxation from its calculated ionization output. `run_sharma_chapter3_same_initial.py` runs the reported relaxation comparison using the separate 1 eV initial electron temperature. The latter uses the same relaxation solver. Both validation scripts check the hashes of the original model files against the identical copies in the validation directory. `reference_data/sharma_figure4_comparison.csv` contains the archived summary for the matched-initial-condition comparison.

```sh
python "验证/代码/run_sharma_validation.py"
python "验证/代码/run_sharma_chapter3_same_initial.py"
```

The atmospheric solver and the Sharma validation runs can generate large time-series files. The package checks covered imports, file paths, Table 1 inputs, the 11 October ERA5 pressure calculation, and selected values in the archived results.
