import argparse
from pathlib import Path

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description="Interpolate ERA5 geopotential pressure levels at the study location.")
parser.add_argument("grib_file", type=Path, help="ERA5 pressure-level GRIB file containing geopotential")
file_path = parser.parse_args().grib_file

# -----------------------------
# 读取 GRIB 文件
# -----------------------------
ds = xr.open_dataset(
    file_path,
    engine="cfgrib",
    backend_kwargs={"filter_by_keys": {"shortName": "z"}, "indexpath": ""},
)

# -----------------------------
# 1. 找到最接近 39N, 116E 的格点
# -----------------------------
lat_target = 39
lon_target = 116

# GRIB 的经度可能是 0~360，需转换
if ds.longitude.max() > 180:
    lon_target = (lon_target + 360) % 360

lat_idx = int(np.abs(ds.latitude.values - lat_target).argmin())
lon_idx = int(np.abs(ds.longitude.values - lon_target).argmin())

lat_real = float(ds.latitude[lat_idx])
lon_real = float(ds.longitude[lon_idx])

print(f"\n最近格点：lat={lat_real}, lon={lon_real}\n")

# -----------------------------
# 2. 提取该点全部 pressure level 的位势（z）
# -----------------------------
pressure_hPa = ds["isobaricInhPa"].values  # shape (37,)
z_point = ds["z"][:, lat_idx, lon_idx].values  # shape (37,)

# -----------------------------
# 3. 位势 z → 位势高度（km）
# -----------------------------
g = 9.80665
height_km = (z_point / g) / 1000.0  # 转 km

# -----------------------------
# 4. 按高度从低到高排序
# -----------------------------
order = np.argsort(height_km)
height_km = height_km[order]
pressure_hPa = pressure_hPa[order]

# -----------------------------
# 5. 压力 hPa → atm
# -----------------------------
pressure_atm = pressure_hPa / 1013.25  # 1 atm = 1013.25 hPa

# -----------------------------
# 6. 使用 log-pressure 进行插值
# -----------------------------
interp_func = interp1d(height_km, np.log(pressure_hPa), fill_value="extrapolate")

target_heights = np.array([0, 5,10,11,15, 20,25, 30,32,35, 40])  # km
pressure_interp_hPa = np.exp(interp_func(target_heights))
pressure_interp_atm = pressure_interp_hPa / 1013.25

# -----------------------------
# 7. 输出插值压力（核心结果）
# -----------------------------
print("===== 指定高度对应压力（插值，hPa & atm） =====\n")
for h, p_hPa, p_atm in zip(target_heights, pressure_interp_hPa, pressure_interp_atm):
    print(f"高度 {h:2d} km → {p_hPa:8.3f} hPa → {p_atm:8.5f} atm")

# -----------------------------
# 8. 输出全部层高度 vs 压力
# -----------------------------
print("\n===== 全部模式层高度-压力（hPa & atm） =====\n")
for h, p_hPa, p_atm in zip(height_km, pressure_hPa, pressure_atm):
    print(f"{h:6.2f} km → {p_hPa:7.2f} hPa → {p_atm:8.5f} atm")
