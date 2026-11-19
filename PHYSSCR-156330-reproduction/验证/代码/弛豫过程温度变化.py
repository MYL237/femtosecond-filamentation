import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INIT_CSV = BASE_DIR / "critical_outputs" / "chapter3_initial_conditions_Te11000K.csv"
OUT_DIR = BASE_DIR / "critical_outputs" / "relax_outputs_Te11000K_full_radial_pde_20260728"

TE0_FIXED_K = 11000.0
T_END_NS = 1000.0
DT_NS = 0.001
SAVE_EVERY = 1
RUN_DATE = "2024-10-11 01:00"
RUN_MAX_HEIGHT_KM = 30.0
FLOW_MODEL = "axisymmetric_radial_conservative_ns_electron_vibration_pde"
HYDRO_NR = 81
FILAMENT_SIGMA_M = 0.5e-3
HYDRO_R_MAX_FACTOR = 5.0
GAS_THERMAL_CONDUCTIVITY_W_M_K = 0.025
GAS_DYNAMIC_VISCOSITY_PA_S = 1.5e-5
GAS_CV_J_KG_K = 710.0

CM3_PER_M3 = 1.0e6
EV_PER_K = 1.0 / 11604.51812
J_PER_EV = 1.602176634e-19
K_B = 1.380649e-23
M_E = 9.1093837015e-31
R_GAS = 8.31446261815324

MOLAR_MASS_KG_MOL = {
    "N2": 28.0134e-3,
    "O2": 31.9988e-3,
    "Ar": 39.948e-3,
}

IDX = {
    "ne": 0,
    "O2p": 1,
    "O": 2,
    "O2": 3,
    "O2m": 4,
    "O4p": 5,
    "N2": 6,
    "Ar": 7,
    "N2p": 8,
    "Arp": 9,
}

SPECIES_COLUMNS = [
    "ne_cm3",
    "n_O2p_cm3",
    "n_O_cm3",
    "n_O2_cm3",
    "n_O2m_cm3",
    "n_O4p_cm3",
    "n_N2_cm3",
    "n_Ar_cm3",
    "n_N2p_cm3",
    "n_Arp_cm3",
]

TEMPERATURE_COLUMNS = ["Te_K", "Tv_K", "T_K"]
Q_COLUMNS = [
    "QeT_J_m3_s",
    "QVT_J_m3_s",
    "QR_J_m3_s",
    "Qind_J_m3_s",
    "Q_total_to_gas_J_m3_s",
    "Qev_J_m3_s",
]
FREQUENCY_COLUMNS = ["nu_en_s", "nu_ei_s", "nu_ev_s"]

TE_EV_FLOOR = 1.0e-3
TEMP_FLOOR_K = 1.0
NE_FLOOR_M3 = 1.0e12
EPS = 1.0e-300

KVT_N2 = 2.0e-30
KVT_O2 = 5.0e-31
KVT_AR = 2.0e-31
KVT_O = 2.0e-21
VIB_HV_EV = 0.29
ELECTRON_VIB_A_S = 8.917e10
ELECTRON_VIB_B_EV = 9.93
ELECTRON_VIB_EXP_EV3 = 2.36

DIRECT_REACTION_KEYS = ("r1", "r2", "r3")
INDIRECT_REACTION_KEYS = ("r4", "r5", "r6", "r7")
TABLE_REACTION_HEAT_EV = {
    "r1": 6.9,
    "r2": 0.45,
    "r3": 11.6,
    "r4": 0.43,
    "r5": 0.43,
    "r6": 3.51,
    "r7": 3.69,
}


def k_to_ev(temperature_k):
    values = np.asarray(temperature_k, dtype=float) * EV_PER_K
    return float(values) if values.ndim == 0 else values


def ev_to_k(temperature_ev):
    values = np.asarray(temperature_ev, dtype=float) / EV_PER_K
    return float(values) if values.ndim == 0 else values


def rate_constants_cm(Te_K):
    Te_eV = np.maximum(k_to_ev(Te_K), TE_EV_FLOOR)
    return {
        "k1": 2.1e-8 * Te_eV ** (-0.56),
        "k2": 3.64e-31 * Te_eV ** (-1.0) * np.exp(-0.052 / Te_eV),
        "k3": 3.4e-8 * Te_eV ** (-1.0),
        "k4": 9.0e-35 * Te_eV ** (-2.8),
        "k5": 3.3e-35 * Te_eV ** (-3.2),
        "k6": 5.2e-13 * Te_eV ** (-0.8),
        "k7": 5.1e-11,
    }


def reaction_rates_cm3_s(y, Te_K):
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    k = rate_constants_cm(Te_K)

    ne = y[IDX["ne"]]
    n_O2p = y[IDX["O2p"]]
    n_O2 = y[IDX["O2"]]
    n_O4p = y[IDX["O4p"]]
    n_N2 = y[IDX["N2"]]
    n_Ar = y[IDX["Ar"]]
    n_N2p = y[IDX["N2p"]]
    n_Arp = y[IDX["Arp"]]

    return {
        "r1": k["k1"] * ne * n_O2p,
        "r2": k["k2"] * ne * n_O2 * n_O2,
        "r3": k["k3"] * ne * n_O4p,
        "r4": k["k4"] * n_O2p * n_O2 * n_O2,
        "r5": k["k5"] * n_O2p * n_O2 * n_N2,
        "r6": k["k6"] * n_N2p * n_O2,
        "r7": k["k7"] * n_Arp * n_O2,
    }


def species_derivative_cm3_s(r):
    rate_shape = np.asarray(r["r1"], dtype=float).shape
    dy = np.zeros((len(IDX), *rate_shape), dtype=float)

    dy[IDX["ne"]] = -(r["r1"] + r["r2"] + r["r3"])
    dy[IDX["O2p"]] = -(r["r1"] + r["r4"] + r["r5"]) + r["r6"] + r["r7"]
    dy[IDX["O"]] = 2.0 * r["r1"]
    dy[IDX["O2"]] = (
        -r["r2"]
        + 2.0 * r["r3"]
        - r["r4"]
        - r["r5"]
        - r["r6"]
        - r["r7"]
    )
    dy[IDX["O2m"]] = r["r2"]
    dy[IDX["O4p"]] = -r["r3"] + r["r4"] + r["r5"]
    dy[IDX["N2"]] = r["r6"]
    dy[IDX["Ar"]] = r["r7"]
    dy[IDX["N2p"]] = -r["r6"]
    dy[IDX["Arp"]] = -r["r7"]

    return dy


def coulomb_log(ne_m3, Te_eV):
    ne_cm3 = np.maximum(np.asarray(ne_m3, dtype=float) / CM3_PER_M3, 1.0)
    Te = np.maximum(np.asarray(Te_eV, dtype=float), TE_EV_FLOOR)
    value = 23.0 - np.log(np.sqrt(ne_cm3) * Te ** (-1.5) + EPS)
    return np.maximum(value, 2.0)


def collision_frequencies_s(ne_m3, N_air_m3, Te_K):
    Te_eV = np.maximum(k_to_ev(Te_K), TE_EV_FLOOR)
    N_air = np.maximum(np.asarray(N_air_m3, dtype=float), 1.0)
    ne = np.maximum(np.asarray(ne_m3, dtype=float), 1.0)
    nu_en = 3.91e-14 * N_air * np.sqrt(Te_eV)
    nu_ei = 2.91e-12 * ne * coulomb_log(ne, Te_eV) * Te_eV ** (-1.5)
    return nu_en, nu_ei, nu_en + nu_ei


def vibrational_energy_per_molecule_j(T_K):
    T_eV = np.maximum(k_to_ev(T_K), 1.0e-12)
    exponent = np.minimum(VIB_HV_EV / T_eV, 700.0)
    eps_eV = VIB_HV_EV / np.expm1(exponent)
    return eps_eV * J_PER_EV


def vibrational_temperature_from_energy_k(energy_j):
    eps_eV = np.maximum(np.asarray(energy_j, dtype=float) / J_PER_EV, 1.0e-300)
    T_eV = VIB_HV_EV / np.log1p(VIB_HV_EV / eps_eV)
    return np.maximum(ev_to_k(T_eV), TEMP_FLOOR_K)


def vibrational_relaxation_rate_s(n_N2_m3, n_O2_m3, n_Ar_m3, n_O_m3):
    return (
        n_N2_m3 * KVT_N2
        + n_O2_m3 * KVT_O2
        + n_Ar_m3 * KVT_AR
        + n_O_m3 * KVT_O
    )


def electron_vibration_frequency_s(Te_K):
    Te_eV = np.maximum(k_to_ev(Te_K), TE_EV_FLOOR)
    return (
        ELECTRON_VIB_A_S
        * Te_eV ** (-3.0)
        * (ELECTRON_VIB_B_EV + Te_eV ** 5)
        * np.exp(-ELECTRON_VIB_EXP_EV3 / (Te_eV ** 3.0))
        / VIB_HV_EV
    )


def mixture_cv_j_kg_k(row):
    return GAS_CV_J_KG_K


def mixture_r_specific_j_kg_k(row):
    return R_GAS / float(row["Mmix_kg_mol"])


def radial_grid_m():
    r_max = HYDRO_R_MAX_FACTOR * FILAMENT_SIGMA_M
    return np.linspace(0.0, r_max, HYDRO_NR)


def gaussian_source_profile(r_m):
    return np.exp(-(r_m * r_m) / (2.0 * FILAMENT_SIGMA_M * FILAMENT_SIGMA_M))


def radial_gradient(values, r_m, center_zero=True):
    grad = np.gradient(values, r_m, edge_order=2)
    if center_zero:
        grad[0] = 0.0
    return grad


def axisymmetric_divergence(r_m, radial_flux):
    r_flux = r_m * radial_flux
    denominator = r_m.copy()
    denominator[0] = 1.0
    div = np.gradient(r_flux, r_m, edge_order=2) / denominator
    if len(r_m) > 1:
        div[0] = 2.0 * (radial_flux[1] - radial_flux[0]) / (r_m[1] - r_m[0])
    return div


def scalar_laplacian_axisymmetric(values, r_m):
    return axisymmetric_divergence(r_m, radial_gradient(values, r_m, center_zero=True))


def compressible_viscous_terms(u_m_s, r_m):
    """
    严格按正文轴对称 Navier-Stokes 方程计算黏性项。

    momentum:
        2 eta/3 * [
            div_r(2 du/dr - u/r) + (du/dr)/r - u/r^2
        ]

    energy:
        2 eta/3 * div_r[u * (2 du/dr - u/r)]
    """
    eta = GAS_DYNAMIC_VISCOSITY_PA_S
    du_dr = radial_gradient(u_m_s, r_m, center_zero=False)
    u_over_r = np.empty_like(u_m_s)
    u_over_r[1:] = u_m_s[1:] / r_m[1:]
    u_over_r[0] = du_dr[0]

    strain_combination = 2.0 * du_dr - u_over_r
    geometric_term = np.zeros_like(u_m_s)
    geometric_term[1:] = (
        du_dr[1:] / r_m[1:]
        - u_m_s[1:] / (r_m[1:] * r_m[1:])
    )
    stress_force = (2.0 / 3.0) * eta * (
        axisymmetric_divergence(r_m, strain_combination)
        + geometric_term
    )
    stress_force[0] = 0.0

    viscous_energy = (
        (2.0 / 3.0)
        * eta
        * axisymmetric_divergence(
            r_m,
            u_m_s * strain_combination,
        )
    )
    return stress_force, viscous_energy


def initial_hydro_state(row):
    r_m = radial_grid_m()
    T0 = float(row["Tg0_K"])
    rho0 = float(row["rho0_kg_m3"])
    return {
        "r_m": r_m,
        "source_profile": gaussian_source_profile(r_m),
        "rho_kg_m3": np.full_like(r_m, rho0, dtype=float),
        "T_K": np.full_like(r_m, T0, dtype=float),
        "u_m_s": np.zeros_like(r_m, dtype=float),
        "rho_ambient_kg_m3": rho0,
        "T_ambient_K": T0,
    }


def hydro_step_axisymmetric(hydro, Q_J_m3_s, row, dt_s):
    """
    对正文的质量、径向动量和总能量守恒方程做显式欧拉更新。

    守恒变量依次为 rho、rho*u 和
    rho*(Cv*T + u^2/2)。Q 可以是完整径向数组；保留标量输入兼容性。
    """
    r_m = hydro["r_m"]
    rho = hydro["rho_kg_m3"]
    T = hydro["T_K"]
    u = hydro["u_m_s"]

    cv = mixture_cv_j_kg_k(row)
    r_specific = mixture_r_specific_j_kg_k(row)
    p = rho * r_specific * T

    momentum = rho * u
    total_energy = rho * (cv * T + 0.5 * u * u)

    drho_dt = -axisymmetric_divergence(r_m, rho * u)
    dp_dr = radial_gradient(p, r_m, center_zero=True)
    stress_force, viscous_energy = compressible_viscous_terms(u, r_m)
    dmomentum_dt = (
        -axisymmetric_divergence(r_m, rho * u * u)
        - dp_dr
        + stress_force
    )

    heat_source = np.asarray(Q_J_m3_s, dtype=float)
    if heat_source.ndim == 0:
        heat_source = float(heat_source) * hydro["source_profile"]
    conduction = GAS_THERMAL_CONDUCTIVITY_W_M_K * scalar_laplacian_axisymmetric(T, r_m)
    dtotal_energy_dt = (
        -axisymmetric_divergence(r_m, (total_energy + p) * u)
        + heat_source
        + conduction
        + viscous_energy
    )

    rho_next = np.maximum(rho + dt_s * drho_dt, 1.0e-12)
    momentum_next = momentum + dt_s * dmomentum_dt
    total_energy_next = total_energy + dt_s * dtotal_energy_dt

    u_next = momentum_next / rho_next
    specific_internal_energy = (
        total_energy_next / rho_next - 0.5 * u_next * u_next
    )
    specific_internal_energy = np.maximum(
        specific_internal_energy,
        cv * TEMP_FLOOR_K,
    )
    T_next = specific_internal_energy / cv

    rho_next[0] = rho_next[1]
    T_next[0] = T_next[1]
    u_next[0] = 0.0

    rho_next[-1] = hydro["rho_ambient_kg_m3"]
    T_next[-1] = hydro["T_ambient_K"]
    u_next[-1] = 0.0

    hydro["rho_kg_m3"] = rho_next
    hydro["T_K"] = T_next
    hydro["u_m_s"] = u_next
    return hydro


def initial_species_from_row(row):
    return np.array(
        [
            float(row["ne0_cm3"]),
            float(row["n_O2_plus0_cm3"]),
            float(row["n_O0_cm3"]),
            float(row["neutral_O2_0_cm3"]),
            float(row["n_O2_minus0_cm3"]),
            float(row["n_O4_plus0_cm3"]),
            float(row["neutral_N2_0_cm3"]),
            float(row["neutral_Ar_0_cm3"]),
            float(row["n_N2_plus0_cm3"]),
            float(row["n_Ar_plus0_cm3"]),
        ],
        dtype=float,
    )


def initial_species_profiles_from_row(row, source_profile):
    """
    由第二章给出的轴心初值建立高斯径向粒子分布。

    电子及带电/反应产物取高斯分布；中性 N2、O2、Ar 在轴心严格
    等于输入CSV，并向径向外边界恢复到电离前的背景数密度。
    """
    center = initial_species_from_row(row)
    profile = np.asarray(source_profile, dtype=float)
    y = np.zeros((len(IDX), profile.size), dtype=float)

    localized_species = ("ne", "O2p", "O", "O2m", "O4p", "N2p", "Arp")
    for species in localized_species:
        y[IDX[species]] = center[IDX[species]] * profile

    n2_background = center[IDX["N2"]] + center[IDX["N2p"]]
    ar_background = center[IDX["Ar"]] + center[IDX["Arp"]]
    o2_background = (
        center[IDX["O2"]]
        + center[IDX["O2p"]]
        + center[IDX["O2m"]]
        + 2.0 * center[IDX["O4p"]]
        + 0.5 * center[IDX["O"]]
    )

    y[IDX["N2"]] = n2_background - (
        n2_background - center[IDX["N2"]]
    ) * profile
    y[IDX["O2"]] = o2_background - (
        o2_background - center[IDX["O2"]]
    ) * profile
    y[IDX["Ar"]] = ar_background - (
        ar_background - center[IDX["Ar"]]
    ) * profile
    return np.maximum(y, 0.0)


def initial_temperature_profiles(row, hydro):
    """
    局部电子服从麦克斯韦能量分布，Te 是其温度矩；丝状体径向
    扰动取与正文一致的高斯分布，外边界回到环境温度。
    """
    profile = hydro["source_profile"]
    ambient = hydro["T_ambient_K"]
    Te = ambient + (float(row["Te0_K"]) - ambient) * profile
    Tv = ambient + (float(row["Tv0_K"]) - ambient) * profile
    return Te, Tv


def reaction_heat_components_si(rates_cm3_s):
    qr_ev_cm3_s = sum(
        TABLE_REACTION_HEAT_EV[key] * rates_cm3_s[key]
        for key in DIRECT_REACTION_KEYS
    )
    qind_ev_cm3_s = sum(
        TABLE_REACTION_HEAT_EV[key] * rates_cm3_s[key]
        for key in INDIRECT_REACTION_KEYS
    )
    factor = CM3_PER_M3 * J_PER_EV
    return qr_ev_cm3_s * factor, qind_ev_cm3_s * factor


def state_diagnostics(y, Tg_K, Te_K, Tv_K, row):
    rates = reaction_rates_cm3_s(y, Te_K)
    dy = species_derivative_cm3_s(rates)

    ne_m3 = np.maximum(y[IDX["ne"]] * CM3_PER_M3, NE_FLOOR_M3)
    n_N2_m3 = np.maximum(y[IDX["N2"]] * CM3_PER_M3, 0.0)
    n_O2_m3 = np.maximum(y[IDX["O2"]] * CM3_PER_M3, 0.0)
    n_Ar_m3 = np.maximum(y[IDX["Ar"]] * CM3_PER_M3, 0.0)
    n_O_m3 = np.maximum(y[IDX["O"]] * CM3_PER_M3, 0.0)
    N_air_m3 = np.maximum(n_N2_m3 + n_O2_m3 + n_Ar_m3, 1.0)

    MWMSA_kg = float(row["MWMSA_kg"])

    nu_en, nu_ei, nu_sum = collision_frequencies_s(ne_m3, N_air_m3, Te_K)
    nu_ev = electron_vibration_frequency_s(Te_K)

    QeT = 1.5 * ne_m3 * K_B * (Te_K - Tg_K) * nu_sum * 2.0 * M_E / MWMSA_kg
    Qev = 1.5 * ne_m3 * K_B * (Te_K - Tv_K) * nu_ev

    vt_rate = vibrational_relaxation_rate_s(n_N2_m3, n_O2_m3, n_Ar_m3, n_O_m3)
    n_vib_m3 = np.maximum(n_N2_m3, 1.0)
    QVT = n_vib_m3 * (
        vibrational_energy_per_molecule_j(Tv_K)
        - vibrational_energy_per_molecule_j(Tg_K)
    ) * vt_rate

    QR, Qind = reaction_heat_components_si(rates)

    return {
        "dy_cm3_s": dy,
        "QeT_J_m3_s": QeT,
        "Qev_J_m3_s": Qev,
        "QVT_J_m3_s": QVT,
        "QR_J_m3_s": QR,
        "Qind_J_m3_s": Qind,
        "Q_total_to_gas_J_m3_s": QeT + QVT + QR + Qind,
        "nu_en_s": nu_en,
        "nu_ei_s": nu_ei,
        "nu_ev_s": nu_ev,
    }


def euler_step(y, hydro, Te_K, Tv_K, row, dt_s, diagnostics=None):
    """
    显式欧拉推进局部反应、电子能量PDE和振动能量PDE。

    电子方程直接推进 Ue=3*ne*k*Te/2，再用同一步更新后的 ne
    反求 Te，因此正文能量密度时间导数中的 dne/dt 已完整保留。
    """
    r_m = hydro["r_m"]
    Tg_K = hydro["T_K"]
    u_m_s = hydro["u_m_s"]
    diagnostics = diagnostics or state_diagnostics(y, Tg_K, Te_K, Tv_K, row)
    dy = diagnostics["dy_cm3_s"]
    y_next = np.maximum(y + dt_s * dy, 0.0)

    QeT = diagnostics["QeT_J_m3_s"]
    Qev = diagnostics["Qev_J_m3_s"]
    QVT = diagnostics["QVT_J_m3_s"]

    ne_m3 = np.maximum(y[IDX["ne"]] * CM3_PER_M3, NE_FLOOR_M3)
    ne_next_m3 = np.maximum(
        y_next[IDX["ne"]] * CM3_PER_M3,
        NE_FLOOR_M3,
    )
    nu_sum = np.maximum(
        diagnostics["nu_en_s"] + diagnostics["nu_ei_s"],
        1.0,
    )

    grad_ne = radial_gradient(ne_m3, r_m, center_zero=True)
    grad_Te = radial_gradient(Te_K, r_m, center_zero=True)
    electron_flux = (
        -K_B * Te_K * grad_ne / (M_E * nu_sum)
        + ne_m3 * u_m_s
    )
    electron_flux[0] = 0.0
    electron_thermal_conductivity = (
        5.0 * ne_m3 * Te_K * K_B * K_B
        / (2.0 * M_E * nu_sum)
    )
    electron_energy_flux = (
        2.5 * electron_flux * K_B * Te_K
        - electron_thermal_conductivity * grad_Te
    )

    electron_energy = 1.5 * ne_m3 * K_B * Te_K
    d_electron_energy_dt = (
        -axisymmetric_divergence(r_m, electron_energy_flux)
        - QeT
        - Qev
    )
    electron_energy_next = electron_energy + dt_s * d_electron_energy_dt
    electron_energy_floor = 1.5 * ne_next_m3 * K_B * TEMP_FLOOR_K
    electron_energy_next = np.maximum(
        electron_energy_next,
        electron_energy_floor,
    )
    Te_next = electron_energy_next / (1.5 * ne_next_m3 * K_B)

    n_vib_m3 = np.maximum(y[IDX["N2"]] * CM3_PER_M3, 1.0)
    n_vib_next_m3 = np.maximum(y_next[IDX["N2"]] * CM3_PER_M3, 1.0)
    vibrational_energy = (
        n_vib_m3 * vibrational_energy_per_molecule_j(Tv_K)
    )
    d_vibrational_energy_dt = (
        -axisymmetric_divergence(r_m, vibrational_energy * u_m_s)
        + Qev
        - QVT
    )
    vibrational_energy_next = np.maximum(
        vibrational_energy + dt_s * d_vibrational_energy_dt,
        0.0,
    )
    Tv_next = vibrational_temperature_from_energy_k(
        vibrational_energy_next / n_vib_next_m3
    )

    Te_next[-1] = hydro["T_ambient_K"]
    Tv_next[-1] = hydro["T_ambient_K"]
    return y_next, Te_next, Tv_next, diagnostics


def format_height_km(height_km):
    value = float(height_km)
    if abs(value - round(value)) < 1.0e-9:
        return f"{int(round(value))}km"
    return f"{value:g}km".replace(".", "p")


def output_case_prefix(date_text, height_km):
    date = pd.to_datetime(date_text)
    return f"{date:%Y-%m-%d}-{date.hour}-{format_height_km(height_km)}"


def simulate_case(row):
    dt_s = DT_NS * 1.0e-9
    steps = int(round(T_END_NS / DT_NS)) + 1

    hydro = initial_hydro_state(row)
    y = initial_species_profiles_from_row(row, hydro["source_profile"])
    Te, Tv = initial_temperature_profiles(row, hydro)

    scalar_columns = [
        "t_ns",
        "Te_K",
        "Tv_K",
        "T_K",
        "T_peak_K",
        "rho_center_kg_m3",
        "rho_min_kg_m3",
        "u_center_m_s",
        "u_peak_m_s",
        "u_peak_signed_m_s",
    ]
    output_columns = [
        *scalar_columns,
        *SPECIES_COLUMNS,
        *Q_COLUMNS,
        *FREQUENCY_COLUMNS,
    ]
    record_count = (steps - 1) // SAVE_EVERY + 1
    output = {
        column: np.empty(record_count, dtype=float)
        for column in output_columns
    }
    record_index = 0

    for step in range(steps):
        diagnostics = state_diagnostics(y, hydro["T_K"], Te, Tv, row)
        if step % SAVE_EVERY == 0:
            u_abs = np.abs(hydro["u_m_s"])
            output["t_ns"][record_index] = step * DT_NS
            output["Te_K"][record_index] = Te[0]
            output["Tv_K"][record_index] = Tv[0]
            output["T_K"][record_index] = hydro["T_K"][0]
            output["T_peak_K"][record_index] = np.max(hydro["T_K"])
            output["rho_center_kg_m3"][record_index] = hydro["rho_kg_m3"][0]
            output["rho_min_kg_m3"][record_index] = np.min(hydro["rho_kg_m3"])
            output["u_center_m_s"][record_index] = hydro["u_m_s"][0]
            output["u_peak_m_s"][record_index] = np.max(u_abs)
            output["u_peak_signed_m_s"][record_index] = hydro["u_m_s"][
                int(np.argmax(u_abs))
            ]
            for column, value in zip(SPECIES_COLUMNS, y[:, 0]):
                output[column][record_index] = value
            for column in Q_COLUMNS:
                output[column][record_index] = diagnostics[column][0]
            for column in FREQUENCY_COLUMNS:
                output[column][record_index] = diagnostics[column][0]
            record_index += 1

        if step == steps - 1:
            break
        y, Te, Tv, _ = euler_step(
            y,
            hydro,
            Te,
            Tv,
            row,
            dt_s,
            diagnostics=diagnostics,
        )
        hydro = hydro_step_axisymmetric(
            hydro,
            diagnostics["Q_total_to_gas_J_m3_s"],
            row,
            dt_s,
        )

    return pd.DataFrame(
        {
            column: values[:record_index]
            for column, values in output.items()
        },
        columns=output_columns,
    )


def write_case_outputs(result, row):
    prefix = output_case_prefix(row["date"], row["H_km"])
    particle_dir = OUT_DIR / "particle_density"
    temperature_dir = OUT_DIR / "temperature"
    q_dir = OUT_DIR / "intermediate_Q"
    u_dir = OUT_DIR / "intermediate_u_by_case"
    particle_dir.mkdir(parents=True, exist_ok=True)
    temperature_dir.mkdir(parents=True, exist_ok=True)
    q_dir.mkdir(parents=True, exist_ok=True)
    u_dir.mkdir(parents=True, exist_ok=True)

    particle_csv = particle_dir / f"{prefix}-n.csv"
    te_csv = temperature_dir / f"{prefix}-Te.csv"
    tv_csv = temperature_dir / f"{prefix}-Tv.csv"
    t_csv = temperature_dir / f"{prefix}-T.csv"
    q_csv = q_dir / f"{prefix}-Q.csv"
    u_csv = u_dir / f"{prefix}-u.csv"

    result[["t_ns", *SPECIES_COLUMNS]].to_csv(
        particle_csv,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "t_ns": result["t_ns"],
            "Te_K": result["Te_K"],
            "Te_eV": result["Te_K"] * EV_PER_K,
        }
    ).to_csv(te_csv, index=False, encoding="utf-8-sig")
    result[["t_ns", "Tv_K"]].to_csv(
        tv_csv,
        index=False,
        encoding="utf-8-sig",
    )
    result[["t_ns", "T_K", "T_peak_K", "rho_center_kg_m3", "rho_min_kg_m3"]].to_csv(
        t_csv,
        index=False,
        encoding="utf-8-sig",
    )
    result[["t_ns", *Q_COLUMNS, *FREQUENCY_COLUMNS]].to_csv(
        q_csv,
        index=False,
        encoding="utf-8-sig",
    )
    result[
        [
            "t_ns",
            "u_center_m_s",
            "u_peak_m_s",
            "u_peak_signed_m_s",
        ]
    ].to_csv(
        u_csv,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "particle_csv": str(particle_csv),
        "Te_csv": str(te_csv),
        "Tv_csv": str(tv_csv),
        "T_csv": str(t_csv),
        "Q_csv": str(q_csv),
        "u_csv_case": str(u_csv),
    }


def integrate_over_time(result, column):
    t_s = result["t_ns"].to_numpy(dtype=float) * 1.0e-9
    values = result[column].to_numpy(dtype=float)
    return float(np.trapz(values, t_s))


def intermediate_summary(result, row):
    qet_int = integrate_over_time(result, "QeT_J_m3_s")
    qvt_int = integrate_over_time(result, "QVT_J_m3_s")
    qr_int = integrate_over_time(result, "QR_J_m3_s")
    qind_int = integrate_over_time(result, "Qind_J_m3_s")
    qtotal_int = integrate_over_time(result, "Q_total_to_gas_J_m3_s")
    rho_cv = float(row["rho0_kg_m3"]) * mixture_cv_j_kg_k(row)
    qet_initial = float(result.iloc[0]["QeT_J_m3_s"])
    qet_final = float(result.iloc[-1]["QeT_J_m3_s"])
    qvt_initial = float(result.iloc[0]["QVT_J_m3_s"])
    qvt_final = float(result.iloc[-1]["QVT_J_m3_s"])
    qr_initial = float(result.iloc[0]["QR_J_m3_s"])
    qr_final = float(result.iloc[-1]["QR_J_m3_s"])
    qind_initial = float(result.iloc[0]["Qind_J_m3_s"])
    qind_final = float(result.iloc[-1]["Qind_J_m3_s"])
    qtotal_initial = float(result.iloc[0]["Q_total_to_gas_J_m3_s"])
    qtotal_final = float(result.iloc[-1]["Q_total_to_gas_J_m3_s"])

    return {
        "u_max_abs_m_s": float(np.max(result["u_peak_m_s"].to_numpy(dtype=float))),
        "rho_center_initial_kg_m3": float(result.iloc[0]["rho_center_kg_m3"]),
        "rho_center_final_kg_m3": float(result.iloc[-1]["rho_center_kg_m3"]),
        "rho_min_final_kg_m3": float(result.iloc[-1]["rho_min_kg_m3"]),
        "T_peak_max_K": float(result["T_peak_K"].max()),
        "QeT_initial_J_m3_s": qet_initial,
        "QeT_final_J_m3_s": qet_final,
        "QeT_max_J_m3_s": float(result["QeT_J_m3_s"].max()),
        "QVT_initial_J_m3_s": qvt_initial,
        "QVT_final_J_m3_s": qvt_final,
        "QVT_max_J_m3_s": float(result["QVT_J_m3_s"].max()),
        "QR_initial_J_m3_s": qr_initial,
        "QR_final_J_m3_s": qr_final,
        "QR_max_J_m3_s": float(result["QR_J_m3_s"].max()),
        "Qind_initial_J_m3_s": qind_initial,
        "Qind_final_J_m3_s": qind_final,
        "Qind_max_J_m3_s": float(result["Qind_J_m3_s"].max()),
        "Q_total_initial_J_m3_s": qtotal_initial,
        "Q_total_final_J_m3_s": qtotal_final,
        "Q_total_max_J_m3_s": float(result["Q_total_to_gas_J_m3_s"].max()),
        "QeT_integral_J_m3": qet_int,
        "QVT_integral_J_m3": qvt_int,
        "QR_integral_J_m3": qr_int,
        "Qind_integral_J_m3": qind_int,
        "Q_total_integral_J_m3": qtotal_int,
        "rho_cv_J_m3_K": rho_cv,
        "T_delta_from_Q_K": qtotal_int / rho_cv if rho_cv > 0.0 else math.nan,
        "u_generated_by_pressure_gradient": bool(
            np.max(result["u_peak_m_s"].to_numpy(dtype=float)) > 0.0
        ),
        "QeT_nonnegative": bool((result["QeT_J_m3_s"] >= -1.0e-30).all()),
        "QVT_nonnegative": bool((result["QVT_J_m3_s"] >= -1.0e-30).all()),
        "QR_nonnegative": bool((result["QR_J_m3_s"] >= -1.0e-30).all()),
        "Qind_nonnegative": bool((result["Qind_J_m3_s"] >= -1.0e-30).all()),
        "Q_total_nonnegative": bool(
            (result["Q_total_to_gas_J_m3_s"] >= -1.0e-30).all()
        ),
        "QeT_decreases": bool(qet_final < qet_initial),
        "QR_decreases": bool(qr_final < qr_initial),
        "Qind_decreases": bool(qind_final < qind_initial),
        "Q_total_decreases": bool(qtotal_final < qtotal_initial),
        "QVT_starts_zero_and_positive": bool(
            abs(qvt_initial) < 1.0e-20 and result["QVT_J_m3_s"].max() > 0.0
        ),
        "density_channel_forms": bool(
            result.iloc[-1]["rho_center_kg_m3"] < result.iloc[0]["rho_center_kg_m3"]
        ),
    }


def write_u_by_height_csv(case_results):
    intermediate_dir = OUT_DIR / "intermediate_u"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    u_csv = intermediate_dir / "u_by_date_height.csv"

    if not case_results:
        pd.DataFrame(columns=["t_ns"]).to_csv(u_csv, index=False, encoding="utf-8-sig")
        return u_csv

    out = pd.DataFrame({"t_ns": case_results[0][1]["t_ns"]})
    for row, result in case_results:
        prefix = output_case_prefix(row["date"], row["H_km"])
        out[f"{prefix}_u_peak_m_s"] = result["u_peak_m_s"].to_numpy(dtype=float)
        out[f"{prefix}_u_center_m_s"] = result["u_center_m_s"].to_numpy(dtype=float)

    out.to_csv(u_csv, index=False, encoding="utf-8-sig")
    return u_csv


def run_all_cases():
    initial = pd.read_csv(INIT_CSV)
    date_text = initial["date"].astype(str).str.slice(0, 16)
    initial = initial[
        (date_text == RUN_DATE)
        & (initial["H_km"].astype(float) <= RUN_MAX_HEIGHT_KM)
    ].sort_values("H_km")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    case_results = []
    for _, row in initial.iterrows():
        result = simulate_case(row)
        output_paths = write_case_outputs(result, row)

        intermediates = intermediate_summary(result, row)
        summary = {
            "date": row["date"],
            "H_km": row["H_km"],
            "flow_model": FLOW_MODEL,
            **output_paths,
            "Te0_K": result.iloc[0]["Te_K"],
            "ne0_cm3": result.iloc[0]["ne_cm3"],
            "ne_end_cm3": result.iloc[-1]["ne_cm3"],
            "ne_retention": result.iloc[-1]["ne_cm3"] / max(result.iloc[0]["ne_cm3"], EPS),
            "T0_K": result.iloc[0]["T_K"],
            "T_end_K": result.iloc[-1]["T_K"],
            "T_delta_K": result.iloc[-1]["T_K"] - result.iloc[0]["T_K"],
            "T_peak_max_K": result["T_peak_K"].max(),
            "T_peak_delta_K": result["T_peak_K"].max() - result.iloc[0]["T_K"],
            "Te_end_K": result.iloc[-1]["Te_K"],
            "Tv_end_K": result.iloc[-1]["Tv_K"],
            "Tv_max_K": result["Tv_K"].max(),
            "O4p_max_cm3": result["n_O4p_cm3"].max(),
            **intermediates,
        }
        summary_rows.append(summary)
        case_results.append(
            (
                row,
                result[["t_ns", "u_peak_m_s", "u_center_m_s"]].copy(),
            )
        )

    u_csv = write_u_by_height_csv(case_results)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["u_csv"] = str(u_csv)
    summary_df.to_csv(
        OUT_DIR / "relax_Te11000K_trend_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    run_all_cases()
