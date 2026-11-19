import csv
import math
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


# ==========================================================
# 1. PPT / MO-PPT 原理函数
# ==========================================================

EV_TO_AU_E = 1.0 / 27.211386245988
FIELD_AU_SI = 5.14220674763e11
C_LIGHT = 2.99792458e8
EPS0 = 8.8541878128e-12
AU_FREQ = 4.134137333e16
E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31
K_PER_EV = 11604.51812

species_params = {
    "Ar": {
        "Ip_eV": 15.76,
        "Zc": 1.0,
        "l": 1,
        "m": 0,
        "model": "PPT",
    },
    "N2": {
        "Ip_eV": 15.58,
        "Zc": 1.0,
        "m": 0,
        "C_l": {0: 2.02, 2: 0.78, 4: 0.04},
        "model": "MO-PPT",
    },
    "O2": {
        "Ip_eV": 12.03,
        "Zc": 1.0,
        "m": 1,
        "C_l": {2: 0.62, 4: 0.03},
        "model": "MO-PPT",
    },
}


def n_Zc(Zc, E0):
    return Zc / np.sqrt(2.0 * E0)


def F_0(E0):
    return (2.0 * E0) ** 1.5


def l_Zc(n_eff):
    return n_eff - 1.0


def C_n_l(Zc, E0):
    n_eff = n_Zc(Zc, E0)
    l_eff = l_Zc(n_eff)
    return (
        2.0 ** (2.0 * n_eff)
        / (
            n_eff
            * math.gamma(n_eff + l_eff + 1.0)
            * math.gamma(n_eff - l_eff)
        )
    )


def f_l_m(l, m):
    m_abs = abs(int(m))
    return (
        (2 * l + 1) * math.factorial(int(l + m_abs))
        / (
            2.0 ** m_abs
            * math.factorial(m_abs)
            * math.factorial(int(l - m_abs))
        )
    )


def g_gamma(gamma):
    gamma = float(gamma)

    if gamma < 1e-6:
        return 1.0

    term1 = (1.0 + 1.0 / (2.0 * gamma ** 2)) * math.asinh(gamma)
    term2 = math.sqrt(1.0 + gamma ** 2) / (2.0 * gamma)

    return 3.0 * (term1 - term2) / (2.0 * gamma)


def intensity_to_field_au(I_SI):
    F_SI = np.sqrt(2.0 * I_SI / (C_LIGHT * EPS0))
    return F_SI / FIELD_AU_SI


def omega_to_au(lambda_m):
    omega_SI = 2.0 * np.pi * C_LIGHT / lambda_m
    return omega_SI / AU_FREQ


def optical_critical_plasma_density_m3(lambda_m):
    """
    光学临界等离子体密度 rho_c = eps0*m_e*omega^2/e^2。
    800 nm 时约为 1.74e27 m^-3 = 1.74e21 cm^-3。
    """
    omega_SI = 2.0 * math.pi * C_LIGHT / float(lambda_m)
    return EPS0 * M_ELECTRON * omega_SI ** 2 / E_CHARGE ** 2


def calc_Am_omega(
    omega_au,
    gamma,
    Ip_au,
    m,
    tol=1e-10,
    max_terms=5000,
    n_quad=400,
):
    gamma = float(gamma)
    omega_au = float(omega_au)
    Ip_au = float(Ip_au)
    m_abs = abs(int(m))

    if gamma < 1e-6:
        return 1.0

    if omega_au <= 0.0:
        raise ValueError("omega_au must be positive")

    if Ip_au <= 0.0:
        raise ValueError("Ip_au must be positive")

    nu = (Ip_au / omega_au) * (1.0 + 1.0 / (2.0 * gamma ** 2))

    alpha = 2.0 * (
        math.asinh(gamma)
        - gamma / math.sqrt(1.0 + gamma ** 2)
    )

    beta = 2.0 * gamma / math.sqrt(1.0 + gamma ** 2)

    s = np.linspace(0.0, 1.0, n_quad)
    y = 1.0 - s ** 2
    y_m = y ** m_abs

    def w_m_func(x):
        if x <= 0.0:
            return 0.0

        integrand = y_m * np.exp(-(x ** 2) * y)
        return x ** (2 * m_abs + 1) * np.trapz(integrand, s)

    prefactor = (
        4.0 / math.sqrt(3.0 * math.pi)
        * 1.0 / math.factorial(m_abs)
        * gamma ** 2 / (1.0 + gamma ** 2)
    )

    n_start = int(math.floor(nu)) + 1
    total = 0.0

    for k in range(max_terms):
        n = n_start + k
        delta = n - nu

        if delta <= 0.0:
            continue

        x = math.sqrt(beta * delta)
        term = math.exp(-alpha * delta) * w_m_func(x)
        total += term

        if k > 20 and abs(term) < tol * max(abs(total), 1.0):
            break

    return prefactor * total


def w_PPT_au(F_au, Am_omega, omega_au, E0_au, Zc=1.0, l=0, m=0):
    F0 = F_0(E0_au)
    m_abs = abs(int(m))

    gamma = omega_au * np.sqrt(2.0 * E0_au) / F_au

    Am_omega = calc_Am_omega(omega_au, gamma, E0_au, m)

    C_sq = C_n_l(Zc, E0_au)
    flm = f_l_m(l, m)
    n_eff = n_Zc(Zc, E0_au)
    g_val = g_gamma(gamma)

    return (
        C_sq
        * flm
        * E0_au
        * ((2.0 * F0 / F_au) ** (2.0 * n_eff - m_abs - 1.0))
        * ((1.0 + gamma ** 2) ** (m_abs / 2.0 + 3.0 / 4.0))
        * Am_omega
        * math.exp(-2.0 * F0 * g_val / (3.0 * F_au))
    )


def w_PPT_SI(I_SI, Am_omega, Ip_eV, Zc=1.0, l=0, m=0, lambda_m=800e-9):
    E0_au = Ip_eV * EV_TO_AU_E
    F_au = intensity_to_field_au(I_SI)
    omega_au = omega_to_au(lambda_m)

    return (
        w_PPT_au(
            F_au,
            Am_omega,
            omega_au,
            E0_au,
            Zc=Zc,
            l=l,
            m=m,
        )
        * AU_FREQ
    )


def Q_l_m(l, m):
    m_abs = abs(int(m))
    phase = (-1) ** int((m + m_abs) / 2)

    return phase * math.sqrt(
        ((2 * l + 1) * math.factorial(l + m_abs))
        / (2.0 * math.factorial(l - m_abs))
    )


def B_m(m, C_l_dict):
    return sum(C_l * Q_l_m(l, m) for l, C_l in C_l_dict.items())


def w_MO_PPT_au(F_au, Am_omega, omega_au, Ip_au, Zc, m, C_l_dict):
    kappa = math.sqrt(2.0 * Ip_au)
    m_abs = abs(int(m))

    gamma = omega_au * kappa / F_au

    Am_omega = calc_Am_omega(omega_au, gamma, Ip_au, m)

    B = B_m(m, C_l_dict)
    g_val = g_gamma(gamma)

    denom = (
        (2.0 ** m_abs)
        * math.factorial(m_abs)
        * (kappa ** (2.0 * Zc / kappa - 1.0))
    )

    pref = (B ** 2) * Am_omega / denom

    power_exp = 2.0 * Zc / kappa - m_abs - 1.0

    # MO-PPT 修正式采用 F * sqrt(1 + gamma^2)。
    # Zhao 等 Optics Communications 2014 原 Eq.(13) 曾将该处分母误印为
    # F * (1 + gamma^2)，corrigendum 已更正；这里按更正公式实现。
    power = (
        2.0 * kappa ** 3
        / (F_au * math.sqrt(1.0 + gamma ** 2))
    ) ** power_exp

    expo = math.exp(
        -2.0 * kappa ** 3 * g_val / (3.0 * F_au)
    )

    return pref * power * expo


def w_MO_PPT_SI(I_SI, Am_omega, Ip_eV, Zc, m, C_l_dict, lambda_m=800e-9):
    Ip_au = Ip_eV * EV_TO_AU_E
    F_au = intensity_to_field_au(I_SI)
    omega_au = omega_to_au(lambda_m)

    return (
        w_MO_PPT_au(
            F_au,
            Am_omega,
            omega_au,
            Ip_au,
            Zc,
            m,
            C_l_dict,
        )
        * AU_FREQ
    )


def ionization_rate(I_SI, gas, lambda_m=800e-9, Am_omega=1.0):
    p = species_params[gas]
    if p["model"] == "PPT":
        return w_PPT_SI(
            I_SI,
            Am_omega,
            p["Ip_eV"],
            Zc=p["Zc"],
            l=p["l"],
            m=p["m"],
            lambda_m=lambda_m,
        )
    return w_MO_PPT_SI(
        I_SI,
        Am_omega,
        p["Ip_eV"],
        Zc=p["Zc"],
        m=p["m"],
        C_l_dict=p["C_l"],
        lambda_m=lambda_m,
    )


# ==========================================================
# 2. 临界光强模型设置
# ==========================================================

LAMBDA_M = 800e-9
# Pcr 公式中的线性折射率。中文版 Word 中说明低空可忽略 n0
# 随压力和温度的微小变化，因此这里按论文近似取 1。
N0_FOR_PCR = 1.0
# Kerr-plasma 平衡式中的临界等离子体密度应为光学临界密度。
# 800 nm 时约 1.74e27 m^-3。
N_E_CR_MODEL_M3 = optical_critical_plasma_density_m3(LAMBDA_M)
# 一个周期设为 100 fs
PULSE_PERIOD_S = 100e-15
# 高斯强度 FWHM 设为 100 fs
PULSE_FWHM_S = 100e-15
# 显式欧拉步长 0.1 fs
DT_S = 0.1e-15
# 一个周期设为 1；三个周期改成 3
N_BALANCE_PERIODS = 1

# 平衡公式中的 tau
TAU_BALANCE_S = N_BALANCE_PERIODS * PULSE_PERIOD_S

# 高斯脉冲积分窗口与强度 FWHM 分开定义。
# 这里积分 0~10 FWHM，中心在 5 FWHM，避免用单个 t=0.5 fs 点代表脉冲过程。
PULSE_WINDOW_FWHM = 10.0
PULSE_CENTER_S = 0.5 * PULSE_WINDOW_FWHM * PULSE_FWHM_S
T_END_PULSE_S = PULSE_WINDOW_FWHM * PULSE_FWHM_S

# 临界光强搜索范围，单位 W/m^2
I_SEARCH_MIN = 1.0e16
I_SEARCH_MAX = 1.0e20

# 逐高度试差参数
N_TRIAL_COARSE = 70
N_TRIAL_LOCAL = 31
N_TRIAL_REFINEMENTS = 6

# n2' 的组分参数，单位 cm^2/W。
# 如果你论文表格中有更准确的 N2/O2/Ar n2 值，直接改这里。
N2_COMPONENT_CM2_W = {
    "N2": 7.9e-20,
    "O2": 8.1e-20,
    "Ar": 10.1e-20,
}

# 由数密度和温度计算压强时使用。主计算中 n2'=n2*p 的 p
# 直接取中文版 Word 表 1 给出的 P(atm)，此函数仅保留为一致性检查。
K_B = 1.380649e-23


# ==========================================================
# 3. 大气初始数据
# ==========================================================

def cm3_to_m3(x_cm3):
    return float(x_cm3) * 1.0e6


def build_atmosphere_cases():

    H = [0, 5, 10, 11, 15, 20, 25, 30, 32, 35, 40]

    blocks = [
        {
            "date": "2024-10-27 01:00",
            "H_km": H,
            "N2_cm3": [
                1.955e19, 1.164e19, 6.639e18, 5.851e18, 3.324e18,
                1.480e18, 6.473e17, 2.938e17, 2.149e17, 1.352e17,
                6.440e16,
            ],
            "O2_cm3": [
                5.244e18, 3.121e18, 1.780e18, 1.569e18, 8.913e17,
                3.969e17, 1.736e17, 7.880e16, 5.762e16, 3.627e16,
                1.727e16,
            ],
            "Ar_cm3": [
                2.337e17, 1.391e17, 7.934e16, 6.993e16, 3.972e16,
                1.769e16, 7.736e15, 3.512e15, 2.568e15, 1.616e15,
                7.697e14,
            ],
            "T_K": [
                290.0, 262.4, 229.9, 224.6, 211.0,
                210.9, 219.2, 225.5, 229.0, 235.0, 244.0,
            ],
            "P_atm": [
                1.01092, 0.54188, 0.26947, 0.23214, 0.12400,
                0.05479, 0.02472, 0.01148, 0.00850, 0.00549,
                0.00271,
            ],
        },
        {
            "date": "2024-10-11 01:00",
            "H_km": H,
            "N2_cm3": [
                1.933e19, 1.159e19, 6.663e18, 5.892e18, 3.388e18,
                1.507e18, 6.583e17, 2.994e17, 2.192e17, 1.381e17,
                6.581e16,
            ],
            "O2_cm3": [
                5.184e18, 3.108e18, 1.787e18, 1.580e18, 9.085e17,
                4.042e17, 1.765e17, 8.028e16, 5.878e16, 3.704e16,
                1.765e16,
            ],
            "Ar_cm3": [
                2.310e17, 1.385e17, 7.963e16, 7.042e16, 4.049e16,
                1.801e16, 7.867e15, 3.578e15, 2.620e15, 1.651e15,
                7.865e14,
            ],
            "T_K": [
                293.3, 265.2, 232.5, 226.7, 211.0,
                210.9, 219.7, 226.1, 229.5, 235.4, 244.7,
            ],
            "P_atm": [
                1.00457, 0.54319, 0.26866, 0.23083, 0.12307,
                0.05498, 0.02518, 0.01168, 0.00862, 0.00549,
                0.00268,
            ],
        },
    ]

    cases = []

    for block in blocks:
        for i, h in enumerate(block["H_km"]):
            cases.append({
                "date": block["date"],
                "H_km": h,
                "T_K": block["T_K"][i],
                "P_atm_table": block["P_atm"][i],
                "N2_m3": cm3_to_m3(block["N2_cm3"][i]),
                "O2_m3": cm3_to_m3(block["O2_cm3"][i]),
                "Ar_m3": cm3_to_m3(block["Ar_cm3"][i]),
            })

    return cases


# ==========================================================
# 4. 时间网格、光强、n2'、W 实时计算
# ==========================================================

def critical_plasma_density_m3(lambda_m=800e-9):
    return optical_critical_plasma_density_m3(lambda_m)


def build_time_grid():
    n_steps = int(np.floor(T_END_PULSE_S / DT_S)) + 1
    t_grid = np.arange(n_steps, dtype=float) * DT_S

    pulse_shape = np.exp(
        -4.0 * np.log(2.0) * ((t_grid - PULSE_CENTER_S) / PULSE_FWHM_S) ** 2
    )

    return t_grid, pulse_shape


T_GRID_S, PULSE_SHAPE = build_time_grid()
IDX_PEAK = int(np.argmax(PULSE_SHAPE))


def gaussian_intensity_array(I_peak_SI):
    return I_peak_SI * PULSE_SHAPE


def n2_mix_from_neutral_cm2_W(neutral_m3):
    """
    按当前中性组分比例计算 1 atm 下混合气体 n2，单位 cm^2/W。
    """
    gases = ("N2", "O2", "Ar")
    neutral_total = sum(float(neutral_m3.get(gas, 0.0)) for gas in gases)
    if neutral_total <= 0.0:
        return 0.0

    return sum(
        N2_COMPONENT_CM2_W[gas] * float(neutral_m3.get(gas, 0.0)) / neutral_total
        for gas in gases
    )


def effective_n2_prime_from_neutral_m2_W(neutral_m3, p_atm):
    """
    用当前剩余中性粒子密度计算 n2'(t)，单位 m^2/W。

    论文中采用 n2' = n2 * p。对混合气体，先按当前中性粒子
    组分比例得到 1 atm 下的混合气体 n2，再乘中文版 Word
    表 1 给出的无量纲相对压强 p：

        n2'(t) = p_table * sum_i x_i(t) n2_i

    其中 x_i = n_i / sum_j n_j。neutral_m3 只用于计算 x_i，
    不能直接乘入 n2，否则单位会变成 cm^2/W*m^-3。
    """
    n2_mix_cm2_W = n2_mix_from_neutral_cm2_W(neutral_m3)
    return n2_mix_cm2_W * float(p_atm) * 1.0e-4


def self_focusing_critical_power_W(n2_prime_m2_W, lambda_m=800e-9, n0=N0_FOR_PCR):
    """
    自聚焦临界功率 Pcr，单位 W。

        Pcr = 3.72 lambda^2 / (8 pi n0 n2')

    n2' 使用当前高度/日期对应的有效非线性折射率，单位 m^2/W。
    """
    n2_prime_m2_W = float(n2_prime_m2_W)
    n0 = float(n0)
    if n2_prime_m2_W <= 0.0 or n0 <= 0.0:
        return math.nan
    return 3.72 * float(lambda_m) ** 2 / (
        8.0 * math.pi * n0 * n2_prime_m2_W
    )


def pcr_row_for_case(case, lambda_m=800e-9, n0=N0_FOR_PCR):
    """
    按论文 Pcr 公式计算单个日期/高度初始时刻的自聚焦临界功率。
    """
    neutral = {
        "N2": float(case.get("N2_m3", 0.0)),
        "O2": float(case.get("O2_m3", 0.0)),
        "Ar": float(case.get("Ar_m3", 0.0)),
    }
    neutral_total = neutral["N2"] + neutral["O2"] + neutral["Ar"]
    fractions = {
        gas: neutral[gas] / neutral_total if neutral_total > 0.0 else math.nan
        for gas in ("N2", "O2", "Ar")
    }

    n2_mix_cm2_W = n2_mix_from_neutral_cm2_W(neutral)
    n2_prime_m2_W = effective_n2_prime_from_neutral_m2_W(
        neutral,
        float(case["P_atm_table"]),
    )
    pcr_W = self_focusing_critical_power_W(n2_prime_m2_W, lambda_m, n0)
    pulse_energy_J = pcr_W * PULSE_FWHM_S if np.isfinite(pcr_W) else math.nan

    return {
        "date": case["date"],
        "H_km": case["H_km"],
        "t_fs": 0.0,
        "T_K": case["T_K"],
        "P_atm": case["P_atm_table"],
        "lambda_m": lambda_m,
        "n0_for_Pcr": n0,
        "N2_fraction_initial": fractions["N2"],
        "O2_fraction_initial": fractions["O2"],
        "Ar_fraction_initial": fractions["Ar"],
        "n2_mix_initial_1atm_cm2_W": n2_mix_cm2_W,
        "n2_prime_initial_m2_W": n2_prime_m2_W,
        "Pcr_initial_W": pcr_W,
        "Pcr_initial_GW": pcr_W / 1.0e9 if np.isfinite(pcr_W) else math.nan,
        "pulse_fwhm_fs": PULSE_FWHM_S / 1.0e-15,
        "Pcr_initial_times_fwhm_J": pulse_energy_J,
        "Pcr_initial_times_fwhm_mJ": pulse_energy_J * 1.0e3
        if np.isfinite(pulse_energy_J)
        else math.nan,
    }


def _nan_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full_like(numerator, np.nan, dtype=float)
    mask = denominator > 0.0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _last_finite_value(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return math.nan
    return float(arr[np.where(finite)[0][-1]])


def photon_energy_eV(lambda_m):
    """单光子能量 h nu，单位 eV。"""
    return 1239.841984 / (float(lambda_m) * 1.0e9)


def ponderomotive_energy_eV(I_W_m2, lambda_m):
    """
    有质动力势 Up，单位 eV。

        Up(eV) = 9.337e-14 I(W/cm^2) lambda(um)^2
    """
    I_W_cm2 = np.asarray(I_W_m2, dtype=float) / 1.0e4
    lambda_um = float(lambda_m) * 1.0e6
    return 9.337e-14 * I_W_cm2 * lambda_um ** 2


def min_photon_number_array(I_W_m2, Ip_eV, lambda_m):
    hnu = photon_energy_eV(lambda_m)
    Up = ponderomotive_energy_eV(I_W_m2, lambda_m)
    return np.ceil((float(Ip_eV) + Up) / hnu).astype(int)


def electron_kinetic_energy_array_eV(I_W_m2, Ip_eV, lambda_m):
    """
    新生电子动能：

        E_i(t) = K_i(t) h nu - Ip_i - Up(t)

    这里的 E_i 才进入电子温度能量守恒式，不能用 Ip_i 直接替代。
    """
    hnu = photon_energy_eV(lambda_m)
    Up = ponderomotive_energy_eV(I_W_m2, lambda_m)
    K = min_photon_number_array(I_W_m2, Ip_eV, lambda_m)
    E = K * hnu - float(Ip_eV) - Up
    return np.maximum(E, 0.0), K, Up


def compute_Te_time_arrays(I_t, R_N2, R_O2, R_Ar, dt_s, lambda_m):
    """
    按论文中的能量守恒关系计算电子温度：

        sum_i n_{e,i} E_i e = 3/2 n_e k_B T_e

    这里采用显式欧拉步进得到的新增电子密度
    Delta n_{e,i} = R_i Delta t。E_i 是新生电子动能，
    不是电离势 Ip_i 本身。
    """
    I_t = np.asarray(I_t, dtype=float)
    R_N2 = np.asarray(R_N2, dtype=float)
    R_O2 = np.asarray(R_O2, dtype=float)
    R_Ar = np.asarray(R_Ar, dtype=float)

    E_N2, K_N2, Up = electron_kinetic_energy_array_eV(
        I_t, species_params["N2"]["Ip_eV"], lambda_m
    )
    E_O2, K_O2, _ = electron_kinetic_energy_array_eV(
        I_t, species_params["O2"]["Ip_eV"], lambda_m
    )
    E_Ar, K_Ar, _ = electron_kinetic_energy_array_eV(
        I_t, species_params["Ar"]["Ip_eV"], lambda_m
    )

    dne_N2 = R_N2 * float(dt_s)
    dne_O2 = R_O2 * float(dt_s)
    dne_Ar = R_Ar * float(dt_s)
    dne_total = dne_N2 + dne_O2 + dne_Ar

    energy_step = dne_N2 * E_N2 + dne_O2 * E_O2 + dne_Ar * E_Ar
    mean_electron_energy_inst_eV = _nan_divide(energy_step, dne_total)
    Te_inst_eV = (2.0 / 3.0) * mean_electron_energy_inst_eV

    energy_cum = np.cumsum(energy_step)
    dne_cum = np.cumsum(dne_total)
    mean_electron_energy_cum_eV = _nan_divide(energy_cum, dne_cum)
    Te_cum_eV = (2.0 / 3.0) * mean_electron_energy_cum_eV

    return {
        "K_N2": K_N2,
        "K_O2": K_O2,
        "K_Ar": K_Ar,
        "Up_eV": Up,
        "Ekin_N2_eV": E_N2,
        "Ekin_O2_eV": E_O2,
        "Ekin_Ar_eV": E_Ar,
        "dne_N2_m-3": dne_N2,
        "dne_O2_m-3": dne_O2,
        "dne_Ar_m-3": dne_Ar,
        "Te_inst_eV": Te_inst_eV,
        "Te_inst_K": Te_inst_eV * K_PER_EV,
        "Te_cum_eV": Te_cum_eV,
        "Te_cum_K": Te_cum_eV * K_PER_EV,
        "mean_electron_energy_cum_eV": mean_electron_energy_cum_eV,
    }


def neutral_pressure_atm(neutral_m3, T_K):
    """
    由当前剩余中性粒子数密度估算中性气体压强。
    注意：飞秒阶段这只是数密度对应的状态量，不代表已经发生流体膨胀。
    """
    n_total = neutral_m3["N2"] + neutral_m3["O2"] + neutral_m3["Ar"]
    P_Pa = n_total * K_B * T_K
    return P_Pa / 101325.0


def ionization_rate_time_array(I_SI_array, gas, lambda_m=800e-9):
    """
    对给定 I(t) 逐点实时计算 W_i[I(t)]。
    不使用 W 表、不使用插值；PPT/MO-PPT 速率仍由 ionization_rate() 给出。
    """
    I_arr = np.asarray(I_SI_array, dtype=float)
    W = np.empty_like(I_arr, dtype=float)

    for index, I_value in np.ndenumerate(I_arr):
        W[index] = ionization_rate(
            float(I_value),
            gas,
            lambda_m=lambda_m,
            Am_omega=1.0,
        )

    return np.nan_to_num(W, nan=0.0, posinf=1.0e300, neginf=0.0)


def ionization_rate_at_peak(I_SI, gas, lambda_m=800e-9):
    """直接按当前光强计算单个时刻的 W_i(I)，用于夹持光强试差。"""
    W = ionization_rate(float(I_SI), gas, lambda_m=lambda_m, Am_omega=1.0)
    return float(np.nan_to_num(W, nan=0.0, posinf=1.0e300, neginf=0.0))


# ==========================================================
# 5. 显式欧拉计算：给定 I_peak，计算 residual
# ==========================================================

def explicit_euler_species_arrays(n0_i, W_i_array):
    """
    对单个组分按论文方程做显式欧拉：

        d n_e,i / dt = W_i[I(t)] * (n_i - n_e,i)

    对第 k 个时间步：

        R_i^k = W_i^k * (n_i - n_e,i^k)
        Delta n_e,i^k = R_i^k * Delta t
        n_e,i^{k+1} = n_e,i^k + Delta n_e,i^k

    这里的 R_i 是体积电离产生率，单位 m^-3 s^-1。
    W 是单粒子电离率，单位 s^-1；不得在 W 或 R 中重复乘中性密度。
    """
    W_i_array = np.asarray(W_i_array, dtype=float)
    n0_i = max(float(n0_i), 0.0)

    neutral_before = np.empty_like(W_i_array, dtype=float)
    neutral_after = np.empty_like(W_i_array, dtype=float)
    ne_after = np.empty_like(W_i_array, dtype=float)
    R_array = np.empty_like(W_i_array, dtype=float)

    ne_current = 0.0

    for k, W_k in enumerate(W_i_array):
        neutral_current = max(n0_i - ne_current, 0.0)
        neutral_before[k] = neutral_current

        R_raw = max(float(W_k), 0.0) * neutral_current
        dne = R_raw * DT_S

        if dne > neutral_current:
            dne = neutral_current

        ne_current += dne
        if ne_current > n0_i:
            ne_current = n0_i

        neutral_after[k] = max(n0_i - ne_current, 0.0)
        ne_after[k] = ne_current
        R_array[k] = dne / DT_S if DT_S > 0.0 else 0.0

    return neutral_before, neutral_after, ne_after, R_array


def simulate_case_euler(
    case,
    I_peak_SI,
    lambda_m=800e-9,
    return_time_rows=False,
):
    """
    给定峰值光强 I_peak_SI，对单个高度/日期进行显式欧拉动态电离。

    临界平衡采用峰值光强时刻的：

        n2'(t_peak) * I_peak = tau * R_total(t_peak)/(2 n_e^cr)
    """
    gases = ("N2", "O2", "Ar")

    n0 = {
        "N2": float(case.get("N2_m3", 0.0)),
        "O2": float(case.get("O2_m3", 0.0)),
        "Ar": float(case.get("Ar_m3", 0.0)),
    }

    I_t = gaussian_intensity_array(I_peak_SI)

    W = {}
    neutral_before = {}
    neutral_after = {}
    ne_after = {}
    R = {}

    for gas in gases:
        W[gas] = ionization_rate_time_array(I_t, gas, lambda_m=lambda_m)
        (
            neutral_before[gas],
            neutral_after[gas],
            ne_after[gas],
            R[gas],
        ) = explicit_euler_species_arrays(n0[gas], W[gas])

    R_total = R["N2"] + R["O2"] + R["Ar"]
    ne_total = ne_after["N2"] + ne_after["O2"] + ne_after["Ar"]
    neutral_total_after = (
        neutral_after["N2"] + neutral_after["O2"] + neutral_after["Ar"]
    )
    Te = compute_Te_time_arrays(
        I_t,
        R["N2"],
        R["O2"],
        R["Ar"],
        DT_S,
        lambda_m,
    )

    n_cr = critical_plasma_density_m3(lambda_m)

    idx = IDX_PEAK

    neutral_peak_for_R = {
        "N2": float(neutral_before["N2"][idx]),
        "O2": float(neutral_before["O2"][idx]),
        "Ar": float(neutral_before["Ar"][idx]),
    }

    p_for_n2_at_peak_atm = float(case["P_atm_table"])
    n2_prime_peak = effective_n2_prime_from_neutral_m2_W(
        neutral_peak_for_R,
        p_for_n2_at_peak_atm,
    )

    R_peak_total = float(R_total[idx])
    I_peak_seen = float(I_t[idx])

    kerr_term = n2_prime_peak * I_peak_seen
    plasma_term = TAU_BALANCE_S * R_peak_total / (2.0 * n_cr)
    balance_residual = plasma_term - kerr_term

    Te0_final_eV = _last_finite_value(Te["Te_cum_eV"])
    Te0_final_K = Te0_final_eV * K_PER_EV if np.isfinite(Te0_final_eV) else math.nan
    mean_electron_energy_final_eV = _last_finite_value(
        Te["mean_electron_energy_cum_eV"]
    )
    Te_inst_peak_eV = float(Te["Te_inst_eV"][idx])
    Te_inst_peak_K = float(Te["Te_inst_K"][idx])

    summary = {
        "date": case["date"],
        "H_km": case["H_km"],
        "T_K": case["T_K"],
        "P_atm_table": case["P_atm_table"],

        "status": "",
        "lambda_m": lambda_m,
        "N_BALANCE_PERIODS": N_BALANCE_PERIODS,
        "pulse_period_fs": PULSE_PERIOD_S / 1e-15,
        "pulse_fwhm_fs": PULSE_FWHM_S / 1e-15,
        "tau_balance_fs": TAU_BALANCE_S / 1e-15,
        "dt_fs": DT_S / 1e-15,

        "I_peak_W_m2": I_peak_SI,
        "I_peak_W_cm2": I_peak_SI / 1e4,
        "I_peak_seen_W_m2": I_peak_seen,
        "I_peak_seen_W_cm2": I_peak_seen / 1e4,

        "n_e_cr_m-3": n_cr,
        "n_e_cr_cm-3": n_cr / 1e6,

        "n0_N2_m-3": n0["N2"],
        "n0_O2_m-3": n0["O2"],
        "n0_Ar_m-3": n0["Ar"],
        "n0_total_m-3": n0["N2"] + n0["O2"] + n0["Ar"],

        "neutral_N2_at_peak_for_R_m-3": neutral_peak_for_R["N2"],
        "neutral_O2_at_peak_for_R_m-3": neutral_peak_for_R["O2"],
        "neutral_Ar_at_peak_for_R_m-3": neutral_peak_for_R["Ar"],
        "neutral_total_at_peak_for_R_m-3": (
            neutral_peak_for_R["N2"]
            + neutral_peak_for_R["O2"]
            + neutral_peak_for_R["Ar"]
        ),
        "P_for_n2_at_peak_atm": p_for_n2_at_peak_atm,

        "W_N2_at_peak_s-1": float(W["N2"][idx]),
        "W_O2_at_peak_s-1": float(W["O2"][idx]),
        "W_Ar_at_peak_s-1": float(W["Ar"][idx]),

        "R_N2_at_peak_m-3_s-1": float(R["N2"][idx]),
        "R_O2_at_peak_m-3_s-1": float(R["O2"][idx]),
        "R_Ar_at_peak_m-3_s-1": float(R["Ar"][idx]),
        "R_at_peak_total_m-3_s-1": R_peak_total,

        "ne_N2_at_peak_m-3": float(ne_after["N2"][idx]),
        "ne_O2_at_peak_m-3": float(ne_after["O2"][idx]),
        "ne_Ar_at_peak_m-3": float(ne_after["Ar"][idx]),
        "ne_total_at_peak_m-3": float(ne_total[idx]),
        "ne_total_at_peak_cm-3": float(ne_total[idx]) / 1e6,

        "final_ne_N2_m-3": float(ne_after["N2"][-1]),
        "final_ne_O2_m-3": float(ne_after["O2"][-1]),
        "final_ne_Ar_m-3": float(ne_after["Ar"][-1]),
        "final_ne_total_m-3": float(ne_total[-1]),
        "final_ne_total_cm-3": float(ne_total[-1]) / 1e6,

        "max_ne_total_m-3": float(np.max(ne_total)),
        "max_ne_total_cm-3": float(np.max(ne_total)) / 1e6,
        "max_R_total_m-3_s-1": float(np.max(R_total)),

        "Te0_final_eV": Te0_final_eV,
        "Te0_final_K": Te0_final_K,
        "Te_inst_peak_eV": Te_inst_peak_eV,
        "Te_inst_peak_K": Te_inst_peak_K,
        "mean_electron_energy_final_eV": mean_electron_energy_final_eV,

        "n2_prime_at_peak_m2_W": n2_prime_peak,
        "n2I_at_peak": kerr_term,
        "tauR_over_2necr_at_peak": plasma_term,
        "balance_residual_dimensionless": balance_residual,
        "relative_residual": balance_residual / max(abs(kerr_term), 1e-300),
    }

    if not return_time_rows:
        return summary

    rows = []

    for k, t in enumerate(T_GRID_S):
        neutral_for_R = {
            "N2": float(neutral_before["N2"][k]),
            "O2": float(neutral_before["O2"][k]),
            "Ar": float(neutral_before["Ar"][k]),
        }

        p_for_n2_atm_t = float(case["P_atm_table"])
        n2_prime_t = effective_n2_prime_from_neutral_m2_W(
            neutral_for_R,
            p_for_n2_atm_t,
        )
        n2I_t = n2_prime_t * float(I_t[k])
        tauR_t = TAU_BALANCE_S * float(R_total[k]) / (2.0 * n_cr)
        residual_t = tauR_t - n2I_t

        rows.append({
            "date": case["date"],
            "H_km": case["H_km"],
            "T_K": case["T_K"],
            "P_atm_table": case["P_atm_table"],

            "Icrit_W_m2": I_peak_SI,
            "Icrit_W_cm2": I_peak_SI / 1e4,

            "t_s": float(t),
            "t_fs": float(t / 1e-15),

            "I_W_m2": float(I_t[k]),
            "I_W_cm2": float(I_t[k] / 1e4),

            "W_N2_s-1": float(W["N2"][k]),
            "W_O2_s-1": float(W["O2"][k]),
            "W_Ar_s-1": float(W["Ar"][k]),

            "R_N2_m-3_s-1": float(R["N2"][k]),
            "R_O2_m-3_s-1": float(R["O2"][k]),
            "R_Ar_m-3_s-1": float(R["Ar"][k]),
            "R_total_m-3_s-1": float(R_total[k]),

            "K_N2": int(Te["K_N2"][k]),
            "K_O2": int(Te["K_O2"][k]),
            "K_Ar": int(Te["K_Ar"][k]),
            "Up_eV": float(Te["Up_eV"][k]),
            "Ekin_N2_eV": float(Te["Ekin_N2_eV"][k]),
            "Ekin_O2_eV": float(Te["Ekin_O2_eV"][k]),
            "Ekin_Ar_eV": float(Te["Ekin_Ar_eV"][k]),
            "dne_N2_m-3": float(Te["dne_N2_m-3"][k]),
            "dne_O2_m-3": float(Te["dne_O2_m-3"][k]),
            "dne_Ar_m-3": float(Te["dne_Ar_m-3"][k]),
            "Te_inst_eV": float(Te["Te_inst_eV"][k]),
            "Te_inst_K": float(Te["Te_inst_K"][k]),
            "Te_cum_eV": float(Te["Te_cum_eV"][k]),
            "Te_cum_K": float(Te["Te_cum_K"][k]),
            "mean_electron_energy_cum_eV": float(
                Te["mean_electron_energy_cum_eV"][k]
            ),

            "ne_N2_m-3": float(ne_after["N2"][k]),
            "ne_O2_m-3": float(ne_after["O2"][k]),
            "ne_Ar_m-3": float(ne_after["Ar"][k]),
            "ne_total_m-3": float(ne_total[k]),
            "ne_total_cm-3": float(ne_total[k] / 1e6),

            "neutral_N2_m-3": float(neutral_after["N2"][k]),
            "neutral_O2_m-3": float(neutral_after["O2"][k]),
            "neutral_Ar_m-3": float(neutral_after["Ar"][k]),
            "neutral_total_m-3": float(neutral_total_after[k]),

            "P_for_n2_atm": p_for_n2_atm_t,

            "n2_prime_m2_W": n2_prime_t,
            "n2I": n2I_t,
            "tauR_over_2necr": tauR_t,
            "balance_residual": residual_t,

            "n_e_cr_m-3": n_cr,
            "ne_over_necr": float(ne_total[k] / n_cr),
        })

    return summary, rows


def peak_balance_residual(case, I_peak_SI, lambda_m=800e-9):
    """
    按论文夹持条件在峰值光强处直接计算 residual：

        n2'(I_peak) I_peak = tau R(I_peak)/(2 n_e^cr)

    此处 W_i(I_peak) 逐项实时调用 PPT/MO-PPT，不使用插值表。
    完整的 n_e(t)、T_e(t) 在最终 Icrit 下再由 simulate_case_euler()
    做显式欧拉时序推进。
    """
    gases = ("N2", "O2", "Ar")
    neutral = {
        "N2": float(case.get("N2_m3", 0.0)),
        "O2": float(case.get("O2_m3", 0.0)),
        "Ar": float(case.get("Ar_m3", 0.0)),
    }
    I_peak_seen = float(I_peak_SI) * float(PULSE_SHAPE[IDX_PEAK])
    p_atm = float(case["P_atm_table"])
    n2_prime = effective_n2_prime_from_neutral_m2_W(neutral, p_atm)
    R_total = sum(
        neutral[gas] * ionization_rate_at_peak(I_peak_seen, gas, lambda_m)
        for gas in gases
    )
    n_cr = critical_plasma_density_m3(lambda_m)
    kerr_term = n2_prime * I_peak_seen
    plasma_term = TAU_BALANCE_S * R_total / (2.0 * n_cr)
    return plasma_term - kerr_term


# ==========================================================
# 6. 真正逐高度试差法
# ==========================================================

def find_critical_intensity_trial(
    case,
    lambda_m=800e-9,
    I_min=I_SEARCH_MIN,
    I_max=I_SEARCH_MAX,
):
    """
    对每个 case 单独试差，寻找：

        n2'I = tau R(I)/(2 n_e^cr)
    """
    trial_records = []

    I_grid = np.logspace(np.log10(I_min), np.log10(I_max), N_TRIAL_COARSE)

    for I in I_grid:
        I = float(I)
        residual = peak_balance_residual(case, I, lambda_m=lambda_m)
        trial_records.append((I, residual))

    bracket = None

    for k in range(len(trial_records) - 1):
        I1, f1 = trial_records[k]
        I2, f2 = trial_records[k + 1]

        if f1 == 0.0:
            bracket = (I1, I1)
            break

        if f1 * f2 < 0.0:
            bracket = (I1, I2)
            break

    if bracket is None:
        best_I, best_f = min(
            trial_records,
            key=lambda item: abs(item[1]),
        )

        final_summary, final_rows = simulate_case_euler(
            case=case,
            I_peak_SI=best_I,
            lambda_m=lambda_m,
            return_time_rows=True,
        )

        final_summary["status"] = "NO_SIGN_CHANGE_BEST_TRIAL"
        final_summary["trial_count"] = len(trial_records)

        return best_I, final_summary, final_rows

    low, high = bracket

    if low == high:
        final_summary, final_rows = simulate_case_euler(
            case=case,
            I_peak_SI=low,
            lambda_m=lambda_m,
            return_time_rows=True,
        )
        final_summary["status"] = "OK_TRIAL_EXACT"
        final_summary["trial_count"] = len(trial_records)

        return low, final_summary, final_rows

    best_I = None

    for _ in range(N_TRIAL_REFINEMENTS):
        local_grid = np.logspace(np.log10(low), np.log10(high), N_TRIAL_LOCAL)
        local_records = []

        for I in local_grid:
            I = float(I)
            residual = peak_balance_residual(case, I, lambda_m=lambda_m)
            local_records.append((I, residual))

        trial_records.extend(local_records)

        best_I, best_f = min(
            local_records,
            key=lambda item: abs(item[1]),
        )

        new_bracket = None

        for k in range(len(local_records) - 1):
            I1, f1 = local_records[k]
            I2, f2 = local_records[k + 1]

            if f1 == 0.0:
                new_bracket = (I1, I1)
                break

            if f1 * f2 < 0.0:
                new_bracket = (I1, I2)
                break

        if new_bracket is None:
            best_index = [item[0] for item in local_records].index(best_I)

            if best_index == 0:
                low = local_records[0][0]
                high = local_records[1][0]
            elif best_index == len(local_records) - 1:
                low = local_records[-2][0]
                high = local_records[-1][0]
            else:
                low = local_records[best_index - 1][0]
                high = local_records[best_index + 1][0]
        else:
            low, high = new_bracket

            if low == high:
                best_I = low
                break

    final_summary, final_rows = simulate_case_euler(
        case=case,
        I_peak_SI=best_I,
        lambda_m=lambda_m,
        return_time_rows=True,
    )

    final_summary["status"] = "OK_TRIAL"
    final_summary["trial_count"] = len(trial_records)

    return best_I, final_summary, final_rows


# ==========================================================
# 7. 输出 CSV 和 n2I 文档
# ==========================================================

TIME_CSV_FIELDNAMES = [
    "date", "H_km", "T_K", "P_atm",
    "Icrit_W_m2", "Icrit_W_cm2",
    "t_s", "t_fs",
    "I_W_m2", "I_W_cm2",
    "W_N2_s^-1", "W_O2_s^-1", "W_Ar_s^-1",
    "R_N2_m^-3_s^-1", "R_O2_m^-3_s^-1", "R_Ar_m^-3_s^-1",
    "R_total_m^-3_s^-1",
    "K_N2", "K_O2", "K_Ar",
    "Up_eV",
    "Ekin_N2_eV", "Ekin_O2_eV", "Ekin_Ar_eV",
    "dne_N2_m-3", "dne_O2_m-3", "dne_Ar_m-3",
    "Te_inst_eV", "Te_inst_K", "Te_cum_eV", "Te_cum_K",
    "ne_N2_m^-3", "ne_O2_m^-3", "ne_Ar_m^-3", "ne_total_m^-3",
    "neutral_N2_m^-3", "neutral_O2_m^-3", "neutral_Ar_m^-3",
    "neutral_total_m^-3",
    "P_for_n2_atm",
    "n2_prime_m2_W", "n2I", "tauR_over_2necr", "balance_residual",
    "n_e_cr_m^-3", "ne_over_necr",
]


SUMMARY_CSV_FIELDNAMES = [
    "date",
    "H_km",
    "T_K",
    "P_atm",
    "Icrit_W_m2",
    "Icrit_W_cm2",
    "Te0_final_eV",
    "Te0_final_K",
    "Te_inst_peak_eV",
    "Te_inst_peak_K",
    "mean_electron_energy_final_eV",
    "n2_prime_at_peak_m2_W",
    "n2I_at_peak",
    "tauR_over_2necr_at_peak",
    "residual",
    "relative_residual",
    "W_N2_at_peak_s^-1",
    "W_O2_at_peak_s^-1",
    "W_Ar_at_peak_s^-1",
    "R_N2_at_peak_m^-3_s^-1",
    "R_O2_at_peak_m^-3_s^-1",
    "R_Ar_at_peak_m^-3_s^-1",
    "R_total_at_peak_m^-3_s^-1",
    "ne_N2_at_peak_m^-3",
    "ne_O2_at_peak_m^-3",
    "ne_Ar_at_peak_m^-3",
    "ne_total_at_peak_m^-3",
    "neutral_N2_at_peak_m^-3",
    "neutral_O2_at_peak_m^-3",
    "neutral_Ar_at_peak_m^-3",
    "trial_count",
    "status",
    "timeseries_csv",
]


PCR_CSV_FIELDNAMES = [
    "date",
    "H_km",
    "t_fs",
    "T_K",
    "P_atm",
    "lambda_m",
    "n0_for_Pcr",
    "N2_fraction_initial",
    "O2_fraction_initial",
    "Ar_fraction_initial",
    "n2_mix_initial_1atm_cm2_W",
    "n2_prime_initial_m2_W",
    "Pcr_initial_W",
    "Pcr_initial_GW",
    "pulse_fwhm_fs",
    "Pcr_initial_times_fwhm_J",
    "Pcr_initial_times_fwhm_mJ",
]


def _time_row_for_csv(row):
    """把内部字段名映射为审稿要求的单位字段名。"""
    return {
        "date": row["date"],
        "H_km": row["H_km"],
        "T_K": row["T_K"],
        "P_atm": row["P_atm_table"],
        "Icrit_W_m2": row["Icrit_W_m2"],
        "Icrit_W_cm2": row["Icrit_W_cm2"],
        "t_s": row["t_s"],
        "t_fs": row["t_fs"],
        "I_W_m2": row["I_W_m2"],
        "I_W_cm2": row["I_W_cm2"],
        "W_N2_s^-1": row["W_N2_s-1"],
        "W_O2_s^-1": row["W_O2_s-1"],
        "W_Ar_s^-1": row["W_Ar_s-1"],
        "R_N2_m^-3_s^-1": row["R_N2_m-3_s-1"],
        "R_O2_m^-3_s^-1": row["R_O2_m-3_s-1"],
        "R_Ar_m^-3_s^-1": row["R_Ar_m-3_s-1"],
        "R_total_m^-3_s^-1": row["R_total_m-3_s-1"],
        "K_N2": row["K_N2"],
        "K_O2": row["K_O2"],
        "K_Ar": row["K_Ar"],
        "Up_eV": row["Up_eV"],
        "Ekin_N2_eV": row["Ekin_N2_eV"],
        "Ekin_O2_eV": row["Ekin_O2_eV"],
        "Ekin_Ar_eV": row["Ekin_Ar_eV"],
        "dne_N2_m-3": row["dne_N2_m-3"],
        "dne_O2_m-3": row["dne_O2_m-3"],
        "dne_Ar_m-3": row["dne_Ar_m-3"],
        "Te_inst_eV": row["Te_inst_eV"],
        "Te_inst_K": row["Te_inst_K"],
        "Te_cum_eV": row["Te_cum_eV"],
        "Te_cum_K": row["Te_cum_K"],
        "ne_N2_m^-3": row["ne_N2_m-3"],
        "ne_O2_m^-3": row["ne_O2_m-3"],
        "ne_Ar_m^-3": row["ne_Ar_m-3"],
        "ne_total_m^-3": row["ne_total_m-3"],
        "neutral_N2_m^-3": row["neutral_N2_m-3"],
        "neutral_O2_m^-3": row["neutral_O2_m-3"],
        "neutral_Ar_m^-3": row["neutral_Ar_m-3"],
        "neutral_total_m^-3": row["neutral_total_m-3"],
        "P_for_n2_atm": row["P_for_n2_atm"],
        "n2_prime_m2_W": row["n2_prime_m2_W"],
        "n2I": row["n2I"],
        "tauR_over_2necr": row["tauR_over_2necr"],
        "balance_residual": row["balance_residual"],
        "n_e_cr_m^-3": row["n_e_cr_m-3"],
        "ne_over_necr": row["ne_over_necr"],
    }


def _summary_row_for_csv(s):
    """把单个高度/日期的峰值结果映射到 critical_summary.csv。"""
    return {
        "date": s["date"],
        "H_km": s["H_km"],
        "T_K": s["T_K"],
        "P_atm": s["P_atm_table"],
        "Icrit_W_m2": s["I_peak_W_m2"],
        "Icrit_W_cm2": s["I_peak_W_cm2"],
        "Te0_final_eV": s["Te0_final_eV"],
        "Te0_final_K": s["Te0_final_K"],
        "Te_inst_peak_eV": s["Te_inst_peak_eV"],
        "Te_inst_peak_K": s["Te_inst_peak_K"],
        "mean_electron_energy_final_eV": s["mean_electron_energy_final_eV"],
        "n2_prime_at_peak_m2_W": s["n2_prime_at_peak_m2_W"],
        "n2I_at_peak": s["n2I_at_peak"],
        "tauR_over_2necr_at_peak": s["tauR_over_2necr_at_peak"],
        "residual": s["balance_residual_dimensionless"],
        "relative_residual": s["relative_residual"],
        "W_N2_at_peak_s^-1": s["W_N2_at_peak_s-1"],
        "W_O2_at_peak_s^-1": s["W_O2_at_peak_s-1"],
        "W_Ar_at_peak_s^-1": s["W_Ar_at_peak_s-1"],
        "R_N2_at_peak_m^-3_s^-1": s["R_N2_at_peak_m-3_s-1"],
        "R_O2_at_peak_m^-3_s^-1": s["R_O2_at_peak_m-3_s-1"],
        "R_Ar_at_peak_m^-3_s^-1": s["R_Ar_at_peak_m-3_s-1"],
        "R_total_at_peak_m^-3_s^-1": s["R_at_peak_total_m-3_s-1"],
        "ne_N2_at_peak_m^-3": s["ne_N2_at_peak_m-3"],
        "ne_O2_at_peak_m^-3": s["ne_O2_at_peak_m-3"],
        "ne_Ar_at_peak_m^-3": s["ne_Ar_at_peak_m-3"],
        "ne_total_at_peak_m^-3": s["ne_total_at_peak_m-3"],
        "neutral_N2_at_peak_m^-3": s["neutral_N2_at_peak_for_R_m-3"],
        "neutral_O2_at_peak_m^-3": s["neutral_O2_at_peak_for_R_m-3"],
        "neutral_Ar_at_peak_m^-3": s["neutral_Ar_at_peak_for_R_m-3"],
        "trial_count": s["trial_count"],
        "status": s["status"],
        "timeseries_csv": s.get("timeseries_csv", ""),
    }


def summary_warnings(s):
    """根据审稿意见生成需要人工复核的 warning。"""
    warnings = []

    if not (1.0e13 <= s["I_peak_W_cm2"] <= 1.0e14):
        warnings.append(
            "Icrit_W_cm2 不在 1e13~1e14 W/cm^2 典型夹持光强范围内，"
            "需检查 n2'、W(I)、n_e^cr 或单位。"
        )

    Te0 = s.get("Te0_final_eV", math.nan)
    if np.isfinite(Te0) and (Te0 > 10.0 or Te0 < 0.01):
        warnings.append(
            "Te0_final_eV 明显偏离 0.01~10 eV，需检查 K_i(t)、"
            "U_p(t)、新生电子动能 E_i(t)、显式欧拉新增电子密度和单位。"
        )
    elif not np.isfinite(Te0):
        warnings.append(
            "Te0_final_eV 为 NaN，需检查该时间序列是否没有新生电子。"
        )

    if s["W_O2_at_peak_s-1"] < min(
        s["W_N2_at_peak_s-1"],
        s["W_Ar_at_peak_s-1"],
    ):
        warnings.append(
            "O2 在当前强场下的 W 低于 N2 和 Ar，需复核 MO-PPT 参数或插值。"
        )

    return warnings


def write_timeseries_csv(path, rows, write_header=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if write_header else "a"

    with path.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TIME_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(_time_row_for_csv(row) for row in rows)


def append_rows_csv(path, rows, write_header=False):
    """兼容旧入口；实际写出使用 write_timeseries_csv()。"""
    write_timeseries_csv(path, rows, write_header=write_header)


def write_critical_summary_csv(path, summary_rows):
    """写出每个日期/高度一行的夹持光强汇总表。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_summary_row_for_csv(row) for row in summary_rows)


def write_pcr_csv(path, cases, lambda_m=800e-9, n0=N0_FOR_PCR):
    """写出不同日期/高度初始时刻的自聚焦临界功率 Pcr。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        pcr_row_for_case(case, lambda_m=lambda_m, n0=n0)
        for case in sorted(cases, key=lambda item: (item["date"], float(item["H_km"])))
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PCR_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_n2I_report(path, summary_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("n2'I balance report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Critical condition:")
    lines.append("    n2' * I = tau * R(I) / (2 * n_e^cr)")
    lines.append("")
    lines.append(f"default lambda = {LAMBDA_M:.6e} m")
    lines.append(f"n_e^cr = {N_E_CR_MODEL_M3:.6e} m^-3")
    lines.append(f"pulse FWHM = {PULSE_FWHM_S / 1e-15:.6f} fs")
    lines.append(f"pulse integration window = {T_END_PULSE_S / 1e-15:.6f} fs")
    lines.append(f"Euler dt = {DT_S / 1e-15:.6f} fs")
    lines.append(f"N_BALANCE_PERIODS = {N_BALANCE_PERIODS}")
    lines.append(f"tau = {TAU_BALANCE_S / 1e-15:.6f} fs")
    lines.append("")
    lines.append("Definitions:")
    lines.append("    W(I): single-particle ionization rate, unit s^-1")
    lines.append("    W_i[I(t)] is evaluated directly from PPT/MO-PPT at each time point;")
    lines.append("    no precomputed W table or W interpolation is used.")
    lines.append("    R_i(I,t) = W_i(I(t)) * n_i,neutral(t), unit m^-3 s^-1")
    lines.append("    R_total = sum_i R_i; neutral density is not multiplied twice.")
    lines.append("    n2'(t) = p_table * sum_i x_i(t) n2_i, where p_table is the")
    lines.append("    P(atm) value listed in the Chinese Word Table 1, and x_i")
    lines.append("    is the neutral-species fraction.")
    lines.append("    P/Pcr is not evaluated here; high-altitude sustainable filamentation")
    lines.append("    must be judged with an additional P/Pcr check.")
    lines.append("")
    lines.append("Units:")
    lines.append("    Icrit_W_m2: W/m^2; Icrit_W_cm2: W/cm^2")
    lines.append("    n2_prime_at_peak_m2_W: m^2/W")
    lines.append("    R_at_peak_total_m-3_s-1: m^-3 s^-1")
    lines.append("")
    lines.append("=" * 100)
    lines.append("")

    header = (
        "date\tH_km\tstatus\tlambda_m\tIcrit_W_m2\tIcrit_W_cm2\t"
        "n2_prime_at_peak_m2_W\tn2I_at_peak\t"
        "tauR_over_2necr_at_peak\tresidual\trelative_residual\t"
        "R_at_peak_total_m-3_s-1\tP_for_n2_at_peak_atm\ttrial_count"
    )

    lines.append(header)

    for s in summary_rows:
        lines.append(
            f"{s['date']}\t"
            f"{s['H_km']}\t"
            f"{s['status']}\t"
            f"{s['lambda_m']:.12e}\t"
            f"{s['I_peak_W_m2']:.12e}\t"
            f"{s['I_peak_W_cm2']:.12e}\t"
            f"{s['n2_prime_at_peak_m2_W']:.12e}\t"
            f"{s['n2I_at_peak']:.12e}\t"
            f"{s['tauR_over_2necr_at_peak']:.12e}\t"
            f"{s['balance_residual_dimensionless']:.12e}\t"
            f"{s['relative_residual']:.12e}\t"
            f"{s['R_at_peak_total_m-3_s-1']:.12e}\t"
            f"{s['P_for_n2_at_peak_atm']:.12e}\t"
            f"{s['trial_count']}"
        )

    lines.append("")
    lines.append("Warnings:")

    any_warning = False

    for s in summary_rows:
        warnings = summary_warnings(s)

        for warning in warnings:
            any_warning = True
            lines.append(
                f"WARNING {s['date']} H={s['H_km']} km: {warning}"
            )

    i_values = [s["I_peak_W_cm2"] for s in summary_rows]

    if i_values:
        lines.append("")
        lines.append("Scale comparison:")
        lines.append(
            "Icrit range = "
            f"{min(i_values):.6e} ~ {max(i_values):.6e} W/cm^2; "
            "this is in the 10^13 W/cm^2 filament clamping-intensity order."
        )
        te_values = [
            s["Te0_final_eV"]
            for s in summary_rows
            if np.isfinite(s["Te0_final_eV"])
        ]
        if te_values:
            lines.append(
                "Te0_final range = "
                f"{min(te_values):.6e} ~ {max(te_values):.6e} eV."
            )

        rel_span = (
            (max(i_values) - min(i_values))
            / max(abs(max(i_values)), 1.0e-300)
        )

        if rel_span < 1.0e-12:
            any_warning = True
            lines.append(
                "WARNING: 所有高度的 Icrit 几乎完全相同。"
                "这通常说明 n2' 与 R 都随中性密度近似线性缩放，"
                "但仍需检查 n2'、R(I) 中性密度因子、单位以及是否需要 P/Pcr 判据。"
            )

    if not any_warning:
        lines.append("No automatic warning triggered by the current checks.")

    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_filename_token(value):
    text = str(value)
    for ch in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", " "]:
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _height_filename_token(height_km):
    height = float(height_km)
    if height.is_integer():
        height_text = str(int(height))
    else:
        height_text = f"{height:g}".replace("-", "m").replace(".", "p")
    return f"H{height_text}km"


def time_height_csv_path(case_csv_dir, summary):
    date_token = _safe_filename_token(summary["date"])
    height_token = _height_filename_token(summary["H_km"])
    return Path(case_csv_dir) / f"{date_token}_{height_token}.csv"


def _column_from_rows(rows_or_dataframe, name):
    if hasattr(rows_or_dataframe, "__getitem__") and not isinstance(rows_or_dataframe, list):
        try:
            return np.asarray(rows_or_dataframe[name], dtype=float)
        except Exception:
            pass
    return np.asarray([row[name] for row in rows_or_dataframe], dtype=float)


def plot_Te_time_trend(rows_or_dataframe, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib 不可用，跳过 Te 图 {output_path}: {exc}")
        return

    t_fs = _column_from_rows(rows_or_dataframe, "t_fs")
    Te_cum = _column_from_rows(rows_or_dataframe, "Te_cum_eV")
    Te_inst = _column_from_rows(rows_or_dataframe, "Te_inst_eV")

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    mask_cum = np.isfinite(t_fs) & np.isfinite(Te_cum)
    mask_inst = np.isfinite(t_fs) & np.isfinite(Te_inst)
    plotted = False

    if np.any(mask_cum):
        ax.plot(t_fs[mask_cum], Te_cum[mask_cum], label="Te_cum", linewidth=1.8)
        plotted = True
    if np.any(mask_inst):
        ax.plot(
            t_fs[mask_inst],
            Te_inst[mask_inst],
            label="Te_inst",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
        )
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No finite Te data", ha="center", va="center")

    ax.set_xlabel("t_fs")
    ax.set_ylabel("Te_eV")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_Te0_final_vs_height(summary_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib 不可用，跳过 Te0 总图 {output_path}: {exc}")
        return

    dates = []
    for row in summary_rows:
        if row["date"] not in dates:
            dates.append(row["date"])

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    plotted = False
    for date in dates:
        subset = [row for row in summary_rows if row["date"] == date]
        subset.sort(key=lambda item: float(item["H_km"]))
        H = np.asarray([float(row["H_km"]) for row in subset], dtype=float)
        Te0 = np.asarray([float(row["Te0_final_eV"]) for row in subset], dtype=float)
        mask = np.isfinite(H) & np.isfinite(Te0)
        if np.any(mask):
            ax.plot(H[mask], Te0[mask], marker="o", label=date)
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No finite Te0 data", ha="center", va="center")

    ax.set_xlabel("H_km")
    ax.set_ylabel("Te0_final_eV")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_title("Final cumulative Te vs height")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ==========================================================
# 8. 主程序
# ==========================================================
def main():
    output_dir = BASE_DIR / "critical_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "critical_summary.csv"
    pcr_csv = output_dir / "pcr_initial_by_height_date.csv"
    report_txt = output_dir / "n2I_balance_report.txt"
    case_csv_dir = output_dir / "time_height_csv"
    te_plot_dir = output_dir / "Te_plots"
    case_csv_dir.mkdir(parents=True, exist_ok=True)
    te_plot_dir.mkdir(parents=True, exist_ok=True)

    print("===== 逐高度试差法计算夹持光强 =====")
    print("Critical equation:")
    print("    n2' I = tau R(I)/(2 n_e^cr)")
    print(f"lambda = {LAMBDA_M:.6e} m")
    print(f"n_e^cr = {N_E_CR_MODEL_M3:.6e} m^-3")
    print(f"pulse FWHM = {PULSE_FWHM_S / 1e-15:.3f} fs")
    print(f"Euler dt = {DT_S / 1e-15:.3f} fs")
    print(f"tau = {TAU_BALANCE_S / 1e-15:.3f} fs")
    print(f"I search range = {I_SEARCH_MIN:.3e} ~ {I_SEARCH_MAX:.3e} W/m^2")

    cases = build_atmosphere_cases()
    write_pcr_csv(pcr_csv, cases, lambda_m=LAMBDA_M, n0=N0_FOR_PCR)

    summary_rows = []

    for case in cases:
        print(f"\nTrial search: {case['date']}  H={case['H_km']} km")

        Icrit, summary, rows = find_critical_intensity_trial(
            case=case,
            lambda_m=LAMBDA_M,
            I_min=I_SEARCH_MIN,
            I_max=I_SEARCH_MAX,
        )

        case_csv_path = time_height_csv_path(case_csv_dir, summary)
        summary["timeseries_csv"] = str(case_csv_path)
        write_timeseries_csv(
            case_csv_path,
            rows,
            write_header=True,
        )
        summary_rows.append(summary)

        date_token = _safe_filename_token(summary["date"])
        height_token = _height_filename_token(summary["H_km"])
        te_plot_path = te_plot_dir / f"Te_trend_{date_token}_{height_token}.png"
        plot_Te_time_trend(
            rows,
            te_plot_path,
            (
                f"{summary['date']}  H={summary['H_km']} km  "
                f"Icrit={summary['I_peak_W_cm2']:.6e} W/cm^2"
            ),
        )

        print(
            f"  {summary['status']}: "
            f"Icrit = {Icrit:.12e} W/m^2 "
            f"= {Icrit / 1e4:.12e} W/cm^2"
        )
        print(
            f"  n2I = {summary['n2I_at_peak']:.12e}, "
            f"tauR/(2necr) = {summary['tauR_over_2necr_at_peak']:.12e}, "
            f"residual = {summary['balance_residual_dimensionless']:.12e}"
        )
        print(
            f"  W_peak: "
            f"N2={summary['W_N2_at_peak_s-1']:.12e}, "
            f"O2={summary['W_O2_at_peak_s-1']:.12e}, "
            f"Ar={summary['W_Ar_at_peak_s-1']:.12e}"
        )
        print(
            f"  R_peak total = {summary['R_at_peak_total_m-3_s-1']:.12e} m^-3 s^-1"
        )
        print(
            f"  Te0_final = {summary['Te0_final_eV']:.12e} eV "
            f"= {summary['Te0_final_K']:.12e} K"
        )
        print(f"  case CSV = {case_csv_path}")

        for warning in summary_warnings(summary):
            print(f"  WARNING: {warning}")

    write_critical_summary_csv(summary_csv, summary_rows)
    write_n2I_report(report_txt, summary_rows)
    plot_Te0_final_vs_height(
        summary_rows,
        te_plot_dir / "Te0_final_vs_height.png",
    )

    print("\n===== 高度/日期计算完成 =====")
    print(f"critical summary:      {summary_csv}")
    print(f"Pcr summary:           {pcr_csv}")
    print(f"time-height CSV folder: {case_csv_dir}")
    print(f"n2I report:            {report_txt}")
    print(f"Te plots:              {te_plot_dir}")


if __name__ == "__main__":
    main()
