# -*- coding: utf-8 -*-
"""
载流子迁移率计算核心函数 (形变势理论)

2D 公式:  mu = (2 e hbar^3 C) / (3 kB T |m*|^2 E1^2)

用法:  python -m spinlife.main mobility  (交互式)
"""

import os
import numpy as np

eV_to_J = 1.602176634e-19
hbar = 1.054571817e-34
kB = 1.380649e-23
e_ch = 1.602176634e-19
m0 = 9.10938356e-31


def read_POSCAR_A0(path):
    """从 POSCAR 读晶格面积 A0 (A^2)"""
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    a_vec = [float(x) * scale for x in lines[2].split()[:3]]
    b_vec = [float(x) * scale for x in lines[3].split()[:3]]
    a = np.sqrt(a_vec[0]**2 + a_vec[1]**2 + a_vec[2]**2)
    b = np.sqrt(b_vec[0]**2 + b_vec[1]**2 + b_vec[2]**2)
    return a * b


def fit_C2D(strain, E_total):
    """
    二次拟合 E vs strain -> d2E/deps2 (eV)
    返回: d2E, R^2, coeffs (A2, A1, A0)
    """
    coeffs = np.polyfit(strain, E_total, 2)
    A2, A1, A0 = coeffs
    d2E = 2 * A2
    E_fit = np.polyval(coeffs, strain)
    ss_res = np.sum((E_total - E_fit)**2)
    ss_tot = np.sum((E_total - np.mean(E_total))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    return d2E, r2, coeffs


def fit_E1(strain, E_band):
    """
    线性拟合 E_band vs strain -> E1 (eV)
    返回: E1, R^2, coeffs (E1, E0)
    """
    coeffs = np.polyfit(strain, E_band, 1)
    E1, E0 = coeffs
    E_fit = np.polyval(coeffs, strain)
    ss_res = np.sum((E_band - E_fit)**2)
    ss_tot = np.sum((E_band - np.mean(E_band))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    return E1, r2, coeffs


def calc_mu(C2D_Jm2, E1_eV, m_star_m0, T=300):
    """2D 形变势迁移率, 返回 (mu_cm2, tau_p_ps)"""
    m_kg = m_star_m0 * m0
    E1_J = E1_eV * eV_to_J
    mu_SI = (2 * e_ch * hbar**3 * C2D_Jm2) / (3 * kB * T * m_kg**2 * E1_J**2)
    mu_cm2 = mu_SI * 1e4
    # tau_p = mu * m* / e
    tau_p_s = mu_SI * m_kg / e_ch
    return mu_cm2, tau_p_s * 1e12


def C2D_Jm2_from_d2E(d2E_eV, A0_A2):
    """d2E/deps2 (eV) + A0 (A^2) -> C2D (J/m^2)"""
    return (d2E_eV / A0_A2) * eV_to_J / 1e-20


