# -*- coding: utf-8 -*-
"""
α, β 拟合 + 自旋寿命 (DP/D'yakonov-Perel 机制)

理论基础:
  PPTX Slide 7-9, 15-16:
  1. 能带劈裂 → √(α²+β²)
  2. 自旋织构 → α/β 比值
  3. 自旋扩散方程 → τ_s

输入:
  - PROCAR (ik→ik, iband) 的能带 + 自旋数据
  - 用户指定的 VBM/CBM 能带序号
  - 动量散射时间 τ_p (实验或计算值)
"""

import numpy as np
from scipy.optimize import root_scalar

# 物理常数
hbar = 6.582119569e-16       # eV·s
eV_to_J = 1.602176634e-19    # J/eV
m0 = 9.10938356e-31           # kg (电子静质量)
A_to_m = 1e-10                # Å → m


def fit_effmass(k, E, k0, k_range=0.05):
    """
    抛物线拟合有效质量: E(k) ≈ E₀ + ℏ²(k-k₀)²/(2m*)
    m*/m₀ = 3.81 / |A|, A = ℏ²/(2m*)
    """
    dk = np.abs(k - k0)
    mask = dk <= k_range
    if np.sum(mask) < 3:
        return None, 0, 0

    k_fit = k[mask] - k0
    E_fit = E[mask]
    coeffs = np.polyfit(k_fit**2, E_fit, 1)
    A = coeffs[0]

    m_star = 3.81 / abs(A) if abs(A) > 1e-12 else 999

    pred = A * k_fit**2 + coeffs[1]
    ss_res = np.sum((E_fit - pred)**2)
    ss_tot = np.sum((E_fit - np.mean(E_fit))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0

    return m_star, r2, np.sum(mask)


def fit_alpha_beta(k, E_upper, E_lower, k0, k_range=0.05):
    """
    从能带劈裂拟合 √(α²+β²)

    方法:
      ΔE = |E₂ - E₁|
      ΔE² = 4Δ² + 4(α²+β²)k²
      → 线性拟合 ΔE² vs k²
    """
    dk = np.abs(k - k0)
    mask = dk <= k_range
    if np.sum(mask) < 3:
        return None, None, 0

    k_rel = dk[mask]
    dE = np.abs(E_upper[mask] - E_lower[mask])

    coeffs = np.polyfit(k_rel**2, dE**2, 1)
    slope, intercept = coeffs[0], coeffs[1]

    ab_norm = np.sqrt(max(slope, 0)) / 2   # √(α²+β²)
    Delta = np.sqrt(max(intercept, 0)) / 2  # Δ

    pred = slope * k_rel**2 + intercept
    ss_res = np.sum((dE**2 - pred)**2)
    ss_tot = np.sum((dE**2 - np.mean(dE**2))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0

    return ab_norm, Delta, r2


def separate_alpha_beta(ab_norm, sx, sy, k0, k_range=0.05):
    """
    从自旋期望值分离 α, β:
    ⟨σ_x⟩/⟨σ_y⟩ = α/β  (k→0 极限)
    """
    dk = np.abs(k0)  # 假设 k 坐标已经对齐
    # 注意: sx, sy 是 k 点的数组
    # 取 |k-k0| 最小的点
    if len(sx) < 1:
        return None, None, None

    # 用 k0 附近值
    w = 1.0 / (np.abs(np.arange(len(sx)) - np.argmin(np.abs(k0 - 0))) + 1)
    # 简化: 直接取 k=k₀ 附近的自旋值
    closest = np.argmin(np.abs(np.linspace(-0.5, 0.5, len(sx))))

    # 更稳健: 取近 k₀ 处的平均
    return None  # 将在 pipeline 中处理


def calc_spin_lifetime(alpha, beta, m_star, tau_p, T=300):
    """
    自旋寿命 (DP 机制简化模型)

    所有输入:
      alpha, beta: meV·Å
      m_star: 有效质量 (m₀ 单位)
      tau_p: 动量散射时间 (ps)
      T: 温度 (K)

    输出:
      - L_PSH (μm): PSH 周期
      - tau_s (ps): DP 自旋寿命
    """
    if m_star is None or alpha is None:
        return None

    ab = np.sqrt(alpha**2 + beta**2) if beta else abs(alpha)
    if ab < 1e-12 or tau_p < 1e-15:
        return None

    # --- PSH 周期: L_PSH = πℏ²/(m*|α|) ---
    # SI 单位制:
    #   ℏ = 1.0546e-34 J·s
    #   m₀ = 9.109e-31 kg
    #   α_J·m = α_meVA · 1e-3(eV/meV) · 1.602e-19(J/eV) · 1e-10(m/Å)
    #         = α_meVA · 1.602e-32 J·m
    # L (m) = πℏ²/(m*m₀ · |α|_Jm)
    prefac = np.pi * hbar**2 / m0  # m²/s² · J²·s²/kg = J²·m²/J = J·m²
    alpha_Jm = ab * 1e-3 * eV_to_J * 1e-10
    L_PSH_m = prefac / (m_star * abs(alpha_Jm))
    L_PSH_um = L_PSH_m * 1e6

    # --- DP 机制自旋寿命 ---
    # alpha_eff = |α - β|: 当 α≈β (SU(2)对称), τ_s→∞
    alpha_eff = abs(alpha - beta) if beta else abs(alpha)
    if alpha_eff < 1e-12:
        alpha_eff = ab * 0.01  # 避免发散

    # τ_s ≈ ℏ² / (2 m* m₀ α_eff² τ_p)  — DP 估算
    # τ_s (s) = ℏ² / (2 * m* * m₀ * (α_eff_Jm)² * τ_p_s)
    alpha_eff_Jm = alpha_eff * 1e-3 * eV_to_J * 1e-10
    tau_p_s = tau_p * 1e-12
    tau_s_s = hbar**2 / (2 * m_star * m0 * alpha_eff_Jm**2 * tau_p_s)
    tau_s_ps = tau_s_s * 1e12

    # 自旋进动频率估算
    omega_avg = 2 * ab * 1e-3 * eV_to_J * 1e-10 / hbar  # rad/s
    omega_tau_p = omega_avg * tau_p_s  # Ω·τ_p (无量纲)

    return {
        'tau_s_ps': tau_s_ps,
        'L_PSH_um': L_PSH_um,
        'alpha_eff_meVA': alpha_eff * 1e3 if abs(alpha_eff) < 1 else alpha_eff,
        'omega_tau_p': omega_tau_p,
    }
