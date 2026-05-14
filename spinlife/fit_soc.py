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


def fit_alpha_beta_band_average(k, E_upper, E_lower, k0=None, k_range=0.05, max_iter=5):
    """
    Band-average iterative fitting for √(α²+β²) and m* (recommended).

    Rashba/Dresselhaus SOC model:
      E_±(k) = E₀ + A·(k-k₀)² ± α_eff·|k-k₀|

    Method:
      Step 1: E_avg = (E_upper + E_lower)/2 → purely parabolic
              → fit parabola → m* = 3.81/|A|
      Step 2: |E_up - E_lo|/2 = α_eff·|k-k₀|
              → linear fit → √(α²+β²)
      Step 3: Iterate: remove linear term, re-average, refine m*

    Advantage: not limited by SOC gap size, signal is ~10x larger than ΔE² method.

    Returns
    -------
    m_star : float — effective mass (m₀)
    ab_norm : float — √(α²+β²) (eV·Å)
    r2_parab : float — parabolic fit R²
    r2_linear : float — linear fit R²
    n_pts : int — number of fitting points
    """
    dk = np.abs(k - k0) if k0 is not None else np.abs(k)
    mask = dk <= k_range
    if np.sum(mask) < 3:
        return None, None, 0, 0, 0

    k_rel = k[mask] - (k0 or 0)
    k2 = k_rel ** 2
    E_up = E_upper[mask]
    E_lo = E_lower[mask]

    # Step 1: Average → parabola → m*
    E_avg = (E_up + E_lo) / 2.0
    coeffs = np.polyfit(k2, E_avg, 1)
    A, E0 = coeffs[0], coeffs[1]
    m_star = 3.81 / abs(A) if abs(A) > 1e-12 else None

    E_avg_fit = A * k2 + E0
    ss_res = np.sum((E_avg - E_avg_fit)**2)
    ss_tot = np.sum((E_avg - np.mean(E_avg))**2)
    r2_parab = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0

    # Step 2: Residual → linear fit → √(α²+β²)
    abs_k = np.abs(k_rel)
    dE_half = np.abs(E_up - E_lo) / 2.0

    nonzero = abs_k > 1e-10
    if np.sum(nonzero) < 2:
        return m_star, None, r2_parab, 0, np.sum(mask)

    coeffs_lin = np.polyfit(abs_k[nonzero], dE_half[nonzero], 1)
    ab_norm = abs(coeffs_lin[0])

    dE_fit = np.polyval(coeffs_lin, abs_k[nonzero])
    ss_res = np.sum((dE_half[nonzero] - dE_fit)**2)
    ss_tot = np.sum((dE_half[nonzero] - np.mean(dE_half[nonzero]))**2)
    r2_linear = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0

    # Step 3: Iterative refinement: remove linear SOC term, re-fit parabola on single band
    for _ in range(max_iter):
        # Correct upper band: remove SOC linear term -> should be pure parabola
        E_up_corr = E_up - ab_norm * abs_k
        coeffs_new = np.polyfit(k2, E_up_corr, 1)
        A_new = coeffs_new[0]
        if abs(A_new - A) / (abs(A) + 1e-30) < 1e-4:
            break
        A = A_new
        E0 = coeffs_new[1]
        m_star = 3.81 / abs(A) if abs(A) > 1e-12 else None

    # Compute final m* and update R² from corrected single-band fit
    if m_star:
        E_corr = E_up - ab_norm * abs_k  # SOC-removed upper band
        E_parab_fit = A * k2 + E0
        ss_res = np.sum((E_corr - E_parab_fit)**2)
        ss_tot = np.sum((E_corr - np.mean(E_corr))**2)
        r2_parab = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0

    return m_star, ab_norm, r2_parab, r2_linear, np.sum(mask)



def calc_spin_lifetime(alpha=None, beta=None, m_star=None, tau_p=None, T=300, ab_norm=None):
    """
    自旋寿命 (DP 机制简化模型)

    支持两种调用方式:
      1. (alpha, beta, m_star, tau_p, T) — 已知 α/β 比值
      2. (ab_norm, m_star, tau_p, T) — 仅知 √(α²+β²) 时

    参数:
      alpha, beta: meV·Å (若不可用传 None)
      m_star: 有效质量 (m₀ 单位)
      tau_p: 动量散射时间 (ps)
      T: 温度 (K)
      ab_norm: √(α²+β²) in meV·Å (优先使用)

    输出:
      - L_PSH (μm): PSH 周期
      - tau_s (ps): DP 自旋寿命
    """
    if m_star is None or tau_p is None:
        return None

    # 优先使用 ab_norm (能带平均法结果), 其次 α/β
    if ab_norm is not None:
        ab = ab_norm * 1e-3  # meV·Å → eV·Å
        alpha_eff = ab_norm * 1e-3  # 保守估计: α_eff = √(α²+β²)
    elif alpha is not None:
        ab = np.sqrt(alpha**2 + beta**2) if beta else abs(alpha)
        alpha_eff = abs(alpha - beta) if beta else abs(alpha)
        if alpha_eff < 1e-12:
            alpha_eff = ab * 0.01
        ab = ab * 1e-3  # meV·Å → eV·Å
    else:
        return None

    if ab < 1e-15 or tau_p < 1e-15:
        return None

    # --- PSH 周期: L_PSH = πℏ²/(m*|α|) ---
    # SI 单位: α_eV-A · 1.602e-19(J/eV) · 1e-10(m/Å) = α_Jm
    prefac = np.pi * hbar**2 / m0
    if alpha is not None:
        alpha_for_lpsh = abs(alpha) * 1e-3      # meV.A -> eV.A, use |alpha|
    else:
        alpha_for_lpsh = ab                       # eV.A, proxy via sqrt(a^2+b^2)
    alpha_Jm = alpha_for_lpsh * eV_to_J * 1e-10
    L_PSH_m = prefac / (m_star * abs(alpha_Jm))
    L_PSH_um = L_PSH_m * 1e6

    # --- DP 机制自旋寿命 ---
    # τ_s ≈ ℏ² / (2 · m* · m₀ · α_eff² · τ_p)
    # α_eff_Jm: same unit conversion as above
    alpha_eff_Jm = alpha_eff * eV_to_J * 1e-10  # alpha_eff is in eV·Å
    tau_p_s = tau_p * 1e-12
    tau_s_s = hbar**2 / (2 * m_star * m0 * alpha_eff_Jm**2 * tau_p_s)
    tau_s_ps = tau_s_s * 1e12

    # 自旋进动频率估算
    omega_avg = 2 * ab * eV_to_J * 1e-10 / hbar  # rad/s
    omega_tau_p = omega_avg * tau_p_s

    # alpha_eff 恢复为 meV·Å 输出
    alpha_eff_meva = alpha_eff * 1e3 if alpha_eff is not None else None

    return {
        'tau_s_ps': tau_s_ps,
        'L_PSH_um': L_PSH_um,
        'alpha_eff_meVA': alpha_eff_meva,
        'omega_tau_p': omega_tau_p,
        'ab_norm_meVA': ab * 1e3,
    }
