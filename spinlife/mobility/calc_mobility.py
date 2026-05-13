# -*- coding: utf-8 -*-
"""
载流子迁移率计算 (形变势理论)

2D 公式:  mu = (2 e hbar^3 C) / (3 kB T |m*|^2 E1^2)

用法:
  1. 在下方 [USER DATA] 填入你的数据
  2. 把 POSCAR 放在同目录下 (或修改 POSCAR_PATH)
  3. python -m spinlife.mobility.calc_mobility

输出: 终端 + mobility_report.txt + mobility_fit.png
"""

import numpy as np
import os, re

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ====================================================================
# [USER DATA] 在这里填入你的数据
# ====================================================================

TITLE = 'y direction'          # 方向标识

# --- 应变-总能量 (C2D) ---
# strain(%):  形变百分比 (-3 表示 -3%)
# E_total(eV): 对应 relax OUTCAR 的 energy without entropy (最后一个)
STRAIN_pct = [-3, -2, -1, 0, 1, 2, 3]
E_TOTAL_eV = [-277.6587049, -277.7078816, -277.7374091, -277.7485044,
              -277.7471965, -277.7327427, -277.7150140]

# --- 应变-VBM/CBM 能量 (E1) ---
# E_VBM(eV): SOC 计算中 VBM 能量 (可以是 eV 或 meV, 可空)
# E_CBM(eV): SOC 计算中 CBM 能量 (可以是 eV 或 meV, 可空)
STRAIN_E1_pct = [-3, -2, -1, 0, 1, 2, 3]
E_VBM_eV = []     # 填入数据或留空 []
E_CBM_eV = []     # 填入数据或留空 []

# --- 有效质量 (从 spinlife 拟合得到) ---
M_STAR_VBM = 4.32    # m0 单位, 空穴
M_STAR_CBM = None    # m0 单位, 电子, None 表示不计算

# --- 温度 ---
TEMPERATURE = 300     # K

# --- POSCAR 路径 (用于读 A0) ---
POSCAR_PATH = 'POSCAR'    # 或 '/full/path/to/POSCAR'

# ====================================================================
# 物理常数 (不要改)
# ====================================================================
eV_to_J = 1.602176634e-19
hbar = 1.054571817e-34
kB = 1.380649e-23
e_ch = 1.602176634e-19
m0 = 9.10938356e-31


def read_POSCAR_A0(path):
    """从 POSCAR 读晶格常数, 返回 a, b, A0(A^2)"""
    if not os.path.exists(path):
        print(f"  [POSCAR not found: {path}, 跳过 A0 读取]")
        return None, None, None
    with open(path, 'r') as f:
        lines = f.readlines()
    # L2: scale factor
    scale = float(lines[1].strip())
    # L3-L5: lattice vectors
    a_vec = [float(x) * scale for x in lines[2].split()[:3]]
    b_vec = [float(x) * scale for x in lines[3].split()[:3]]
    # 2D: area = |a x b|
    a = np.sqrt(a_vec[0]**2 + a_vec[1]**2 + a_vec[2]**2)
    b = np.sqrt(b_vec[0]**2 + b_vec[1]**2 + b_vec[2]**2)
    A0 = a * b
    return a, b, A0


def polyfit_quad(x, y):
    """二次拟合: y = A2 x^2 + A1 x + A0"""
    coeffs = np.polyfit(x, y, 2)
    y_fit = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    return coeffs, r2


def polyfit_lin(x, y):
    """线性拟合: y = E1 x + E0"""
    coeffs = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    return coeffs, r2


def calc_mu(C2D_Jm2, E1_eV, m_star_m0, T):
    """2D 形变势迁移率, 返回 cm^2/V.s"""
    m_kg = m_star_m0 * m0
    E1_J = E1_eV * eV_to_J
    mu_SI = (2 * e_ch * hbar**3 * C2D_Jm2) / (3 * kB * T * m_kg**2 * E1_J**2)
    return mu_SI * 1e4  # -> cm^2/V.s


# ====================================================================
# 主计算
# ====================================================================
def main():
    print("=" * 65)
    print("  spinlife.mobility -- 载流子迁移率计算")
    print("=" * 65)

    x = np.array(STRAIN_pct) / 100   # % -> 小数

    # ---- C2D ----
    E = np.array(E_TOTAL_eV)
    coeffs, r2 = polyfit_quad(x, E)
    A2, A1, A0_fit = coeffs
    d2E = 2 * A2

    print(f"\n--- C2D ({TITLE}) ---")
    print(f"  Fit: E = {A2:.4f}*eps^2 + {A1:.4f}*eps + {A0_fit:.6f}")
    print(f"  d2E/deps2 = {d2E:.4f} eV,  R^2 = {r2:.6f}")

    a, b, A0 = read_POSCAR_A0(POSCAR_PATH)
    if A0:
        print(f"  a = {a:.4f} A,  b = {b:.4f} A,  A0 = {A0:.2f} A^2")
        C2D_eV_A2 = d2E / A0
        C2D_Jm2 = C2D_eV_A2 * eV_to_J / 1e-20
        print(f"  C2D = {d2E:.4f} / {A0:.2f} = {C2D_eV_A2:.4f} eV/A^2")
        print(f"       = {C2D_Jm2:.2f} J/m^2")
    else:
        C2D_Jm2 = d2E
        print(f"  (C2D 未计算: POSCAR 未找到)")

    # ---- E1 ----
    x_e1 = np.array(STRAIN_E1_pct) / 100

    E1_vbm = E1_cbm = None
    vbm_r2 = cbm_r2 = None
    vbm_coeffs = cbm_coeffs = None

    if len(E_VBM_eV) > 0:
        y_vbm = np.array(E_VBM_eV)
        vbm_coeffs, vbm_r2 = polyfit_lin(x_e1, y_vbm)
        E1_vbm = vbm_coeffs[0]
        print(f"\n--- E1 (VBM, 空穴 {TITLE}) ---")
        print(f"  E1 = {E1_vbm:.4f} eV,  R^2 = {vbm_r2:.6f}")

    if len(E_CBM_eV) > 0:
        y_cbm = np.array(E_CBM_eV)
        cbm_coeffs, cbm_r2 = polyfit_lin(x_e1, y_cbm)
        E1_cbm = cbm_coeffs[0]
        print(f"\n--- E1 (CBM, 电子 {TITLE}) ---")
        print(f"  E1 = {E1_cbm:.4f} eV,  R^2 = {cbm_r2:.6f}")

    # ---- 迁移率 ----
    print(f"\n--- 迁移率 (T={TEMPERATURE} K) ---")

    if C2D_Jm2 and E1_vbm is not None and M_STAR_VBM is not None:
        mu_h = calc_mu(C2D_Jm2, E1_vbm, M_STAR_VBM, TEMPERATURE)
        print(f"  空穴 (VBM):  mu = {mu_h:.2f} cm^2/V.s")
        print(f"    m* = {M_STAR_VBM:.3f} m0,  E1 = {E1_vbm:.4f} eV")
    else:
        mu_h = None

    if C2D_Jm2 and E1_cbm is not None and M_STAR_CBM is not None:
        mu_e = calc_mu(C2D_Jm2, E1_cbm, M_STAR_CBM, TEMPERATURE)
        print(f"  电子 (CBM):  mu = {mu_e:.2f} cm^2/V.s")
        print(f"    m* = {M_STAR_CBM:.3f} m0,  E1 = {E1_cbm:.4f} eV")
    else:
        mu_e = None

    # ---- 绘图 ----
    if _HAS_MPL:
        ncol = 1 + (1 if len(E_VBM_eV) > 0 else 0)
        if ncol > 0:
            fig, axes = plt.subplots(1, ncol, figsize=(6*ncol, 5))
            if ncol == 1:
                axes = [axes]
            c = 0

            # C2D 图
            ax = axes[c]; c += 1
            xs = np.linspace(min(x), max(x), 200)
            ys = np.polyval(coeffs, xs)
            ax.plot(STRAIN_pct, E, 'o', ms=8, label=f'{TITLE} data')
            ax.plot(xs*100, ys, '-',
                    label=rf'fit: {A2:.2f}$\varepsilon^2$ + {A1:.2f}$\varepsilon$')
            ax.axhline(E[len(E)//2], color='gray', ls='--', alpha=0.4)
            ax.set_xlabel('Strain (%)')
            ax.set_ylabel('Total Energy (eV)')
            ax.legend(fontsize=9)
            ax.set_title(f'$C_{{2D}}$ Fit ({TITLE})')
            ax.grid(alpha=0.3)

            # E1 图
            if vbm_coeffs is not None:
                ax = axes[c]; c += 1
                y_vbm = np.array(E_VBM_eV)
                xs = np.linspace(min(x_e1), max(x_e1), 200)
                ys = np.polyval(vbm_coeffs, xs)
                ax.plot(STRAIN_E1_pct, y_vbm, 's', ms=8, label='VBM')
                if len(E_CBM_eV) > 0:
                    y_cbm = np.array(E_CBM_eV)
                    ax.plot(STRAIN_E1_pct, y_cbm, '^', ms=8, label='CBM')
                    ys2 = np.polyval(cbm_coeffs, xs)
                    ax.plot(xs*100, ys2, '--')
                ax.plot(xs*100, ys, '-',
                        label=rf'E1 = {E1_vbm:.3f} eV')
                ax.axhline(y_vbm[len(y_vbm)//2], color='gray', ls='--', alpha=0.4)
                ax.set_xlabel('Strain (%)')
                ax.set_ylabel('Band Energy (eV)')
                ax.legend(fontsize=9)
                ax.set_title(f'$E_1$ Fit ({TITLE})')
                ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig('mobility_fit.png', dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  [Plot: mobility_fit.png]")

    # ---- 报告 ----
    with open('mobility_report.txt', 'w') as f:
        f.write("spinlife.mobility -- 载流子迁移率报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"Title: {TITLE}\n")
        f.write(f"T: {TEMPERATURE} K\n")
        f.write(f"Formula: mu = 2e*hbar^3*C / (3*kB*T*m*^2*E1^2)\n\n")
        if A0:
            f.write(f"a = {a:.4f} A, b = {b:.4f} A, A0 = {A0:.2f} A^2\n")
        f.write(f"C2D = {C2D_Jm2:.4f} J/m^2\n" if isinstance(C2D_Jm2, float) else "")
        f.write(f"d2E/deps2 = {d2E:.4f} eV, R^2 = {r2:.6f}\n")
        if E1_vbm is not None:
            f.write(f"E1_VBM = {E1_vbm:.4f} eV, R^2 = {vbm_r2:.6f}\n")
        if E1_cbm is not None:
            f.write(f"E1_CBM = {E1_cbm:.4f} eV, R^2 = {cbm_r2:.6f}\n")
        f.write(f"\n--- Results ---\n")
        if mu_h:
            f.write(f"hole (VBM): mu = {mu_h:.2f} cm^2/V.s\n")
        if mu_e:
            f.write(f"electron (CBM): mu = {mu_e:.2f} cm^2/V.s\n")

    print(f"\n  [Report: mobility_report.txt]")
    print("=" * 65)


if __name__ == '__main__':
    main()
