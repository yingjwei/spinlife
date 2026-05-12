# -*- coding: utf-8 -*-
"""
spinlife — VASP PROCAR 自旋寿命计算流水线

用法:
  python -m spinlife.main PROCAR [--vbm N] [--cbm N] [--tau-p 0.1]
         [--k-range 0.05] [--T 300] [--output-dir .]

步骤:
  1. 解析 PROCAR → 提取 k 点、能带能量、自旋期望值
  2. 自动检测 VBM/CBM (或用户指定)
  3. 提取 SOC 劈裂的双带能量 + 自旋织构
  4. 拟合有效质量 m*
  5. 拟合 √(α²+β²) + 分离 α, β
  6. 计算自旋寿命 τ_s + PSH 周期 L_PSH
  7. 绘图 + 输出报告

输出:
  - 终端报告
  - spinlife_report.txt   — 文本报告
  - alpha_beta_fit.png    — α,β 拟合图 (含能带劈裂+自旋织构)
  - spin_texture.png      — 自旋织构图 (可选)
"""

import sys
import os
import numpy as np

# matplotlib 可选
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from .read_procar import PROCAR
from .fit_soc import fit_effmass, fit_alpha_beta, calc_spin_lifetime


def build_k_grid(kpoints):
    """
    从 k 点列表重建 2D 网格

    从 PROCAR 读取的 k 点是 2D 网格 (如 9×9=81 点),
    需要重建为 (nkx, nky) 网格用于自旋织构绘图.
    """
    kx_vals = sorted(set(kp[0] for kp in kpoints))
    ky_vals = sorted(set(kp[1] for kp in kpoints))
    nkx, nky = len(kx_vals), len(ky_vals)
    return kx_vals, ky_vals, nkx, nky


def find_nearest_k(kpoints, kx_target, ky_target):
    """找最近的 k 点索引"""
    best = 0
    best_d = 1e10
    for i, kp in enumerate(kpoints):
        d = (kp[0] - kx_target)**2 + (kp[1] - ky_target)**2
        if d < best_d:
            best_d = d
            best = i
    return best


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)

    procar_file = sys.argv[1] if os.path.exists(sys.argv[1]) else None
    if not procar_file:
        print(f"文件不存在: {sys.argv[1]}")
        sys.exit(1)

    # --- 参数 ---
    vbm_band = None
    cbm_band = None
    tau_p = 0.1         # ps, 动量散射时间 (默认)
    k_range = 0.05      # Å⁻¹, 拟合范围
    T = 300             # K
    output_dir = '.'

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--vbm' and i+1 < len(sys.argv):
            vbm_band = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--cbm' and i+1 < len(sys.argv):
            cbm_band = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--tau-p' and i+1 < len(sys.argv):
            tau_p = float(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--k-range' and i+1 < len(sys.argv):
            k_range = float(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--T' and i+1 < len(sys.argv):
            T = float(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--output-dir' and i+1 < len(sys.argv):
            output_dir = sys.argv[i+1]; i += 2
        else:
            i += 1

    os.makedirs(output_dir, exist_ok=True)

    # ========== 步骤 1: 解析 PROCAR ==========
    print("=" * 65)
    print("  spinlife — VASP 自旋寿命计算流水线")
    print("=" * 65)
    print(f"  PROCAR: {procar_file}")
    print(f"  τ_p   : {tau_p} ps")
    print(f"  拟合范围: ±{k_range} Å⁻¹")
    print(f"  T     : {T} K")

    procar = PROCAR(procar_file)
    procar.summary()

    # ========== 步骤 2: 检测 VBM/CBM ==========
    vbm, cbm = procar.find_vbm_cbm()
    if vbm_band is not None:
        vbm = vbm_band
        cbm = vbm + 1
    if cbm_band is not None:
        cbm = cbm_band
        vbm = cbm - 1

    print(f"\n  VBM: band {vbm}, CBM: band {cbm}")

    # ========== 步骤 3: 提取数据 ==========
    kpts = procar.get_kpoints_cart()
    E_vbm = procar.get_band_energy(vbm)
    E_cbm = procar.get_band_energy(cbm)
    sx_vbm, sy_vbm, sz_vbm = procar.get_spin(vbm)
    sx_cbm, sy_cbm, sz_cbm = procar.get_spin(cbm)

    # 重建 k 网格
    kx_vals, ky_vals, nkx, nky = build_k_grid(procar.kpoints)
    print(f"  网格: {nkx}×{nky} = {nkx*nky} k 点")

    # --- 初始化变量 (防止条件分支未赋值导致 UnboundLocalError) ---
    m_star_vbm = r2_m = None
    ab_norm = Delta = r2_ab = alpha = beta = ratio = None
    k0 = 0.0
    k_scan = E_up_scan = E_lo_scan = None
    sx_up_scan = sy_up_scan = None
    result = None
    soc_upper = vbm
    soc_lower = max(vbm - 1, 1)

    # ========== 步骤 4: 确定 k 空间切面 ==========
    # 尝试沿 ky=const 或 kx=const 找极值点
    # 首先检查各 ky 切片
    kx_vals = sorted(set(kpts[:, 0]))
    ky_vals = sorted(set(kpts[:, 1]))
    dkx = max(abs(kx_vals[1] - kx_vals[0]), 0.01) if len(kx_vals) > 1 else 0.01

    # 尝试 ky=0 切片, 若不存在则用全部 k 点
    idx_slice = [i for i, kp in enumerate(procar.kpoints) if abs(kp[1] - 0) < dkx/2]
    if len(idx_slice) < 3:
        idx_slice = list(range(procar.nk))

    k_slice = kpts[idx_slice, 0] if abs(kpts[idx_slice[0], 1]) < 0.01 else kpts[idx_slice, 1]
    E_slice = E_vbm[idx_slice]
    order = np.argsort(k_slice)
    k_scan = k_slice[order]
    E_scan = E_slice[order]
    k0 = k_scan[np.argmax(E_scan)]

    # 自动确定合适的 k_range (至少包含 3 个点)
    k_step = np.min(np.diff(sorted(set(k_scan)))) if len(set(k_scan)) > 1 else 0.05
    auto_k_range = max(k_range, k_step * 1.5)

    # ========== 步骤 5: 拟合有效质量 ==========
    m_star_vbm, r2_m, n_pts = fit_effmass(k_scan, E_scan, k0, auto_k_range)
    print(f"\n--- 有效质量 ---")
    if m_star_vbm:
        print(f"  m*(VBM) = {m_star_vbm:.2f} m0  (R^2={r2_m:.3f}, k0={k0:.3f})")
    else:
        print(f"  (拟合范围 ±{auto_k_range} 内仅 {n_pts} 个点, 尝试用更大的 k 范围)")

    # ========== 步骤 6: 拟合 α, β ==========
    print(f"\n--- SOC coefficients alpha, beta ---")

    E_up = procar.get_band_energy(soc_upper)
    E_lo = procar.get_band_energy(soc_lower)
    sx_up, sy_up, sz_up = procar.get_spin(soc_upper)
    sx_lo, sy_lo, sz_lo = procar.get_spin(soc_lower)

    # 沿选定的 k 方向拟合
    if len(idx_slice) >= 3:
        E_up_scan = E_up[idx_slice][order]
        E_lo_scan = E_lo[idx_slice][order]
        sx_up_scan = sx_up[idx_slice][order]
        sy_up_scan = sy_up[idx_slice][order]

        ab_norm, Delta, r2_ab = fit_alpha_beta(k_scan, E_up_scan, E_lo_scan, k0, auto_k_range)

        if ab_norm:
            print(f"  √(α²+β²) = {ab_norm*1000:.2f} meV·Å")
            print(f"  Δ        = {Delta*1000:.2f} meV")
            print(f"  R²       = {r2_ab:.4f}")

            # 从自旋织构分离 α/β
            # 方法: 对 ⟨σ_x⟩ vs k 和 ⟨σ_y⟩ vs k 做线性拟合,
            # 斜率比 = α/β (因为 ⟨σ_x⟩ = 2αk/Δ, ⟨σ_y⟩ = 2βk/Δ)
            dk = np.abs(k_scan - k0)
            near = dk <= auto_k_range
            if np.sum(near) >= 3:
                kn = k_scan[near] - k0
                p_x = np.polyfit(kn, sx_up_scan[near], 1)
                p_y = np.polyfit(kn, sy_up_scan[near], 1)
                slope_x, slope_y = p_x[0], p_y[0]
                ratio = slope_x / (slope_y + 1e-30) if abs(slope_y) > 1e-30 else 1e6
                # 也用 lower band 验证
                p_x2 = np.polyfit(kn, sx_lo[idx_slice][order][near], 1)
                p_y2 = np.polyfit(kn, sy_lo[idx_slice][order][near], 1)
                ratio2 = p_x2[0] / (p_y2[0] + 1e-30) if abs(p_y2[0]) > 1e-30 else 1e6
                if abs(ratio2) > abs(ratio):
                    ratio = ratio2

                if abs(ratio) > 1e-6 and abs(ratio) < 1e6:
                    alpha = ab_norm / np.sqrt(1 + 1/ratio**2)
                    beta = alpha / ratio
                    print(f"  ⟨σ_x⟩/⟨σ_y⟩ = {ratio:.3f}")
                    print(f"  α = {alpha*1000:.2f} meV·Å")
                    print(f"  β = {beta*1000:.2f} meV·Å")
                else:
                    alpha = beta = None
                    ratio = None

                # ========== 步骤 7: 自旋寿命 ==========
                print(f"\n--- Spin Lifetime ---")
                if alpha and m_star_vbm:
                    result = calc_spin_lifetime(alpha, beta, m_star_vbm, tau_p, T)
                    if result:
                        print(f"  τ_s ≈ {result['tau_s_ps']:.2f} ps")
                        print(f"  L_PSH ≈ {result['L_PSH_um']:.2f} μm")

    # ========== 绘图 ==========
    if _HAS_MPL and k_scan is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'spinlife — VBM Band {vbm}, SOC Partner Band {soc_lower}', fontsize=13)

        # (a) SOC 劈裂能带
        ax1 = axes[0, 0]
        ax1.plot(k_scan, E_up_scan*1000, 'o-', ms=4, label=f'Band {soc_upper}')
        ax1.plot(k_scan, E_lo_scan*1000, 's-', ms=4, label=f'Band {soc_lower}')
        ax1.axvspan(k0-auto_k_range, k0+auto_k_range, alpha=0.08, color='blue')
        ax1.axvline(k0, color='gray', ls='--', alpha=0.5)
        ax1.set_xlabel('k (A^-1)')
        ax1.set_ylabel('E (meV)')
        ax1.legend(fontsize=9)
        ax1.set_title('SOC-split Bands')

        # (b) DeltaE^2 vs k^2 拟合
        ax2 = axes[0, 1]
        dE = np.abs(E_up_scan - E_lo_scan) * 1000
        x = (k_scan - k0)**2
        ax2.plot(x, dE**2, 'o', ms=5)
        if ab_norm:
            coeffs = np.polyfit(x, dE**2, 1)
            xs = np.linspace(0, max(x)*1.05, 100)
            ax2.plot(xs, coeffs[0]*xs+coeffs[1], '-',
                     label=rf'ab^2={ab_norm**2*1e6:.1f} (meV.A)^2')
        ax2.set_xlabel('(k-k0)^2 (A^-2)')
        ax2.set_ylabel('ΔE² (meV²)')
        ax2.legend(fontsize=9)
        ax2.set_title('ΔE² Linear Fit')

        # (c) 自旋织构 ⟨σ_x⟩, ⟨σ_y⟩
        ax3 = axes[1, 0]
        ax3.plot(k_scan, sx_up_scan, 'o-', ms=4, label='<sx> upper')
        ax3.plot(k_scan, sy_up_scan, 's-', ms=4, label='<sy> upper')
        ax3.axvspan(k0-k_range, k0+k_range, alpha=0.08, color='blue')
        ax3.axhline(0, color='gray', lw=0.5)
        ax3.set_xlabel('k (A^-1)')
        ax3.set_ylabel('<sigma>')
        ax3.legend(fontsize=9)
        ax3.set_title(f'Spin Texture (Band {soc_upper})')

        # (d) 自旋织构 2D
        ax4 = axes[1, 1]
        # 重建 2D 网格
        if nkx * nky == procar.nk:
            S_mag = np.sqrt(sx_up**2 + sy_up**2 + sz_up**2)
            # 重排成网格
            S_grid = np.zeros((nky, nkx))
            for ik, kp in enumerate(procar.kpoints):
                ix = kx_vals.index(kp[0])
                iy = ky_vals.index(kp[1])
                S_grid[iy, ix] = S_mag[ik]
            im = ax4.imshow(S_grid, origin='lower',
                           extent=[min(kx_vals), max(kx_vals),
                                   min(ky_vals), max(ky_vals)],
                           cmap='RdYlBu_r')
            plt.colorbar(im, ax=ax4, label='|⟨σ⟩|')
            ax4.set_xlabel('k_x')
            ax4.set_ylabel('k_y')
            ax4.set_title('Spin Magnitude (2D)')

        plt.tight_layout()
        out_png = os.path.join(output_dir, 'spinlife_results.png')
        plt.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  [图片: {out_png}]")

    # ========== 保存文本报告 ==========
    report = os.path.join(output_dir, 'spinlife_report.txt')
    with open(report, 'w') as f:
        f.write("spinlife — VASP 自旋寿命计算报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"PROCAR: {procar_file}\n")
        f.write(f"τ_p: {tau_p} ps\n")
        f.write(f"T: {T} K\n\n")
        f.write(f"VBM: band {vbm}, CBM: band {cbm}\n")
        f.write(f"k grid: {nkx}x{nky}\n")
        if m_star_vbm is not None:
            f.write(f"m*(VBM): {m_star_vbm:.2f} m0  (R^2={r2_m:.3f})\n")
        if ab_norm is not None:
            f.write(f"sqrt(a^2+b^2): {ab_norm*1000:.2f} meV.A\n")
            f.write(f"Delta: {Delta*1000:.2f} meV\n")
        if alpha is not None and beta is not None:
            f.write(f"alpha: {alpha*1000:.2f} meV.A\n")
            f.write(f"beta: {beta*1000:.2f} meV.A\n")
            f.write(f"alpha/beta: {ratio:.3f}\n")
        if result is not None:
            f.write(f"τ_s: {result['tau_s_ps']:.2f} ps\n")
            f.write(f"L_PSH: {result['L_PSH_um']:.2f} μm\n")

    print(f"\n  [报告: {report}]")
    print("=" * 65)
