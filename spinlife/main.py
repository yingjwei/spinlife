# -*- coding: utf-8 -*-
"""
spinlife — VASP PROCAR 自旋寿命计算流水线

用法:
  python -m spinlife.main PROCAR [--vbm N] [--cbm N] [--tau-p 0.1]
         [--k-range 0.05] [--T 300] [--output-dir .]
         [--soc-vbm UPPER LOWER] [--soc-cbm UPPER LOWER]
         [--list-bands]

步骤:
  1. 解析 PROCAR -> k 点、能带能量、自旋期望值
  2. 自动检测 VBM/CBM (或用户指定)
  3. 输出 Γ 点能带表, 交互式选择 SOC 带对
  4. 拟合有效质量 m*
  5. 拟合 sqrt(a^2+b^2) + 分离 alpha, beta
  6. 计算自旋寿命 tau_s + PSH 周期 L_PSH
  7. 绘图 + 输出报告

输出:
  - 终端报告
  - spinlife_report.txt   文本报告
  - spinlife_results.png  拟合图 (VBM + CBM 双列)
  - band_table.txt        Γ 点能带数据
"""

import sys
import os
import numpy as np

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
    kx_vals = sorted(set(kp[0] for kp in kpoints))
    ky_vals = sorted(set(kp[1] for kp in kpoints))
    nkx, nky = len(kx_vals), len(ky_vals)
    return kx_vals, ky_vals, nkx, nky


def list_bands(procar, output_dir='.'):
    gamma_ik = 0
    min_dist = 1e10
    for ik in range(1, procar.nk + 1):
        kp = procar.kpoints[ik - 1]
        d = kp[0]**2 + kp[1]**2 + kp[2]**2
        if d < min_dist:
            min_dist = d
            gamma_ik = ik

    print(f"\n{'Band':>6} {'E (eV)':>12} {'Occ':>8}   Note")
    print("-" * 45)
    rows = []
    vbm_found = False
    for ib in range(1, procar.nbands + 1):
        E = procar.bands[(gamma_ik, ib)]['energy']
        occ = procar.bands[(gamma_ik, ib)]['occ']
        note = ''
        if not vbm_found and occ > 0.5 and ib < procar.nbands \
                and procar.bands[(gamma_ik, ib + 1)]['occ'] < 0.5:
            note = '<-- VBM'
            vbm_found = True
        elif vbm_found and occ < 0.5 and not note:
            note = '<-- CBM'
            vbm_found = False
        print(f"{ib:>6} {E:>12.4f} {occ:>8.4f}  {note}")
        rows.append((ib, E, occ))

    path = os.path.join(output_dir, 'band_table.txt')
    with open(path, 'w') as f:
        f.write("# Band  Energy(eV)  Occupation  Note\n")
        for ib, E, occ in rows:
            f.write(f"{ib} {E:.6f} {occ:.6f}\n")
    print(f"\n  [band_table.txt saved]")
    return gamma_ik


def get_k_slice(kpts, procar):
    kx_vals = sorted(set(kpts[:, 0]))
    ky_vals = sorted(set(kpts[:, 1]))
    dkx = max(abs(kx_vals[1] - kx_vals[0]), 0.01) if len(kx_vals) > 1 else 0.01

    idx_slice = [i for i, kp in enumerate(procar.kpoints)
                 if abs(kp[1] - 0) < dkx / 2]
    if len(idx_slice) < 3:
        idx_slice = list(range(procar.nk))

    k_slice = (kpts[idx_slice, 0]
               if abs(kpts[idx_slice[0], 1]) < 0.01
               else kpts[idx_slice, 1])
    order = np.argsort(k_slice)
    return idx_slice, k_slice[order], order


def run_soc_fit(label, procar, kpts, idx_slice, k_scan, order,
                mstar_band, soc_upper, soc_lower,
                tau_p, T, k_range, extrema_type='max'):
    """
    对指定 SOC 带对运行完整拟合流程

    Parameters
    ----------
    label : str           标签 (如 'VBM', 'CBM')
    mstar_band : int      用于有效质量拟合的能带
    soc_upper, soc_lower : int  SOC 带对
    extrema_type : 'max' | 'min'  极值类型
    """
    print(f"\n{'='*62}")
    print(f"  {label}: m* band={mstar_band}, "
          f"SOC pair={soc_upper}/{soc_lower}")
    print(f"{'='*62}")

    # --- k 切片 + 极值点 ---
    E_ext = procar.get_band_energy(mstar_band)
    E_slice = E_ext[idx_slice][order]

    if extrema_type == 'max':
        k0 = k_scan[np.argmax(E_slice)]
    else:
        k0 = k_scan[np.argmin(E_slice)]

    # --- 有效质量 ---
    k_step = (np.min(np.diff(sorted(set(k_scan))))
              if len(set(k_scan)) > 1 else 0.05)
    auto_k_range = max(k_range, k_step * 1.5)

    m_star, r2_m, n_pts = fit_effmass(k_scan, E_slice, k0, auto_k_range)

    if m_star:
        print(f"  m* = {m_star:.2f} m0  (R^2={r2_m:.3f}, "
              f"k0={k0:.3f}, range=+/-{auto_k_range:.3f})")
    else:
        print(f"  m*: fit failed ({n_pts} pts in +/-{auto_k_range})")

    # --- SOC 劈裂 + alpha/beta ---
    E_up = procar.get_band_energy(soc_upper)
    E_lo = procar.get_band_energy(soc_lower)
    sx_up, sy_up, sz_up = procar.get_spin(soc_upper)
    sx_lo, sy_lo, _ = procar.get_spin(soc_lower)

    res = {
        'label': label,
        'bands': (soc_upper, soc_lower),
        'm_star': m_star,
        'k0': k0,
        'auto_k_range': auto_k_range,
    }

    E_up_scan = E_up[idx_slice][order]
    E_lo_scan = E_lo[idx_slice][order]
    sx_up_scan = sx_up[idx_slice][order]
    sy_up_scan = sy_up[idx_slice][order]
    sx_lo_scan = sx_lo[idx_slice][order]

    res['k_scan'] = k_scan
    res['E_up_scan'] = E_up_scan
    res['E_lo_scan'] = E_lo_scan
    res['sx_up_scan'] = sx_up_scan
    res['sy_up_scan'] = sy_up_scan

    ab_norm, Delta, r2_ab = fit_alpha_beta(
        k_scan, E_up_scan, E_lo_scan, k0, auto_k_range)

    if ab_norm:
        print(f"  sqrt(a^2+b^2) = {ab_norm*1000:.2f} meV.A")
        print(f"  Delta         = {Delta*1000:.2f} meV  "
              f"(R^2={r2_ab:.4f})")
        res.update(ab_norm=ab_norm, Delta=Delta, r2_ab=r2_ab)

        # 自旋织构 -> alpha/beta 分离
        dk = np.abs(k_scan - k0)
        near = dk <= auto_k_range
        if np.sum(near) >= 3:
            kn = k_scan[near] - k0
            p_x = np.polyfit(kn, sx_up_scan[near], 1)
            p_y = np.polyfit(kn, sy_up_scan[near], 1)
            ratio = p_x[0] / (p_y[0] + 1e-30) if abs(p_y[0]) > 1e-30 else 1e6

            p_x2 = np.polyfit(kn, sx_lo_scan[near], 1)
            p_y2 = np.polyfit(kn, sy_lo_scan[near], 1)
            ratio2 = p_x2[0] / (p_y2[0] + 1e-30) if abs(p_y2[0]) > 1e-30 else 1e6
            if abs(ratio2) > abs(ratio):
                ratio = ratio2

            if abs(ratio) > 1e-6 and abs(ratio) < 1e6:
                alpha = ab_norm / np.sqrt(1 + 1 / ratio**2)
                beta = alpha / ratio
                print(f"  <sx>/<sy> ratio = {ratio:.3f}")
                print(f"  alpha = {alpha*1000:.2f} meV.A")
                print(f"  beta  = {beta*1000:.2f} meV.A")
                res.update(alpha=alpha, beta=beta, ratio=ratio)

                # --- 自旋寿命 ---
                if m_star:
                    spin = calc_spin_lifetime(alpha, beta, m_star, tau_p, T)
                    if spin:
                        print(f"  tau_s  = {spin['tau_s_ps']:.2f} ps")
                        print(f"  L_PSH  = {spin['L_PSH_um']:.2f} um")
                        res['spin'] = spin
            else:
                print(f"  (spin ratio unstable: {ratio:.3f}, skip)")
                res.update(alpha=None, beta=None, ratio=None)
    else:
        print(f"  alpha/beta: fit failed (R^2={r2_ab})")

    return res


def _interactive_band(prompt, default):
    try:
        inp = input(prompt).strip()
        return int(inp) if inp else default
    except (EOFError, KeyboardInterrupt):
        return default


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)

    procar_file = sys.argv[1] if os.path.exists(sys.argv[1]) else None
    if not procar_file:
        print(f"File not found: {sys.argv[1]}")
        sys.exit(1)

    # --- 解析参数 ---
    vbm_band = None
    cbm_band = None
    soc_vbm_up = soc_vbm_lo = None
    soc_cbm_up = soc_cbm_lo = None
    tau_p = 0.1
    k_range = 0.05
    T = 300
    output_dir = '.'
    list_bands_only = False

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--vbm' and i + 1 < len(sys.argv):
            vbm_band = int(sys.argv[i + 1]); i += 2
        elif arg == '--cbm' and i + 1 < len(sys.argv):
            cbm_band = int(sys.argv[i + 1]); i += 2
        elif arg == '--soc-vbm' and i + 2 < len(sys.argv):
            soc_vbm_up = int(sys.argv[i + 1])
            soc_vbm_lo = int(sys.argv[i + 2]); i += 3
        elif arg == '--soc-cbm' and i + 2 < len(sys.argv):
            soc_cbm_up = int(sys.argv[i + 1])
            soc_cbm_lo = int(sys.argv[i + 2]); i += 3
        elif arg == '--tau-p' and i + 1 < len(sys.argv):
            tau_p = float(sys.argv[i + 1]); i += 2
        elif arg == '--k-range' and i + 1 < len(sys.argv):
            k_range = float(sys.argv[i + 1]); i += 2
        elif arg == '--T' and i + 1 < len(sys.argv):
            T = float(sys.argv[i + 1]); i += 2
        elif arg == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]; i += 2
        elif arg == '--list-bands':
            list_bands_only = True; i += 1
        else:
            i += 1

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  spinlife -- VASP Spin Lifetime Calculator")
    print("=" * 65)
    print(f"  PROCAR : {procar_file}")
    print(f"  tau_p  : {tau_p} ps")
    print(f"  k_range: {k_range} A^-1")
    print(f"  T      : {T} K")

    procar = PROCAR(procar_file)
    procar.summary()

    # ========== 输出能带表 ==========
    list_bands(procar, output_dir)
    if list_bands_only:
        sys.exit(0)

    kpts = procar.get_kpoints_cart()
    _, _, nkx, nky = build_k_grid(procar.kpoints)

    # 检测 VBM/CBM
    vbm, cbm = procar.find_vbm_cbm()
    if vbm_band is not None:
        vbm = vbm_band; cbm = vbm + 1
    if cbm_band is not None:
        cbm = cbm_band; vbm = cbm - 1
    print(f"\n  VBM: band {vbm}, CBM: band {cbm}")

    # ========== k 切片 ==========
    idx_slice, k_scan, order = get_k_slice(kpts, procar)
    if len(idx_slice) < 3:
        idx_slice = list(range(procar.nk))
        k_scan = np.linspace(-0.5, 0.5, procar.nk)
        order = np.argsort(k_scan)
        k_scan = k_scan[order]

    # ========== SOC 能带选择 (交互式 / 命令行) ==========
    if soc_vbm_up is None:
        print("\n--- VBM SOC Bands ---")
        soc_vbm_up = _interactive_band(
            f"  m* band [{vbm}]: ", vbm)
        soc_vbm_lo = _interactive_band(
            f"  SOC lower (partner) band [{max(vbm - 1, 1)}]: ",
            max(vbm - 1, 1))
    if soc_cbm_up is None:
        print("\n--- CBM SOC Bands ---")
        soc_cbm_up = _interactive_band(
            f"  m* band [{cbm}]: ", cbm)
        soc_cbm_lo = _interactive_band(
            f"  SOC upper (partner) band [{min(cbm + 1, procar.nbands)}]: ",
            min(cbm + 1, procar.nbands))

    # ========== VBM 拟合 ==========
    res_vbm = run_soc_fit(
        'VBM', procar, kpts, idx_slice, k_scan, order,
        mstar_band=soc_vbm_up, soc_upper=soc_vbm_up, soc_lower=soc_vbm_lo,
        tau_p=tau_p, T=T, k_range=k_range, extrema_type='max')

    # ========== CBM 拟合 ==========
    res_cbm = run_soc_fit(
        'CBM', procar, kpts, idx_slice, k_scan, order,
        mstar_band=soc_cbm_up, soc_upper=soc_cbm_up, soc_lower=soc_cbm_lo,
        tau_p=tau_p, T=T, k_range=k_range, extrema_type='min')

    results = [res_vbm, res_cbm]

    # ========== 绘图 ==========
    if _HAS_MPL:
        valid = [r for r in results if r.get('k_scan') is not None]
        ncol = len(valid)
        if ncol > 0:
            fig, axes = plt.subplots(2, ncol, figsize=(6 * ncol, 10))
            if ncol == 1:
                axes = axes.reshape(2, 1)
            fig.suptitle('spinlife -- SOC Fitting Results', fontsize=13)

            for col, res in enumerate(valid):
                ks = res['k_scan']
                up, lo = res['bands']
                k0 = res['k0']
                akr = res['auto_k_range']
                E_us = res['E_up_scan']
                E_ls = res['E_lo_scan']

                ax = axes[0, col]
                ax.plot(ks, E_us * 1000, 'o-', ms=4, label=f'Band {up}')
                ax.plot(ks, E_ls * 1000, 's-', ms=4, label=f'Band {lo}')
                ax.axvspan(k0 - akr, k0 + akr, alpha=0.08, color='blue')
                ax.axvline(k0, color='gray', ls='--', alpha=0.5)
                ax.set_xlabel('k (A^-1)')
                ax.set_ylabel('E (meV)')
                ax.legend(fontsize=9)
                ax.set_title(f'{res["label"]}: SOC-split Bands')

                ax = axes[1, col]
                dE = np.abs(E_us - E_ls) * 1000
                x = (ks - k0)**2
                ax.plot(x, dE**2, 'o', ms=5)
                if res.get('ab_norm'):
                    coeffs = np.polyfit(x, dE**2, 1)
                    xs = np.linspace(0, max(x) * 1.05, 100)
                    label = rf'$(\alpha^2+\beta^2)k^2$ fit'
                    ax.plot(xs, coeffs[0] * xs + coeffs[1], '-', label=label)
                ax.set_xlabel('(k-k0)^2 (A^-2)')
                ax.set_ylabel(r'$\Delta E^2$ (meV$^2$)')
                ax.legend(fontsize=9)
                ax.set_title(f'{res["label"]}: DE^2 Fit')

            plt.tight_layout()
            out_png = os.path.join(output_dir, 'spinlife_results.png')
            plt.savefig(out_png, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"\n  [Plot: {out_png}]")

    # ========== 报告 ==========
    report_path = os.path.join(output_dir, 'spinlife_report.txt')
    with open(report_path, 'w') as f:
        f.write("spinlife -- VASP Spin Lifetime Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"PROCAR: {procar_file}\n")
        f.write(f"tau_p : {tau_p} ps\n")
        f.write(f"T     : {T} K\n")
        f.write(f"VBM   : band {vbm}\n")
        f.write(f"CBM   : band {cbm}\n")
        f.write(f"k grid: {nkx}x{nky}\n\n")

        for res in results:
            up, lo = res['bands']
            f.write(f"--- {res['label']}: band {up}/{lo} ---\n")
            if res.get('m_star'):
                f.write(f"m* = {res['m_star']:.2f} m0\n")
            if res.get('ab_norm'):
                f.write(f"sqrt(a^2+b^2) = {res['ab_norm']*1000:.2f} meV.A\n")
                f.write(f"Delta = {res['Delta']*1000:.2f} meV\n")
            if res.get('alpha'):
                f.write(f"alpha = {res['alpha']*1000:.2f} meV.A\n")
                f.write(f"beta  = {res['beta']*1000:.2f} meV.A\n")
                f.write(f"alpha/beta = {res['ratio']:.3f}\n")
            if res.get('spin'):
                s = res['spin']
                f.write(f"tau_s  = {s['tau_s_ps']:.2f} ps\n")
                f.write(f"L_PSH  = {s['L_PSH_um']:.2f} um\n")
            f.write("\n")

    print(f"\n  [Report: {report_path}]")
    print("=" * 65)
