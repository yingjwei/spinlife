# -*- coding: utf-8 -*-
"""
spinlife — VASP PROCAR 自旋寿命 + 载流子迁移率计算

子命令:
  python -m spinlife.main PROCAR [options]        自旋寿命 (PROCAR -> alpha/beta/tau_s)
  python -m spinlife.main mobility [options]       载流子迁移率 (形变势理论)
  python -m spinlife.main PROCAR --dump-band N    导出能带原始数据 (用于校验)

自旋寿命用法:
  python -m spinlife.main PROCAR [--vbm N] [--cbm N] [--tau-p 0.1]
         [--k-range 0.05] [--T 300] [--output-dir .]
         [--soc-vbm UPPER LOWER] [--soc-cbm UPPER LOWER]
         [--dump-band N]

迁移率用法:
  python -m spinlife.main mobility
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
from .mobility.calc_mobility import (read_POSCAR_A0, fit_C2D, fit_E1,
                                     calc_mu, C2D_Jm2_from_d2E)


def build_k_grid(kpoints):
    kx_vals = sorted(set(kp[0] for kp in kpoints))
    ky_vals = sorted(set(kp[1] for kp in kpoints))
    nkx, nky = len(kx_vals), len(ky_vals)
    return kx_vals, ky_vals, nkx, nky


def dump_band_data(procar, band_idx, output_dir='.'):
    """输出指定能带沿 k 切片的原始数据 (k, E, sx, sy, sz)"""
    kpts = procar.get_kpoints_cart()
    idx_slice, k_scan, order = get_k_slice(kpts, procar)

    E = procar.get_band_energy(band_idx)
    sx, sy, sz = procar.get_spin(band_idx)

    k_out = k_scan
    E_out = E[idx_slice][order]
    sx_out = sx[idx_slice][order]
    sy_out = sy[idx_slice][order]
    sz_out = sz[idx_slice][order]

    path = os.path.join(output_dir, f'band_{band_idx}_data.txt')
    with open(path, 'w') as f:
        f.write("# k(A^-1)  E(eV)  <sx>  <sy>  <sz>\n")
        for i in range(len(k_out)):
            f.write(f"{k_out[i]:.8f}  {E_out[i]:.8f}  "
                    f"{sx_out[i]:.8f}  {sy_out[i]:.8f}  {sz_out[i]:.8f}\n")
    print(f"\n  [Band {band_idx} data -> {path}]")
    print(f"  Columns: k(A^-1), E(eV), <sx>, <sy>, <sz>")
    print(f"  ({len(k_out)} k-points along slice)")
    return True


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

    if sys.argv[1] == 'mobility':
        main_mobility()
        return

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
    dump_band = None

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
        elif arg == '--dump-band' and i + 1 < len(sys.argv):
            dump_band = int(sys.argv[i + 1]); i += 2
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

    # ========== 输出单条能带数据 (可选) ==========
    if dump_band is not None:
        if 1 <= dump_band <= procar.nbands:
            dump_band_data(procar, dump_band, output_dir)
        else:
            print(f"  Invalid band: {dump_band} (1-{procar.nbands})")
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


def _input_data(prompt):
    """读取多行数据, 空行结束"""
    print(prompt)
    lines = []
    while True:
        try:
            line = input().strip()
            if not line:
                break
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            break
    return lines


def main_mobility():
    """交互式载流子迁移率计算"""
    print("=" * 65)
    print("  spinlife.mobility -- 载流子迁移率计算 (形变势理论)")
    print("=" * 65)

    # --- 方向 ---
    direction = ''
    try:
        direction = input("\n  方向 (x/y, Enter跳过): ").strip()
    except (EOFError, KeyboardInterrupt):
        pass

    # --- C2D: 应变-能量 ---
    print("\n--- C2D (弹性模量) ---")
    print("  输入 应变(%)  总能量(eV), 一行一个, 空行结束")
    print("  例: -3 -277.65870")
    raw = _input_data("  >>>")
    strain = []
    energy = []
    for line in raw:
        parts = line.split()
        if len(parts) >= 2:
            strain.append(float(parts[0]) / 100)
            energy.append(float(parts[1]))
    if len(strain) < 3:
        print(f"  [至少需要3个数据点, 当前{len(strain)}个]")
        return

    d2E, r2_e, coeffs = fit_C2D(np.array(strain), np.array(energy))
    A2, A1, A0_fit = coeffs
    print(f"\n  Fit: E = {A2:.4f}*eps^2 + {A1:.4f}*eps + {A0_fit:.6f}")
    print(f"  d2E/deps2 = {d2E:.4f} eV,  R^2 = {r2_e:.6f}")

    # --- A0 ---
    A0 = None
    try:
        poscar = input("  POSCAR 路径 (留空则手动输入 A0): ").strip()
        if poscar:
            A0 = read_POSCAR_A0(poscar)
        if A0 is None:
            A0 = float(input("  A0 (A^2): "))
    except (EOFError, KeyboardInterrupt):
        pass

    C2D_Jm2 = None
    if A0:
        C2D_Jm2 = C2D_Jm2_from_d2E(d2E, A0)
        print(f"  C2D = {C2D_Jm2:.2f} J/m^2")

    # --- E1 ---
    E1_vbm = None
    vbm_r2 = None
    try:
        if input("\n  输入 VBM 形变势数据？(y/n, 默认n): ").strip().lower() == 'y':
            raw = _input_data("  输入 应变(%)  E_VBM(eV), 空行结束\n  >>>")
            s_vbm, e_vbm = [], []
            for line in raw:
                parts = line.split()
                if len(parts) >= 2:
                    s_vbm.append(float(parts[0]) / 100)
                    e_vbm.append(float(parts[1]))
            if len(s_vbm) >= 3:
                E1_vbm, vbm_r2, vc = fit_E1(np.array(s_vbm), np.array(e_vbm))
                print(f"  E1 (VBM) = {E1_vbm:.4f} eV,  R^2 = {vbm_r2:.6f}")
    except (EOFError, KeyboardInterrupt):
        pass

    E1_cbm = None
    cbm_r2 = None
    try:
        if input("\n  输入 CBM 形变势数据？(y/n, 默认n): ").strip().lower() == 'y':
            raw = _input_data("  输入 应变(%)  E_CBM(eV), 空行结束\n  >>>")
            s_cbm, e_cbm = [], []
            for line in raw:
                parts = line.split()
                if len(parts) >= 2:
                    s_cbm.append(float(parts[0]) / 100)
                    e_cbm.append(float(parts[1]))
            if len(s_cbm) >= 3:
                E1_cbm, cbm_r2, cc = fit_E1(np.array(s_cbm), np.array(e_cbm))
                print(f"  E1 (CBM) = {E1_cbm:.4f} eV,  R^2 = {cbm_r2:.6f}")
    except (EOFError, KeyboardInterrupt):
        pass

    # --- 有效质量 + 温度 ---
    m_vbm = m_cbm = None
    T = 300
    try:
        if E1_vbm is not None:
            m_vbm = float(input("\n  m* (VBM, m0): "))
        if E1_cbm is not None:
            m_cbm = float(input("  m* (CBM, m0): "))
        T = float(input(f"  T (K) [{T}]: ") or T)
    except (EOFError, KeyboardInterrupt):
        pass

    # --- 计算 ---
    print(f"\n--- Results ({direction if direction else ''}) ---")
    results = []
    if C2D_Jm2 and E1_vbm and m_vbm:
        mu_h = calc_mu(C2D_Jm2, E1_vbm, m_vbm, T)
        print(f"  空穴 (VBM):  mu = {mu_h:.2f} cm^2/V.s")
        results.append(('hole(VBM)', mu_h, m_vbm, E1_vbm))
    if C2D_Jm2 and E1_cbm and m_cbm:
        mu_e = calc_mu(C2D_Jm2, E1_cbm, m_cbm, T)
        print(f"  电子 (CBM):  mu = {mu_e:.2f} cm^2/V.s")
        results.append(('electron(CBM)', mu_e, m_cbm, E1_cbm))

    # --- 绘图 ---
    if _HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # C2D
        ax = axes[0]
        xs = np.linspace(min(strain), max(strain), 200)
        ys = np.polyval(coeffs, xs)
        ax.plot(np.array(strain)*100, energy, 'o', ms=8, label='data')
        ax.plot(xs*100, ys, '-',
                label=rf'fit: {A2:.2f}$\varepsilon^2$ + {A1:.2f}$\varepsilon$')
        ax.axhline(energy[len(energy)//2], color='gray', ls='--', alpha=0.4)
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Total Energy (eV)')
        ax.legend(fontsize=9)
        ax.set_title(f'C2D Fit ({direction})')
        ax.grid(alpha=0.3)
        # E1
        ax = axes[1]
        if E1_vbm and len(s_vbm) > 0:
            xs = np.linspace(min(s_vbm), max(s_vbm), 200)
            ys = np.polyval(vc, xs)
            ax.plot(np.array(s_vbm)*100, e_vbm, 's', ms=8, label=f'VBM (E1={E1_vbm:.3f}eV)')
            ax.plot(xs*100, ys, '-')
        if E1_cbm and len(s_cbm) > 0:
            xs = np.linspace(min(s_cbm), max(s_cbm), 200)
            ys = np.polyval(cc, xs)
            ax.plot(np.array(s_cbm)*100, e_cbm, 'o', ms=8, label=f'CBM (E1={E1_cbm:.3f}eV)')
            ax.plot(xs*100, ys, '--')
        ax.axhline(0, color='gray', ls='--', alpha=0.4)
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Band Energy (eV)')
        ax.legend(fontsize=9)
        ax.set_title(f'E1 Fit ({direction})')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('mobility_fit.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  [Plot: mobility_fit.png]")

    # --- 报告 ---
    with open('mobility_report.txt', 'w') as f:
        f.write("spinlife.mobility -- Carrier Mobility Report\n")
        f.write("=" * 45 + "\n")
        f.write(f"Direction: {direction if direction else '-'}\n")
        f.write(f"T: {T} K\n")
        f.write(f"C2D: {C2D_Jm2:.4f} J/m^2\n" if C2D_Jm2 else "")
        f.write(f"d2E/deps2: {d2E:.4f} eV, R^2: {r2_e:.6f}\n")
        if A0:
            f.write(f"A0: {A0:.2f} A^2\n")
        if E1_vbm:
            f.write(f"E1_VBM: {E1_vbm:.4f} eV, R^2: {vbm_r2:.6f}\n")
        if E1_cbm:
            f.write(f"E1_CBM: {E1_cbm:.4f} eV, R^2: {cbm_r2:.6f}\n")
        if results:
            f.write("\n--- Results ---\n")
            for name, mu, ms, e1 in results:
                f.write(f"{name}: mu = {mu:.2f} cm^2/V.s\n")
    print(f"\n  [Report: mobility_report.txt]")
    print("=" * 65)


if __name__ == '__main__':
    main()
