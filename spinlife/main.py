# -*- coding: utf-8 -*-
"""
spinlife — VASP 自旋寿命 + 载流子迁移率计算

安装后:
  spinlife                   交互式菜单 (默认)
  spinlife PROCAR [options]  PROCAR 传统模式
  spinlife mobility          迁移率模式

子命令:
  python -m spinlife.main                   交互式菜单
  python -m spinlife.main PROCAR [options]  自旋寿命 (PROCAR -> alpha/beta/tau_s)
  python -m spinlife.main mobility          载流子迁移率 (形变势理论)

自旋寿命用法:
  spinlife PROCAR [--vbm N] [--cbm N] [--tau-p 0.1]
         [--k-range 0.05] [--T 300] [--output-dir spinlife]
         [--soc-vbm UPPER LOWER] [--soc-cbm UPPER LOWER]
         [--dump-band N]

迁移率用法:
  spinlife mobility

输出文件统一保存在 spinlife/ 目录.
"""

import sys
import os
from pathlib import Path
import numpy as np

# 允许直接执行: python path/to/spinlife/main.py PROCAR
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from spinlife.read_procar import PROCAR
from spinlife.fit_soc import fit_effmass, fit_alpha_beta, calc_spin_lifetime
from spinlife.mobility.calc_mobility import (read_POSCAR_A0, fit_C2D, fit_E1,
                                             calc_mu, C2D_Jm2_from_d2E,
                                             m0, e_ch)
from spinlife.wannier import read_bands, k_to_reciprocal, find_extremum, band_slice


def build_k_grid(kpoints):
    kx_vals = sorted(set(kp[0] for kp in kpoints))
    ky_vals = sorted(set(kp[1] for kp in kpoints))
    nkx, nky = len(kx_vals), len(ky_vals)
    return kx_vals, ky_vals, nkx, nky


def dump_band_data(procar, band_idx, output_dir=OUTPUT_DIR):
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
    sy_lo_scan = sy_lo[idx_slice][order]

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


# 会话共享状态 — 各模块计算结果自动传递
_ctx = {
    'm_star': None,          # 有效质量 (m₀)
    'alpha_meva': None,      # α (meV·Å)
    'beta_meva': None,       # β (meV·Å)
    'tau_p': None,           # 动量散射时间 (ps)
    'mu': None,              # 迁移率 (cm²/V·s)
    'T': 300,                # 温度 (K)
}

# 所有输出文件统一放入 spinlife/ 目录
OUTPUT_DIR = 'spinlife'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _ctx_summary():
    """当前工作区状态摘要"""
    parts = []
    if _ctx['m_star']:
        parts.append(f"m* = {_ctx['m_star']:.4f} m0")
    if _ctx['alpha_meva']:
        parts.append(f"a = {_ctx['alpha_meva']:.2f} meV.A")
    if _ctx['beta_meva']:
        parts.append(f"b = {_ctx['beta_meva']:.2f} meV.A")
    if _ctx['tau_p']:
        parts.append(f"tau_p = {_ctx['tau_p']:.4f} ps")
    if _ctx['mu']:
        parts.append(f"mu = {_ctx['mu']:.2f} cm2/V.s")
    return "  |  ".join(parts) if parts else None


def _show_band_table(procar, center, label, n_range=5):
    """vaspkit-style: 显示中心能带附近列表"""
    print(f"\n  {label} 附近能带:")
    print(f"  {'Band':>6}  {'Energy(eV)':>12}")
    print(f"  {'-'*22}")
    for b in range(max(1, center - n_range),
                   min(procar.nbands + 1, center + n_range + 1)):
        e = procar.get_band_energy(b)[procar.nk // 2]
        tag = f"<-- {label}" if b == center else ""
        print(f"  {b:>6}  {e:>12.4f}  {tag}")


def _show_main_menu():
    """显示主菜单 (含当前结果状态)"""
    print()
    print("=" * 65)
    print("  spinlife — VASP 自旋寿命 + 载流子迁移率计算")
    print("=" * 65)
    print()
    print("  1)  载流子迁移率 (Mobility)")
    print("  2)  有效质量 (Wannier 能带 → m*)")
    print("  3)  SOC 参数 α/β (Wannier + PROCAR)")
    print("  4)  自旋寿命 (τ_s, L_PSH)")
    print("  5)  导出能带数据")
    print()
    summary = _ctx_summary()
    if summary:
        print(f"  [{summary}]")
    print()
    print("  0)  退出")
    print()


def _interactive_menu():
    """全功能交互式菜单循环"""
    while True:
        _show_main_menu()
        try:
            c = input("  -->> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if c == '0':
            break
        elif c == '1':
            main_mobility()
        elif c == '2':
            main_effmass_wannier()
        elif c == '3':
            main_alpha_beta()
        elif c == '4':
            main_spin_lifetime_menu()
        elif c == '5':
            main_dump_band_menu()
        else:
            print("  [无效选项]")
            continue
        if c in ('1', '2', '3', '4', '5'):
            input("\n  -->> 按 Enter 返回菜单...")


def main():
    """入口: 菜单 / CLI 快捷模式"""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ('-h', '--help'):
            print(__doc__)
            return
        if arg == 'mobility':
            main_mobility()
            return
        # PROCAR 传统模式 (向后兼容)
        procar_file = sys.argv[1] if os.path.exists(sys.argv[1]) else None
        if procar_file:
            _procar_cli_flow()
            return
    # 交互式菜单 (默认)
    _interactive_menu()


def _procar_cli_flow():
    """PROCAR CLI 传统分析 (向后兼容, 读 VASP PROCAR → m* + α/β + τ_s)"""
    procar_file = sys.argv[1]

    vbm_band = None
    cbm_band = None
    soc_vbm_up = soc_vbm_lo = None
    soc_cbm_up = soc_cbm_lo = None
    tau_p = 0.1
    k_range = 0.05
    T = 300
    output_dir = OUTPUT_DIR
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

    if dump_band is not None:
        if 1 <= dump_band <= procar.nbands:
            dump_band_data(procar, dump_band, output_dir)
        else:
            print(f"  Invalid band: {dump_band} (1-{procar.nbands})")
        sys.exit(0)

    kpts = procar.get_kpoints_cart()
    _, _, nkx, nky = build_k_grid(procar.kpoints)

    vbm, cbm = procar.find_vbm_cbm()
    if vbm_band is not None:
        vbm = vbm_band; cbm = vbm + 1
    if cbm_band is not None:
        cbm = cbm_band; vbm = cbm - 1
    print(f"\n  VBM: band {vbm}, CBM: band {cbm}")

    idx_slice, k_scan, order = get_k_slice(kpts, procar)
    if len(idx_slice) < 3:
        idx_slice = list(range(procar.nk))
        k_scan = np.linspace(-0.5, 0.5, procar.nk)
        order = np.argsort(k_scan)
        k_scan = k_scan[order]

    if soc_vbm_up is None:
        print()
        print("=" * 65)
        print("  SOC 带对选择")
        print("=" * 65)
        _show_band_table(procar, vbm, "VBM")
        print()
        soc_vbm_up = int(input("  -->> VBM SOC upper band: ") or vbm)
        soc_vbm_lo = int(input("  -->> VBM SOC lower band: ") or max(vbm - 1, 1))
    if soc_cbm_up is None:
        _show_band_table(procar, cbm, "CBM")
        print()
        soc_cbm_up = int(input("  -->> CBM SOC upper band: ") or cbm)
        soc_cbm_lo = int(input("  -->> CBM SOC lower band: ") or min(cbm + 1, procar.nbands))

    res_vbm = run_soc_fit(
        'VBM', procar, kpts, idx_slice, k_scan, order,
        mstar_band=soc_vbm_up, soc_upper=soc_vbm_up, soc_lower=soc_vbm_lo,
        tau_p=tau_p, T=T, k_range=k_range, extrema_type='max')
    res_cbm = run_soc_fit(
        'CBM', procar, kpts, idx_slice, k_scan, order,
        mstar_band=soc_cbm_up, soc_upper=soc_cbm_up, soc_lower=soc_cbm_lo,
        tau_p=tau_p, T=T, k_range=k_range, extrema_type='min')
    results = [res_vbm, res_cbm]

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
                E_us = res['E_up_scan']; E_ls = res['E_lo_scan']
                axes[0, col].plot(ks, E_us * 1000, 'o-', ms=4, label=f'Band {up}')
                axes[0, col].plot(ks, E_ls * 1000, 's-', ms=4, label=f'Band {lo}')
                axes[0, col].axvspan(k0 - akr, k0 + akr, alpha=0.08, color='blue')
                axes[0, col].axvline(k0, color='gray', ls='--', alpha=0.5)
                axes[0, col].set_xlabel('k (A^-1)'); axes[0, col].set_ylabel('E (meV)')
                axes[0, col].legend(fontsize=9)
                axes[0, col].set_title(f'{res["label"]}: SOC-split Bands')
                dE = np.abs(E_us - E_ls) * 1000
                x = (ks - k0)**2
                axes[1, col].plot(x, dE**2, 'o', ms=5)
                if res.get('ab_norm'):
                    c2 = np.polyfit(x, dE**2, 1)
                    xs = np.linspace(0, max(x) * 1.05, 100)
                    axes[1, col].plot(xs, c2[0] * xs + c2[1], '-', label=r'$(\alpha^2+\beta^2)k^2$ fit')
                axes[1, col].set_xlabel('(k-k0)^2 (A^-2)'); axes[1, col].set_ylabel(r'$\Delta E^2$ (meV$^2$)')
                axes[1, col].legend(fontsize=9)
                axes[1, col].set_title(f'{res["label"]}: DE^2 Fit')
            plt.tight_layout()
            out_png = os.path.join(output_dir, 'spinlife_results.png')
            plt.savefig(out_png, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"\n  [Plot: {out_png}]")

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


def _input_strain_data():
    """通用: 输入应变-能量数据, 返回 (strain_array, energy_array) 或 (None, None)"""
    print("  输入 应变(%)  能量(eV), 一行一个, 空行结束")
    raw = _input_data("")
    strain, energy = [], []
    for line in raw:
        parts = line.split()
        if len(parts) >= 2:
            strain.append(float(parts[0]) / 100)
            energy.append(float(parts[1]))
    if len(strain) < 3:
        return None, None
    return np.array(strain), np.array(energy)


def _input_E1_data(label):
    """输入形变势数据: 应变(%)  E(eV)"""
    print(f"  输入 {label}: 应变(%)  E(eV), 空行结束")
    raw = _input_data("")
    s, e = [], []
    for line in raw:
        parts = line.split()
        if len(parts) >= 2:
            s.append(float(parts[0]) / 100)
            e.append(float(parts[1]))
    return (np.array(s), np.array(e)) if len(s) >= 3 else (None, None)


def main_mobility():
    """交互式载流子迁移率计算 (x + y 双方向)"""
    print("=" * 65)
    print("  spinlife — 载流子迁移率计算")
    print("  (Deformation Potential Theory)")
    print("=" * 65)
    print()
    print("  μ = 2eℏ³C₂D / (3k_B T |m*|² E₁²)")

    T = 300
    try:
        T = float(input(f"\n  -->> 温度 T (K) [{T}]: ") or T)
    except (EOFError, KeyboardInterrupt):
        pass

    all_data = {}

    for dir_label in ('x', 'y'):
        # 如果 y 方向可选, 询问是否跳过
        if dir_label == 'y' and all_data:
            try:
                if input("\n  -->> 输入 Y 方向数据？(y/n, 默认n): ").strip().lower() != 'y':
                    break
            except (EOFError, KeyboardInterrupt):
                break

        print()
        print("=" * 65)
        print(f"  [{dir_label.upper()}] 方向")
        print("=" * 65)

        # --- C2D ---
        print()
        print("  C2D — 应变 vs 总能量:")
        strain, energy = _input_strain_data()
        if strain is None:
            print("  [数据不足, 跳过]")
            continue

        d2E, r2_e, coeffs = fit_C2D(strain, energy)
        A2, A1, A0_fit = coeffs
        print(f"\n  Fit: E = {A2:.4f}*eps^2 + {A1:.4f}*eps + {A0_fit:.6f}")
        print(f"  d2E/deps2 = {d2E:.4f} eV,  R^2 = {r2_e:.6f}")

        A0 = None
        try:
            poscar = input("  -->> POSCAR 路径 (留空手动输入 A0): ").strip()
            if poscar:
                A0 = read_POSCAR_A0(poscar)
            if A0 is None:
                A0 = float(input("  -->> A0 (A^2): "))
        except (EOFError, KeyboardInterrupt):
            pass

        C2D = C2D_Jm2_from_d2E(d2E, A0) if A0 else None
        if C2D:
            print(f"  C2D = {C2D:.2f} J/m^2")

        # --- E1 ---
        E1_vbm, E1_cbm = None, None
        s_vbm_data, e_vbm_data = None, None
        s_cbm_data, e_cbm_data = None, None
        try:
            if input("\n  -->> 输入 VBM 形变势数据？(y/n, 默认n): ").strip().lower() == 'y':
                s_vbm_data, e_vbm_data = _input_E1_data(f"VBM ({dir_label})")
                if s_vbm_data is not None:
                    E1_vbm, vbm_r2, vc = fit_E1(s_vbm_data, e_vbm_data)
                    print(f"  E1 (VBM) = {E1_vbm:.4f} eV,  R^2 = {vbm_r2:.6f}")

            if input("\n  -->> 输入 CBM 形变势数据？(y/n, 默认n): ").strip().lower() == 'y':
                s_cbm_data, e_cbm_data = _input_E1_data(f"CBM ({dir_label})")
                if s_cbm_data is not None:
                    E1_cbm, cbm_r2, cc = fit_E1(s_cbm_data, e_cbm_data)
                    print(f"  E1 (CBM) = {E1_cbm:.4f} eV,  R^2 = {cbm_r2:.6f}")
        except (EOFError, KeyboardInterrupt):
            pass

        # --- 有效质量 (各向异性) ---
        m_vbm, m_cbm = None, None
        mstar_ctx = _ctx.get('m_star')
        mstar_hint = f" [Wannier: {mstar_ctx:.4f}]" if mstar_ctx else ""
        try:
            if E1_vbm is not None:
                m_vbm = float(input(f"\n  -->> m* (VBM, {dir_label}方向, m0){mstar_hint}: "))
            if E1_cbm is not None:
                m_cbm = float(input(f"  -->> m* (CBM, {dir_label}方向, m0){mstar_hint}: "))
        except (EOFError, KeyboardInterrupt):
            pass

        all_data[dir_label] = {
            'strain': strain, 'energy': energy, 'coeffs': coeffs,
            'd2E': d2E, 'r2_e': r2_e, 'A0': A0, 'C2D': C2D,
            'E1_vbm': E1_vbm, 'E1_cbm': E1_cbm,
            's_vbm': s_vbm_data, 'e_vbm': e_vbm_data,
            's_cbm': s_cbm_data, 'e_cbm': e_cbm_data,
            'm_vbm': m_vbm, 'm_cbm': m_cbm,
            'vc': vc if E1_vbm else None,
            'cc': cc if E1_cbm else None,
        }

    if not all_data:
        print("  [无有效数据]")
        return

    # --- 计算迁移率 ---
    print()
    print("=" * 65)
    print("  Results")
    print("=" * 65)
    for d in ('x', 'y'):
        if d not in all_data:
            continue
        dat = all_data[d]
        print(f"\n  [{d.upper()}]  C2D = {dat['C2D']:.2f} J/m^2" if dat['C2D'] else f"\n  [{d.upper()}]")
        if dat['E1_vbm'] and dat['m_vbm'] and dat['C2D']:
            mu_h, tau_h = calc_mu(dat['C2D'], dat['E1_vbm'], dat['m_vbm'], T)
            dat['mu_h'], dat['tau_h'] = mu_h, tau_h
            _ctx['mu'] = mu_h
            _ctx['tau_p'] = tau_h
            _ctx['T'] = T
            print(f"  空穴 (VBM): μ = {mu_h:.2f} cm^2/V.s,  τ_p = {tau_h:.4f} ps")
        if dat['E1_cbm'] and dat['m_cbm'] and dat['C2D']:
            mu_e, tau_e = calc_mu(dat['C2D'], dat['E1_cbm'], dat['m_cbm'], T)
            dat['mu_e'], dat['tau_e'] = mu_e, tau_e
            _ctx['mu'] = mu_e
            _ctx['tau_p'] = tau_e
            _ctx['T'] = T
            print(f"  电子 (CBM): μ = {mu_e:.2f} cm^2/V.s,  τ_p = {tau_e:.4f} ps")

    # --- 对比表 ---
    if 'x' in all_data and 'y' in all_data:
        print()
        print("=" * 65)
        print("  Comparison: X vs Y")
        print("=" * 65)
        rows = [
            ('C2D (J/m^2)',        'C2D',  '{:.2f}'),
            ('E1_VBM (eV)',        'E1_vbm', '{:.4f}'),
            ('E1_CBM (eV)',        'E1_cbm', '{:.4f}'),
            ('mu_h (cm^2/V.s)',    'mu_h',   '{:.2f}'),
            ('mu_e (cm^2/V.s)',    'mu_e',   '{:.2f}'),
            ('tau_p_h (ps)',       'tau_h',  '{:.4f}'),
            ('tau_p_e (ps)',       'tau_e',  '{:.4f}'),
            ('m*_VBM (m0)',        'm_vbm',  '{:.4f}'),
            ('m*_CBM (m0)',        'm_cbm',  '{:.4f}'),
        ]
        print(f"  {'':>20}  {'X':>14}  {'Y':>14}")
        print(f"  {'-'*52}")
        for label, key, fmt in rows:
            xv = all_data['x'].get(key)
            yv = all_data['y'].get(key)
            xs = fmt.format(xv) if xv is not None else '-'
            ys = fmt.format(yv) if yv is not None else '-'
            print(f"  {label:>20}  {xs:>14}  {ys:>14}")

    # --- 绘图 ---
    if _HAS_MPL:
        ndir = len(all_data)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = {'x': '#E24A33', 'y': '#348ABD'}
        markers = {'x': 'o', 'y': 's'}
        # C2D
        ax = axes[0]
        for d, dat in [('x', all_data.get('x')), ('y', all_data.get('y'))]:
            if dat is None:
                continue
            strain, energy = dat['strain'], dat['energy']
            coeffs = dat['coeffs']
            xs = np.linspace(min(strain), max(strain), 200)
            ys = np.polyval(coeffs, xs)
            ax.plot(np.array(strain)*100, energy, markers[d], ms=8, color=colors[d], label=f'{d.upper()} data')
            ax.plot(xs*100, ys, '-', color=colors[d], alpha=0.7, label=f'{d.upper()} fit')
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Total Energy (eV)')
        ax.legend(fontsize=9)
        ax.set_title('C2D Fit')
        ax.grid(alpha=0.3)
        # E1
        ax = axes[1]
        for d, dat in [('x', all_data.get('x')), ('y', all_data.get('y'))]:
            if dat is None:
                continue
            if dat.get('E1_vbm') and dat.get('s_vbm') is not None:
                xs = np.linspace(min(dat['s_vbm']), max(dat['s_vbm']), 200)
                ys = np.polyval(dat['vc'], xs)
                ax.plot(np.array(dat['s_vbm'])*100, dat['e_vbm'], markers[d], ms=8,
                        color=colors[d], label=f"VBM {d.upper()} (E1={dat['E1_vbm']:.3f})")
                ax.plot(xs*100, ys, '-', color=colors[d], alpha=0.5)
            if dat.get('E1_cbm') and dat.get('s_cbm') is not None:
                xs = np.linspace(min(dat['s_cbm']), max(dat['s_cbm']), 200)
                ys = np.polyval(dat['cc'], xs)
                ax.plot(np.array(dat['s_cbm'])*100, dat['e_cbm'], markers[d], ms=8,
                        color=colors[d], label=f"CBM {d.upper()} (E1={dat['E1_cbm']:.3f})",
                        fillstyle='none')
                ax.plot(xs*100, ys, '--', color=colors[d], alpha=0.5)
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Band Energy (eV)')
        ax.legend(fontsize=9)
        ax.set_title('E1 Fit')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'mobility_fit.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  [Plot: {OUTPUT_DIR}/mobility_fit.png]")

    # --- 报告 ---
    with open(os.path.join(OUTPUT_DIR, 'mobility_report.txt'), 'w') as f:
        f.write("spinlife.mobility -- Carrier Mobility Report\n")
        f.write("=" * 45 + "\n")
        f.write(f"T: {T} K\n\n")
        for d, dat in [('x', all_data.get('x')), ('y', all_data.get('y'))]:
            if dat is None:
                continue
            f.write(f"--- {d.upper()} ---\n")
            if dat.get('C2D'):
                f.write(f"C2D: {dat['C2D']:.4f} J/m^2\n")
                f.write(f"d2E/deps2: {dat['d2E']:.4f} eV, R^2: {dat['r2_e']:.6f}\n")
            if dat.get('A0'):
                f.write(f"A0: {dat['A0']:.2f} A^2\n")
            if dat.get('E1_vbm'):
                f.write(f"E1_VBM: {dat['E1_vbm']:.4f} eV\n")
            if dat.get('E1_cbm'):
                f.write(f"E1_CBM: {dat['E1_cbm']:.4f} eV\n")
            if dat.get('mu_h'):
                f.write(f"mu_h: {dat['mu_h']:.2f} cm^2/V.s, tau_p: {dat['tau_h']:.4f} ps\n")
            if dat.get('mu_e'):
                f.write(f"mu_e: {dat['mu_e']:.2f} cm^2/V.s, tau_p: {dat['tau_e']:.4f} ps\n")
            f.write("\n")
    print(f"\n  [Report: {OUTPUT_DIR}/mobility_report.txt]")
    print("=" * 65)


def main_effmass_wannier():
    """有效质量: 读 Wannier 能带 → 抛物线拟合 → m*"""
    print()
    print("=" * 65)
    print("  有效质量 (Effective mass from Wannier bands)")
    print("=" * 65)

    path = input("\n  -->> Wannier 能带文件路径 (wannier90_band.dat): ").strip()
    if not path or not os.path.exists(path):
        print("  [文件不存在]")
        return

    k_frac, energies, nk, nbands = read_bands(path)
    print(f"  能带范围: 1 - {nbands},  k 点数: {nk}")

    a = float(input("  -->> 晶格常数 a (Å): ").strip())
    k = k_to_reciprocal(k_frac, a)

    # 显示参考能带
    mid = nk // 2
    print(f"\n  路径中点能带 (用于参考):")
    print(f"  {'Band':>6}  {'Energy(eV)':>12}")
    print(f"  {'-'*22}")
    low = max(0, nbands // 2 - 6)
    high = min(nbands, nbands // 2 + 7)
    for b in range(low, high):
        print(f"  {b+1:>6}  {energies[mid, b]:>12.4f}")

    band = int(input(f"\n  -->> 能带序号 (1-{nbands}): ")) - 1
    mode = input("  -->> 极值类型 (VBM/CBM): ").strip().upper()
    mode = 'max' if mode == 'VBM' else 'min'

    energy = energies[:, band]
    k0, idx0 = find_extremum(k, energy, mode)
    print(f"  极值: k₀ = {k0:.4f} Å⁻¹,  E₀ = {energy[idx0]:.4f} eV")

    try:
        kr = float(input("  -->> 拟合范围 ±Δk (Å⁻¹) [0.05]: ") or 0.05)
    except (EOFError, KeyboardInterrupt):
        kr = 0.05

    k_slice, e_slice = band_slice(k, energy - energy[idx0], k0, kr)
    if len(k_slice) < 3:
        print(f"  [范围仅有 {len(k_slice)} 个 k 点, 无法拟合]")
        return

    m_star, r2, n_pts = fit_effmass(k_slice, e_slice, k0, kr)
    if m_star:
        print(f"\n  m* = {m_star:.4f} m₀  (R² = {r2:.6f},  {n_pts} pts)")
        _ctx['m_star'] = m_star
        with open(os.path.join(OUTPUT_DIR, 'effmass_report.txt'), 'w') as f:
            f.write("Effective Mass Report (Wannier)\n")
            f.write(f"File: {path}\n")
            f.write(f"Band: {band+1}\n")
            f.write(f"m* = {m_star:.4f} m₀\n")
            f.write(f"R² = {r2:.6f}\n")
        print(f"  [报告 -> {OUTPUT_DIR}/effmass_report.txt]")
    else:
        print("  [拟合失败]")


def main_alpha_beta():
    """SOC α/β: Wannier ΔE² 拟合 → √(α²+β²), PROCAR 自旋 → α/β 比值"""
    print()
    print("=" * 65)
    print("  SOC 参数 α/β 计算")
    print("=" * 65)
    print()
    print("  方法: Wannier 密能带拟合 → √(α²+β²)")
    print("        PROCAR 自旋织构斜率拟合 → α/β 比值")
    print()

    # ---- Part 1: √(α²+β²) from Wannier ----
    sqrt_ab = None
    path = input("  -->> Wannier 能带文件路径 (留空跳过): ").strip()
    if path and os.path.exists(path):
        print("\n  [1/2] √(α²+β²) — Wannier SOC 能带拟合")
        k_frac, energies, nk, nbands = read_bands(path)
        a = float(input("  -->> 晶格常数 a (Å): "))
        k = k_to_reciprocal(k_frac, a)

        up = int(input(f"  -->> SOC 带对上能带 (1-{nbands}): ")) - 1
        lo = int(input(f"  -->> SOC 带对下能带 (1-{nbands}): ")) - 1
        E_up = energies[:, up]
        E_lo = energies[:, lo]

        # k0 at minimum SOC splitting
        dE = np.abs(E_up - E_lo)
        k0 = k[np.argmin(dE)]
        try:
            kr = float(input("  -->> 拟合范围 ±Δk (Å⁻¹) [0.05]: ") or 0.05)
        except (EOFError, KeyboardInterrupt):
            kr = 0.05

        ab_norm, Delta, r2 = fit_alpha_beta(k, E_up, E_lo, k0, kr)
        if ab_norm:
            sqrt_ab = ab_norm
            print(f"\n  √(α²+β²) = {sqrt_ab*1000:.2f} meV·Å")
            print(f"  Δ       = {Delta*1000:.2f} meV  (R² = {r2:.4f})")
    elif path:
        print("  [文件不存在]")

    # ---- Part 2: α/β ratio from PROCAR (斜率拟合, 非点对点平均) ----
    ratio = None
    procar_path = input("\n  -->> PROCAR 路径 (留空跳过): ").strip()
    if procar_path and os.path.exists(procar_path):
        print("\n  [2/2] α/β 比值 — PROCAR 自旋织构斜率拟合")
        print("  方法: ⟨σ_x⟩ = A·k,  ⟨σ_y⟩ = B·k  →  α/β = A/B")
        print()
        procar = PROCAR(procar_path)
        kpts = procar.get_kpoints_cart()

        up = int(input(f"  -->> SOC 带对上能带 (1-{procar.nbands}): "))
        lo = int(input(f"  -->> SOC 带对下能带 (1-{procar.nbands}): "))

        sx_up, sy_up, _ = procar.get_spin(up)
        sx_lo, sy_lo, _ = procar.get_spin(lo)

        # 沿 ky≈0 切片 (Γ-X 方向)
        idx_slice, k_scan, order = get_k_slice(kpts, procar)
        k0 = k_scan[np.argmin(np.abs(k_scan))]

        try:
            kr = float(input("  -->> 拟合范围 ±Δk (Å⁻¹) [0.05]: ") or 0.05)
        except (EOFError, KeyboardInterrupt):
            kr = 0.05

        near = (np.abs(k_scan - k0) <= kr) & (np.abs(k_scan - k0) > 1e-10)
        kn = k_scan[near] - k0

        # 排序后的切片自旋数据
        sx_up_s = sx_up[idx_slice][order]
        sy_up_s = sy_up[idx_slice][order]
        sx_lo_s = sx_lo[idx_slice][order]
        sy_lo_s = sy_lo[idx_slice][order]

        # 显示数据表
        print(f"\n  {'k (Å⁻¹)':>10}  {'⟨σ_x⟩':>10}  {'⟨σ_y⟩':>10}  {'⟨σ_x⟩/⟨σ_y⟩':>12}")
        print(f"  {'-'*46}")
        for i in range(len(k_scan)):
            if near[i]:
                r_str = f"{sx_up_s[i]/sy_up_s[i]:.2f}" if abs(sy_up_s[i]) > 1e-10 else "-"
                print(f"  {k_scan[i]:>10.4f}  {sx_up_s[i]:>10.4f}  "
                      f"{sy_up_s[i]:>10.4f}  {r_str:>12}")

        # 对上下带分别做斜率拟合, 取 R² 更高者
        best_ratio = None
        best_r2 = -1
        for label, sx_s, sy_s in [
            (f"Band {up}", sx_up_s, sy_up_s),
            (f"Band {lo}", sx_lo_s, sy_lo_s),
        ]:
            if np.sum(near) < 3:
                continue
            p_x = np.polyfit(kn, sx_s[near], 1)
            p_y = np.polyfit(kn, sy_s[near], 1)
            sx_fit = np.polyval(p_x, kn)
            sy_fit = np.polyval(p_y, kn)
            r2_x = 1 - np.sum((sx_s[near] - sx_fit)**2) / max(np.sum((sx_s[near] - np.mean(sx_s[near]))**2), 1e-30)
            r2_y = 1 - np.sum((sy_s[near] - sy_fit)**2) / max(np.sum((sy_s[near] - np.mean(sy_s[near]))**2), 1e-30)
            r2_avg = (r2_x + r2_y) / 2
            if abs(p_y[0]) > 1e-10:
                r = p_x[0] / p_y[0]
                print(f"\n  {label}:  ⟨σ_x⟩ slope = {p_x[0]:.4f},  ⟨σ_y⟩ slope = {p_y[0]:.4f}")
                print(f"           α/β = {r:.4f}  (R²_x={r2_x:.3f}, R²_y={r2_y:.3f})")
                if r2_avg > best_r2:
                    best_r2 = r2_avg
                    best_ratio = r

        if best_ratio is not None:
            ratio = best_ratio
            print(f"\n  → α/β = {ratio:.4f}  (取最优拟合)")

    # ---- Part 3: Combine ----
    print()
    print("=" * 65)
    print("  Result")
    print("=" * 65)
    if sqrt_ab and ratio and abs(ratio) > 1e-6:
        alpha = sqrt_ab / np.sqrt(1 + 1 / ratio**2)
        beta = alpha / ratio
        _ctx['alpha_meva'] = alpha * 1000
        _ctx['beta_meva'] = beta * 1000
        print(f"\n  √(α²+β²) = {sqrt_ab*1000:.2f} meV·Å")
        print(f"  α/β     = {ratio:.4f}")
        print(f"  α       = {_ctx['alpha_meva']:.2f} meV·Å")
        print(f"  β       = {_ctx['beta_meva']:.2f} meV·Å")
        with open(os.path.join(OUTPUT_DIR, 'soc_report.txt'), 'w') as f:
            f.write("SOC Parameter Report\n")
            f.write(f"sqrt(a^2+b^2) = {sqrt_ab*1000:.2f} meV.A\n")
            f.write(f"alpha/beta    = {ratio:.4f}\n")
            f.write(f"alpha         = {alpha*1000:.2f} meV.A\n")
            f.write(f"beta          = {beta*1000:.2f} meV.A\n")
        print(f"  [报告 -> {OUTPUT_DIR}/soc_report.txt]")
    elif sqrt_ab and not ratio:
        print(f"\n  √(α²+β²) = {sqrt_ab*1000:.2f} meV·Å")
        print("  (缺少 α/β 比值, 需运行 PROCAR 部分)")
    elif not sqrt_ab and ratio:
        print(f"\n  α/β = {ratio:.4f}")
        print("  (缺少 √(α²+β²), 需运行 Wannier 部分)")


def main_spin_lifetime_menu():
    """自旋寿命: 自动装载上下文结果, 输入不足的, 即算即得"""
    print()
    print("=" * 65)
    print("  自旋寿命 (Spin lifetime τ_s)")
    print("=" * 65)
    print()
    print("  τ_s = ℏ² / (2 · m* · m₀ · α_eff² · τ_p)")
    print("  τ_p = μ · m* / e   (μ 反算)")
    print()

    ctx = _ctx
    # 收集缺失的输入: m*, α, β, τ_p
    # 若有 μ 则反算 τ_p; 若均缺则手工输入
    # 最终自动计算 τ_s, L_PSH

    # --- 1. 有效质量 m* ---
    m_star = ctx['m_star']
    m_star_src = "Wannier" if m_star else None
    if m_star:
        print(f"  [来自 {m_star_src}] m* = {m_star:.4f} m₀")
    try:
        inp = input(f"  -->> m* (m₀) [{m_star or ''}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        inp = ""
    if inp:
        m_star = float(inp)
        if not ctx['m_star']:
            ctx['m_star'] = m_star
    elif m_star is None:
        print("  [需要 m*, 请先运行选项 2 或手工输入]")
        return

    # --- 2. α, β ---
    alpha = ctx['alpha_meva']
    beta = ctx['beta_meva']
    if alpha and beta:
        print(f"  [来自 SOC 计算] α = {alpha:.2f}, β = {beta:.2f} meV·Å")
    try:
        inp_a = input(f"  -->> α (meV·Å) [{alpha or ''}]: ").strip()
        inp_b = input(f"  -->> β (meV·Å) [{beta or ''}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        inp_a = inp_b = ""
    if inp_a:
        alpha = float(inp_a)
    if inp_b:
        beta = float(inp_b)
    if alpha is None or beta is None:
        print("  [需要 α, β, 请先运行选项 3]")
        return
    # 同步回 ctx
    if not ctx['alpha_meva'] and alpha:
        ctx['alpha_meva'] = alpha
    if not ctx['beta_meva'] and beta:
        ctx['beta_meva'] = beta

    # --- 3. τ_p (直接输入 或 从 μ 反算) ---
    tau_p = ctx['tau_p']
    mu = ctx['mu']
    if tau_p:
        print(f"  [来自迁移率] τ_p = {tau_p:.4f} ps")
    elif mu:
        print(f"  [来自迁移率] μ = {mu:.2f} cm²/V·s → 可反算 τ_p")

    try:
        inp_t = input(f"  -->> τ_p (ps) [留空用 μ 反算 / {tau_p or ''}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        inp_t = ""
    if inp_t:
        tau_p = float(inp_t)
    elif tau_p is None and mu:
        # 反算 τ_p = μ · m* / e
        m_kg = m_star * m0
        tau_p = mu * 1e-4 * m_kg / e_ch * 1e12
        print(f"  → τ_p = {tau_p:.4f} ps  (由 μ = {mu:.2f} cm²/V·s 反算)")
    elif tau_p is None:
        try:
            inp_mu = input(f"  -->> μ (cm²/V·s) [反算 τ_p]: ").strip()
        except (EOFError, KeyboardInterrupt):
            inp_mu = ""
        if inp_mu:
            mu = float(inp_mu)
            m_kg = m_star * m0
            tau_p = mu * 1e-4 * m_kg / e_ch * 1e12
            print(f"  → τ_p = {tau_p:.4f} ps  (由 μ = {mu:.2f} cm²/V·s 反算)")
            ctx['mu'] = mu
        else:
            print("  [需要 τ_p 或 μ]")
            return
    if tau_p:
        ctx['tau_p'] = tau_p

    # --- 4. 温度 ---
    T = ctx.get('T', 300)
    try:
        inp_T = input(f"  -->> T (K) [{T}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        inp_T = ""
    if inp_T:
        T = float(inp_T)
        ctx['T'] = T

    # --- 5. 计算 ---
    alpha_eV = alpha * 1e-3
    beta_eV = beta * 1e-3
    result = calc_spin_lifetime(alpha_eV, beta_eV, m_star, tau_p, T)

    if result:
        print(f"\n  α_eff = {result['alpha_eff_meva']:.2f} meV·Å")
        print(f"  τ_s   = {result['tau_s_ps']:.2f} ps")
        print(f"  L_PSH = {result['L_PSH_um']:.2f} μm")
        with open(os.path.join(OUTPUT_DIR, 'spinlife_report.txt'), 'w') as f:
            f.write("Spin Lifetime Report\n")
            f.write(f"m*     = {m_star:.4f} m0\n")
            f.write(f"alpha  = {alpha:.2f} meV.A\n")
            f.write(f"beta   = {beta:.2f} meV.A\n")
            f.write(f"tau_p  = {tau_p:.2f} ps\n")
            f.write(f"T      = {T} K\n")
            f.write(f"tau_s  = {result['tau_s_ps']:.2f} ps\n")
            f.write(f"L_PSH  = {result['L_PSH_um']:.2f} um\n")
        print(f"  [报告 -> {OUTPUT_DIR}/spinlife_report.txt]")


def main_dump_band_menu():
    """导出能带数据子菜单"""
    print()
    print("=" * 65)
    print("  导出能带数据")
    print("=" * 65)
    print()
    print("  1)  PROCAR 能带 (k, E, ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩)")
    print("  2)  Wannier 能带 (k, E)")
    print("  0)  返回")
    print()
    c = input("  -->> ").strip()

    if c == '1':
        path = input("  -->> PROCAR 路径: ").strip()
        if not path or not os.path.exists(path):
            print("  [文件不存在]")
            return
        procar = PROCAR(path)
        band = int(input(f"  能带序号 (1-{procar.nbands}): "))
        dump_band_data(procar, band)

    elif c == '2':
        path = input("  -->> Wannier 能带文件路径: ").strip()
        if not path or not os.path.exists(path):
            print("  [文件不存在]")
            return
        k_frac, energies, nk, nbands = read_bands(path)
        band = int(input(f"  能带序号 (1-{nbands}): ")) - 1
        a = float(input("  -->> 晶格常数 a (Å): ") or "1")
        k = k_to_reciprocal(k_frac, a)

        out = os.path.join(OUTPUT_DIR, f"wannier_band_{band+1}_data.txt")
        with open(out, 'w') as f:
            f.write("# k(A^-1)  E(eV)\n")
            for i in range(nk):
                f.write(f"{k[i]:.8f}  {energies[i, band]:.8f}\n")
        print(f"\n  [{nk} points -> {out}]")


if __name__ == '__main__':
    main()
