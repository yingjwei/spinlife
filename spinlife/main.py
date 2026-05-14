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
from spinlife.fit_soc import (fit_effmass, fit_alpha_beta,
                              fit_alpha_beta_band_average, calc_spin_lifetime)
from spinlife.mobility.calc_mobility import (read_POSCAR_A0, read_POSCAR_a,
                                             fit_C2D, fit_E1,
                                             calc_mu, C2D_Jm2_from_d2E,
                                             m0, e_ch)
from spinlife.wannier import load_band_data, read_bands, k_to_reciprocal, find_extremum, band_slice, read_labelinfo

# 所有输出文件统一放入 spinlife/ 目录 (放在最前, 函数默认参数需用到)
OUTPUT_DIR = 'spinlife'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _clean_input(text):
    """去掉输入中的控制字符 (解决终端 Backspace 回显 ^H 等问题)."""
    return ''.join(c for c in text if c >= ' ' or c == '\t').strip()


def _prompt_path(prompt, default_names=None, allow_skip=False):
    """
    智能文件路径输入: 自动检测当前目录下的常见文件名, 用户可回车确认.

    Parameters
    ----------
    prompt : str
    default_names : list[str] | None — 自动检测的文件名列表
    allow_skip : bool — 是否允许跳过 (返回 None)

    Returns
    -------
    str | None — 文件路径, 或 None 表示跳过
    """
    # 检测默认文件
    hint = ""
    auto = None
    if default_names:
        for name in default_names:
            if os.path.exists(name):
                auto = name
                hint = f" [检测到: {name}]"
                break

    if auto and allow_skip:
        full = f"{prompt}{hint} (Enter 确认, 留空跳过): "
    elif auto:
        full = f"{prompt}{hint} (Enter 确认): "
    elif allow_skip:
        full = f"{prompt} (留空跳过): "
    else:
        full = f"{prompt}: "

    raw = input(f"\n  -->> {full}")
    text = _clean_input(raw)

    if not text and auto:
        return auto
    if not text and allow_skip:
        return None
    if text:
        return text
    return auto  # allow_skip=False 且无输入但有默认


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

    k_step = (np.min(np.diff(sorted(set(k_scan))))
              if len(set(k_scan)) > 1 else 0.05)
    auto_k_range = max(k_range, k_step * 1.5)

    # --- SOC 劈裂 + 能带平均法 ---
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

    # --- 能带平均法: 同时拟合 m* 和 sqrt(a^2+b^2) ---
    m_star, ab_norm, r2_p, r2_l, n_pts = fit_alpha_beta_band_average(
        k_scan, E_up_scan, E_lo_scan, k0, auto_k_range)

    res['m_star'] = m_star
    if m_star:
        print(f"  m* = {m_star:.2f} m0  (parabola R^2={r2_p:.4f}, "
              f"k0={k0:.3f}, range=+/-{auto_k_range:.3f}, {n_pts} pts)")
        _ctx['m_star'] = m_star

    if ab_norm:
        _ctx['ab_norm_meva'] = ab_norm * 1000
        print(f"  sqrt(a^2+b^2) = {ab_norm*1000:.2f} meV.A  "
              f"(linear R^2={r2_l:.4f})")
        res.update(ab_norm=ab_norm, r2_parab=r2_p, r2_linear=r2_l)

        # --- 自旋织构 -> alpha/beta 分离 ---
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

                # --- 自旋寿命 (优先用 ab_norm) ---
                if m_star:
                    spin = calc_spin_lifetime(ab_norm=ab_norm*1000,
                                              m_star=m_star, tau_p=tau_p, T=T)
                    if spin:
                        print(f"  tau_s  = {spin['tau_s_ps']:.2f} ps  "
                              f"(DP via sqrt(a^2+b^2))")
                        print(f"  L_PSH  = {spin['L_PSH_um']:.2f} um")
                        res['spin'] = spin
            else:
                print(f"  (spin ratio unstable: {ratio:.3f}, skip)")
                res.update(alpha=None, beta=None, ratio=None)
                # ab_norm 仍可用
                if m_star:
                    spin = calc_spin_lifetime(ab_norm=ab_norm*1000,
                                              m_star=m_star, tau_p=tau_p, T=T)
                    if spin:
                        print(f"  tau_s  = {spin['tau_s_ps']:.2f} ps  "
                              f"(via sqrt(a^2+b^2) only)")
                        res['spin'] = spin
        else:
            print(f"  (only {np.sum(near)} k-points, skip spin texture)")
            res.update(alpha=None, beta=None, ratio=None)
    else:
        print(f"  alpha/beta: fit failed (R^2={r2_l})")

    return res


# 会话共享状态 — 各模块计算结果自动传递
_ctx = {
    'm_star': None,          # 有效质量 (m₀)
    'ab_norm_meva': None,    # √(α²+β²) (meV·Å) — 能带平均法
    'alpha_meva': None,      # α (meV·Å)
    'beta_meva': None,       # β (meV·Å)
    'tau_p': None,           # 动量散射时间 (ps)
    'mu': None,              # 迁移率 (cm²/V·s)
    'T': 300,                # 温度 (K)
}


def _ctx_summary():
    """当前工作区状态摘要"""
    parts = []
    if _ctx['m_star']:
        parts.append(f"m* = {_ctx['m_star']:.4f} m0")
    if _ctx['ab_norm_meva']:
        parts.append(f"ab = {_ctx['ab_norm_meva']:.2f} meV.A")
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
    print("  6)  生成 KPOINTS 文件 (在指定 k 点附近生成密集网格)")
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
        elif c == '6':
            main_genkpoints()
        else:
            print("  [无效选项]")
            continue
        if c in ('1', '2', '3', '4', '5', '6'):
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
            poscar = _prompt_path("POSCAR 路径", default_names=['POSCAR'], allow_skip=True)
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


def _plot_effmass_result(k, E, k0, m, r2, k_range, band_label, filename,
                         E_offset=0, sym_k=None, sym_labels=None):
    """绘制有效质量拟合图 (参考 fit_effective_mass.py 风格)"""
    if not _HAS_MPL:
        return
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(8, 5))

    # 原始数据
    ax.plot(k, E - E_offset, 'o', ms=6, color='#ff0000', label=f'{band_label}')

    # 极值点 (去掉，仅保留数据点和拟合线)

    # 拟合曲线
    mask = np.abs(k - k0) <= k_range
    if np.sum(mask) > 0:
        k_fit = k[mask] - k0
        E_fit = E[mask]
        A = np.polyfit(k_fit**2, E_fit, 1)[0]
        E0_fit = np.polyfit(k_fit**2, E_fit, 1)[1]
        ks = np.linspace(-k_range, k_range, 200)
        ax.plot(ks + k0, A * ks**2 + E0_fit - E_offset, '-', lw=2.5,
                color='#0000ff', label='Parabolic fit')
        ax.axvspan(k0 - k_range, k0 + k_range, alpha=0.06, color='#0000ff')

    # 高对称点
    if sym_k is not None and sym_labels is not None:
        ymin, ymax = ax.get_ylim()
        ax.set_xticks(sym_k)
        ax.tick_params(axis='x', length=6, direction='out')
        for sk, sl in zip(sym_k, sym_labels):
            ax.axvline(sk, color='black', ls='--', lw=0.8, alpha=0.7)
            ax.text(sk, ymin - 0.06 * (ymax - ymin), sl,
                    ha='center', va='top', fontsize=18, color='black')

    # 标注框
    if m and r2:
        txt = f'$m^*$ = {m:.1f} $m_0$\n$R^2$ = {r2:.3f}'
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, va='top', ha='right',
                fontsize=14,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='black', linewidth=1.2))

    ax.set_xlabel('')
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('E (eV)', fontsize=18)
    ax.legend(fontsize=14, frameon=True, edgecolor='black')
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"  [图片已保存: {filename}]")


def _save_band_data_txt(k, E, band_index, base_name):
    """保存能带数据到 txt 文件"""
    out = os.path.join(OUTPUT_DIR, f'band{band_index}_data.txt')
    data = np.column_stack((k, E))
    header = f"Band {band_index}\nk (1/A)    E (eV)"
    np.savetxt(out, data, fmt='%.8f', header=header)
    print(f"  [数据已保存: {out}]")


def main_effmass_wannier():
    """有效质量: 读 Wannier 能带 → 抛物线拟合 → m* (参考 fit_effective_mass.py)"""
    print()
    print("=" * 65)
    print("  有效质量 (Effective mass from Wannier bands)")
    print("=" * 65)

    path = _prompt_path("Wannier 能带文件路径", default_names=['wannier90_band.dat'])
    if not path or not os.path.exists(path):
        print(f"  [文件不存在]")
        return

    # 先读 labelinfo → 获取 nk_expected (用于两列堆叠格式拆带)
    labelinfo_path = path + '.labelinfo.dat'
    alt_labelinfo = os.path.join(os.path.dirname(path) or '.', 'wannier90.labelinfo.dat')
    labels_info = []
    if os.path.exists(labelinfo_path):
        labels_info = read_labelinfo(labelinfo_path)
    elif os.path.exists(alt_labelinfo):
        labels_info = read_labelinfo(alt_labelinfo)

    nk_expected = labels_info[-1][0] + 1 if labels_info else None
    if labels_info:
        labels_str = ' → '.join(lbl for _, _, lbl in labels_info)
        print(f"  k 路径: {labels_str}")
        print(f"  总 k 点数: {nk_expected}")

    # 解析能带数据
    k, energies, nk, nbands = load_band_data(path, nk_expected)
    print(f"  能带数: {nbands},  每条带 k 点数: {nk}")
    print(f"  k 范围: {k[0]:.6f} ~ {k[-1]:.6f}")

    # 高对称点信息
    if labels_info:
        print(f"\n--- k 路径 ---")
        for kidx, kdist, lbl in labels_info:
            print(f"  {lbl:>4s}  k_index={kidx},  k_dist={kdist:.4f}")

    # 用户输入 VBM 带序号
    try:
        band = int(input(f"\n  价带顶能带序号 (1-{nbands}): "))
    except (EOFError, KeyboardInterrupt):
        print(f"  [输入取消]")
        return
    if band < 1 or band > nbands:
        print(f"  [错误: 能带序号超出范围 (1-{nbands})]")
        return

    vbm_idx = band - 1
    cbm_idx = vbm_idx + 1

    # 准备高对称点标注 (k 点数一致时才启用)
    plot_sym_k = None
    plot_sym_labels = None
    if labels_info and nk_expected and len(k) == nk_expected:
        plot_sym_k = [k[kidx] for kidx, _, _ in labels_info]
        plot_sym_labels = [lbl.replace('GAMMA', 'Γ').replace('G', 'Γ')
                          for _, _, lbl in labels_info]

    fit_ranges = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

    # ===== VBM 有效质量 =====
    E_vbm = energies[:, vbm_idx]
    k_vbm = k[np.argmax(E_vbm)]
    vbm_arg = np.argmax(E_vbm)

    print(f"\n{'='*60}")
    print(f"  VBM (Band {band})")
    print(f"{'='*60}")
    print(f"  位置: k = {k_vbm:.6f}")
    print(f"  能量: E = {E_vbm[vbm_arg]:.6f} eV")
    print(f"\n{'拟合范围':>10s}  {'m*/m0':>8s}  {'R²':>8s}  {'点数'}")

    best_vbm, best_r2_vbm, best_kr_vbm = None, 0, None
    for kr in fit_ranges:
        m_star, r2, n_pts = fit_effmass(k, E_vbm, k_vbm, kr)
        if m_star is None:
            continue
        if r2 > best_r2_vbm and m_star < 999:
            best_r2_vbm = r2
            best_vbm = m_star
            best_kr_vbm = kr
        n = np.sum(np.abs(k - k_vbm) <= kr)
        flag = "  ← 最优" if (r2 > 0.97 and m_star == best_vbm) else ""
        print(f"  k≤{kr:<.3f}   {m_star:>8.1f}  {r2:>7.3f}  {n:3d}{flag}")

    if best_vbm:
        print(f"\n  → VBM: m* = {best_vbm:.1f} m₀  (范围 ±{best_kr_vbm}, R² = {best_r2_vbm:.4f})")
        _ctx['m_star'] = best_vbm

        # 保存数据
        _save_band_data_txt(k, E_vbm, band, path)

        # 报告
        with open(os.path.join(OUTPUT_DIR, 'effmass_report.txt'), 'w') as f:
            f.write("Effective Mass Report (Wannier)\n")
            f.write(f"File: {path}\n")
            f.write(f"VBM Band: {band}\n")
            f.write(f"m*_VBM = {best_vbm:.4f} m₀\n")
            f.write(f"R²_VBM = {best_r2_vbm:.6f}\n")
            f.write(f"Fit range: ±{best_kr_vbm}\n")
        print(f"  [报告 -> {OUTPUT_DIR}/effmass_report.txt]")

        # 绘图
        _plot_effmass_result(k, E_vbm, k_vbm, best_vbm, best_r2_vbm, best_kr_vbm,
                             f"Band {band} (VBM)",
                             os.path.join(OUTPUT_DIR, 'effmass_fit.png'),
                             E_offset=E_vbm[vbm_arg],
                             sym_k=plot_sym_k, sym_labels=plot_sym_labels)
    else:
        print("  VBM 拟合失败 (所有 R² 偏低)")

    # ===== CBM 有效质量 =====
    if cbm_idx < nbands:
        E_cbm = energies[:, cbm_idx]
        k_cbm = k[np.argmin(E_cbm)]
        cbm_arg = np.argmin(E_cbm)

        print(f"\n{'='*60}")
        print(f"  CBM (Band {band+1})")
        print(f"{'='*60}")
        print(f"  位置: k = {k_cbm:.6f}")
        print(f"  能量: E = {E_cbm[cbm_arg]:.6f} eV")
        print(f"\n{'拟合范围':>10s}  {'m*/m0':>8s}  {'R²':>8s}  {'点数'}")

        best_cbm, best_r2_cbm, best_kr_cbm = None, 0, None
        for kr in fit_ranges:
            m_star, r2, n_pts = fit_effmass(k, E_cbm, k_cbm, kr)
            if m_star is None:
                continue
            if r2 > best_r2_cbm and m_star < 999:
                best_r2_cbm = r2
                best_cbm = m_star
                best_kr_cbm = kr
            n = np.sum(np.abs(k - k_cbm) <= kr)
            flag = "  ← 最优" if (r2 > 0.97 and m_star == best_cbm) else ""
            print(f"  k≤{kr:<.3f}   {m_star:>8.1f}  {r2:>7.3f}  {n:3d}{flag}")

        if best_cbm:
            print(f"\n  → CBM: m* = {best_cbm:.1f} m₀  (范围 ±{best_kr_cbm}, R² = {best_r2_cbm:.4f})")
            _save_band_data_txt(k, E_cbm, band + 1, path)
            _plot_effmass_result(k, E_cbm, k_cbm, best_cbm, best_r2_cbm, best_kr_cbm,
                                 f"Band {band+1} (CBM)",
                                 os.path.join(OUTPUT_DIR, 'effmass_cbm_fit.png'),
                                 E_offset=E_cbm[cbm_arg],
                                 sym_k=plot_sym_k, sym_labels=plot_sym_labels)

            # 带隙
            gap = np.min(E_cbm) - np.max(E_vbm)
            print(f"\n  Eg (Band {band} - {band+1}) = {gap:.4f} eV")

            # 追加 CBM 到报告
            with open(os.path.join(OUTPUT_DIR, 'effmass_report.txt'), 'a') as f:
                f.write(f"\nCBM Band: {band+1}\n")
                f.write(f"m*_CBM = {best_cbm:.4f} m₀\n")
                f.write(f"R²_CBM = {best_r2_cbm:.6f}\n")
                f.write(f"Fit range: ±{best_kr_cbm}\n")
                f.write(f"Eg = {gap:.4f} eV\n")
        else:
            print("  CBM 拟合失败 (所有 R² 偏低)")

    # 总结
    print(f"\n{'='*60}")
    print(f"  推荐参数")
    print(f"{'='*60}")
    if best_vbm:
        print(f"  m*_VBM = {best_vbm:.1f} m₀  (R²={best_r2_vbm:.3f})")
    if cbm_idx < nbands and best_cbm:
        print(f"  m*_CBM = {best_cbm:.1f} m₀  (R²={best_r2_cbm:.3f})")
        if 'gap' in locals():
            print(f"  Eg     = {gap:.4f} eV")
    print(f"\n  说明: 选取 R² > 0.97 的最小范围结果为最佳")


def main_alpha_beta():
    """SOC α/β: band-averaging → √(α²+β²) + m*, PROCAR spin → α/β ratio"""
    print()
    print("=" * 65)
    print("  SOC 参数 α/β 计算")
    print("=" * 65)
    print()
    print("  方法: Wannier SOC 双带 → 能带平均法 → m* + √(α²+β²)")
    print("        PROCAR 自旋织构斜率拟合 → α/β 比值")
    print()

    # ---- Part 1: √(α²+β²) from Wannier ----
    sqrt_ab = None
    path = _prompt_path("Wannier 能带文件路径", default_names=['wannier90_band.dat'], allow_skip=True)
    if path and os.path.exists(path):
        print("\n  [1/2] √(α²+β²) — Wannier SOC 能带拟合")
        k, energies, nk, nbands = load_band_data(path)
        if nbands == 0:
            print("  [错误: 未解析到能带, 文件格式可能不正确]")
            return
        labelinfo_path = path + '.labelinfo.dat'
        labels = read_labelinfo(labelinfo_path) if os.path.exists(labelinfo_path) else []
        if labels:
            print(f"  高对称点: {' → '.join(lbl for _, _, lbl in labels)}")
        a = None
        for poscar_name in ['POSCAR', 'CONTCAR']:
            if os.path.exists(poscar_name):
                a = read_POSCAR_a(poscar_name)
                if a:
                    print(f"  晶格常数 a = {a:.4f} Å  (来自 {poscar_name})")
                    break
        if not a:
            a = float(input("  -->> 晶格常数 a (Å): "))
        k = k_to_reciprocal(k, a)

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

        m_star, ab_norm, r2_p, r2_l, n_pts = fit_alpha_beta_band_average(k, E_up, E_lo, k0, kr)
        if ab_norm:
            sqrt_ab = ab_norm
            print(f"\n  √(α²+β²) = {sqrt_ab*1000:.2f} meV·Å  (linear R²={r2_l:.4f})")
            if m_star:
                print(f"  m*      = {m_star:.2f} m0  (parabola R²={r2_p:.4f}, {n_pts} pts)")
                _ctx['m_star'] = m_star

            # 绘图: 能带平均 + 拟合
            if _HAS_MPL:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                # 左: SOC 能带 + 带平均
                E_avg = (E_up + E_lo) / 2
                mask_k = np.abs(k - k0) <= kr
                ax1.plot(k, E_up * 1000, 'o-', ms=2, lw=1, label=f'Band {up+1}')
                ax1.plot(k, E_lo * 1000, 's-', ms=2, lw=1, label=f'Band {lo+1}')
                ax1.plot(k, E_avg * 1000, '-', lw=2, color='#2E8B57', alpha=0.7, label='E_avg')
                if m_star:
                    A_coeff = 3.81 / m_star
                    k2_fit = (k[mask_k] - k0) ** 2
                    E_parab = (A_coeff * k2_fit + np.mean(E_avg[mask_k])) * 1000
                    ax1.plot(k[mask_k], E_parab, '--', lw=1.5, color='#1a6b3c', label='parabola fit')
                ax1.axvspan(k0 - kr, k0 + kr, alpha=0.12, color='blue', label='fit range')
                ax1.axvline(k0, color='gray', ls='--', alpha=0.4)
                ax1.set_xlabel('k (Ang^-1)'); ax1.set_ylabel('E (meV)')
                ax1.legend(fontsize=8); ax1.set_title('SOC Bands + Band Average')
                ax1.grid(alpha=0.3)
                # 右: |E_up - E_lo|/2 vs |k-k0| (linear fit)
                abs_k = np.abs(k - k0)
                dE_half = np.abs(E_up - E_lo) / 2 * 1000
                ax2.plot(abs_k, dE_half, 'o', ms=5, color='#E24A33')
                mask_l = (abs_k <= kr) & (abs_k > 1e-10)
                if np.sum(mask_l) >= 2:
                    c_lin = np.polyfit(abs_k[mask_l], dE_half[mask_l], 1)
                    xs = np.linspace(0, kr * 1.05, 100)
                    ax2.plot(xs, c_lin[0] * xs + c_lin[1], '-', color='#348ABD',
                             label=f'Fit slope={c_lin[0]:.2f} meV/A')
                ax2.set_xlabel(r'$|k - k_0|$  (Ang^{-1}$)')
                ax2.set_ylabel(r'$|E_{up} - E_{lo}|/2$  (meV)')
                ax2.legend(fontsize=9); ax2.set_title(r'Linear fit -> sqrt(alpha^2+beta^2)')
                ax2.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'soc_wannier_fit.png'), dpi=200, bbox_inches='tight')
                plt.close()
                print(f"  [Plot -> {OUTPUT_DIR}/soc_wannier_fit.png]")
    elif path:
        print("  [文件不存在]")

    # ---- Part 2: α/β ratio from PROCAR (斜率拟合, 非点对点平均) ----
    ratio = None
    procar_path = _prompt_path("PROCAR 路径", default_names=['PROCAR'], allow_skip=True)
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

            # 绘图: ⟨σ_x⟩ & ⟨σ_y⟩ vs k + 线性拟合
            if _HAS_MPL:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                for label, sx_s, sy_s, color in [
                    (f"Band {up}", sx_up_s, sy_up_s, '#E24A33'),
                    (f"Band {lo}", sx_lo_s, sy_lo_s, '#348ABD'),
                ]:
                    p_x = np.polyfit(kn, sx_s[near], 1)
                    p_y = np.polyfit(kn, sy_s[near], 1)
                    k_fit_line = np.linspace(min(kn), max(kn), 100)
                    ax1.plot(k_scan[near], sx_s[near], 'o', ms=6, color=color, label=f'{label} data')
                    ax1.plot(k0 + k_fit_line, np.polyval(p_x, k_fit_line), '-',
                             color=color, alpha=0.6, label=f'{label} fit')
                    ax2.plot(k_scan[near], sy_s[near], 'o', ms=6, color=color, label=f'{label} data')
                    ax2.plot(k0 + k_fit_line, np.polyval(p_y, k_fit_line), '-',
                             color=color, alpha=0.6, label=f'{label} fit')
                ax1.axhline(0, color='gray', ls='--', alpha=0.3)
                ax1.set_xlabel('k (Å⁻¹)'); ax1.set_ylabel(r'$\langle\sigma_x\rangle$')
                ax1.legend(fontsize=8); ax1.set_title(r'$\langle\sigma_x\rangle$ vs k')
                ax1.grid(alpha=0.3)
                ax2.axhline(0, color='gray', ls='--', alpha=0.3)
                ax2.set_xlabel('k (Å⁻¹)'); ax2.set_ylabel(r'$\langle\sigma_y\rangle$')
                ax2.legend(fontsize=8); ax2.set_title(r'$\langle\sigma_y\rangle$ vs k')
                ax2.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'soc_spin_fit.png'), dpi=200, bbox_inches='tight')
                plt.close()
                print(f"  [Plot -> {OUTPUT_DIR}/soc_spin_fit.png]")

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
        path = _prompt_path("PROCAR 路径", default_names=['PROCAR'])
        if not path or not os.path.exists(path):
            print(f"  [文件不存在: {path}]")
            return
        procar = PROCAR(path)
        band = int(input(f"  能带序号 (1-{procar.nbands}): "))
        dump_band_data(procar, band)

    elif c == '2':
        path = _prompt_path("Wannier 能带文件路径", default_names=['wannier90_band.dat'])
        if not path or not os.path.exists(path):
            print(f"  [文件不存在: {path}]")
            return
        k, energies, nk, nbands = load_band_data(path)
        band = int(input(f"  能带序号 (1-{nbands}): ")) - 1
        a = None
        for poscar_name in ['POSCAR', 'CONTCAR']:
            if os.path.exists(poscar_name):
                a = read_POSCAR_a(poscar_name)
                if a:
                    print(f"  晶格常数 a = {a:.4f} Å  (来自 {poscar_name})")
                    break
        if not a:
            a = float(input("  -->> 晶格常数 a (Å): ") or "1")
        k = k_to_reciprocal(k, a)

        out = os.path.join(OUTPUT_DIR, f"wannier_band_{band+1}_data.txt")
        with open(out, 'w') as f:
            f.write("# k(A^-1)  E(eV)\n")
            for i in range(nk):
                f.write(f"{k[i]:.8f}  {energies[i, band]:.8f}\n")
        print(f"\n  [{nk} points -> {out}]")


def main_genkpoints():
    """选项 6: 读 Wannier 能带 → 选 SOC 带对 → 找 k₀ → 生成 KPOINTS"""
    from spinlife.genkpoints import generate_grid, write_kpoints

    print()
    print("=" * 65)
    print("  生成 KPOINTS — 从 Wannier 能带确定中心点后生成密集网格")
    print("=" * 65)
    print()

    # 1) 读 Wannier 能带
    path = _prompt_path("Wannier 能带文件路径", default_names=['wannier90_band.dat'])
    if not path or not os.path.exists(path):
        print("  [文件不存在]")
        return

    labelinfo_path = path + '.labelinfo.dat'
    labels_info = []
    if os.path.exists(labelinfo_path):
        labels_info = read_labelinfo(labelinfo_path)
    nk_expected = labels_info[-1][0] + 1 if labels_info else None

    k, energies, nk, nbands = load_band_data(path, nk_expected)
    print(f"  能带数: {nbands},  k 点数: {nk}")

    if labels_info:
        print(f"  k 路径: {' → '.join(lbl for _, _, lbl in labels_info)}")

    # 2) 显示参考能带
    mid = nk // 2
    print(f"\n  路径中点附近能带 (用于选带):")
    print(f"  {'Band':>6}  {'Energy(eV)':>12}")
    low = max(0, nbands // 2 - 6)
    high = min(nbands, nbands // 2 + 7)
    for b in range(low, high):
        print(f"  {b+1:>6}  {energies[mid, b]:>12.4f}")

    # 3) 选 SOC 带对 → 找带边极值 (VBM 取上带 max, CBM 取下带 min)
    up = int(input(f"\n  -->> SOC 上能带 (1-{nbands}): ")) - 1
    lo = int(input(f"  -->> SOC 下能带 (1-{nbands}): ")) - 1
    E_up = energies[:, up]
    E_lo = energies[:, lo]

    mode = input("  -->> 极值类型 (VBM=上带求最大, CBM=下带求最小): ").strip().upper()
    if mode.startswith('C'):
        k0_idx = np.argmin(E_lo)
        k0 = k[k0_idx]
        print(f"\n  CBM (Band {lo+1} 最小): k₀ = {k0:.6f},  E = {E_lo[k0_idx]:.4f} eV")
    else:
        k0_idx = np.argmax(E_up)
        k0 = k[k0_idx]
        print(f"\n  VBM (Band {up+1} 最大): k₀ = {k0:.6f},  E = {E_up[k0_idx]:.4f} eV")

    # 4) 确认中心点
    cx = float(input(f"\n  -->> KPOINTS 中心 kx [{k0:.6f}]: ") or k0)
    cy = float(input(f"  -->> KPOINTS 中心 ky [0]: ") or 0)
    cz = float(input(f"  -->> KPOINTS 中心 kz [0]: ") or 0)
    center = [cx, cy, cz]

    # 5) 网格参数
    k_range = float(input("  -->> 范围 ± (分数坐标) [0.05]: ") or 0.05)
    nx = int(input("  -->> kx 方向点数: ") or 5)
    ny = int(input("  -->> ky 方向点数: ") or 5)
    nz = int(input("  -->> kz 方向点数: ") or 1)
    out = input("  -->> 输出文件名 [KPOINTS]: ").strip() or "KPOINTS"

    kpoints = generate_grid(center, k_range, nx, ny, nz)
    write_kpoints(out, kpoints,
                  comment=f"SOC band {up+1}/{lo+1} k0={k0:.4f}",
                  center=center, k_range=k_range, nx=nx, ny=ny, nz=nz)


if __name__ == '__main__':
    main()
