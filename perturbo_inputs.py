# Perturbo 输入文件生成 - W6CCl16 (真实结构)
# 基于 QE + Wannier90 计算结果

import os
from params import *

QE_PREFIX = "wannier90"  # QE 计算的 prefix，对应 wannier90.mmn/.amn/.eig
                          # 这些文件应在 QE outdir 中


# ============================================================
# Perturbo qe2pert.in — QE → Perturbo 数据转换
# ============================================================
def make_qe2pert_input(calc_dir="calc/perturbo"):
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "qe2pert.in")
    with open(fname, "w") as f:
        f.write(f"""&perturbo
    prefix = '{QE_PREFIX}'
    outdir = '../scf/out/'
    itype = 0
/
""")
    print(f"[Perturbo] Generated: {fname}")
    return fname


# ============================================================
# Perturbo teout.in — 电子-声子散射矩阵
# ============================================================
def make_teout_input(calc_dir="calc/perturbo"):
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "teout.in")
    with open(fname, "w") as f:
        f.write(f"""&perturbo
    prefix = '{QE_PREFIX}'
    outdir = '../scf/out/'
    itype = 1
/

&cell
/

&system
/

&bands
/

&kpoints
/

&fermi_energy
    fermi_energy_type = 'from_energies'
/

&elph_mat
/
""")
    print(f"[Perturbo] Generated: {fname}")
    return fname


# ============================================================
# Perturbo td.in — 自旋动力学 / 自旋寿命
# ============================================================
def make_tdin_input(calc_dir="calc/perturbo"):
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "td.in")
    content = f"""&perturbo
    prefix = '{QE_PREFIX}'
    outdir = '../scf/out/'
    itype = 4
/

&cell
/

&system
/

&bands
/

&kpoints
/

&fermi_energy
    fermi_energy_type = 'from_energies'
/

&spin
    spin_qe_flag = .true.
/

&temp
    temperatures(1) = {temp}
/

&deph
    deph_scatter = 'dephasing'
    wmin = {energy_min}
    wmax = {energy_max}
    deph_targ = 0.2
/

&scbt
    calc_cond    = .true.
    calc_spin    = .true.
    calc_transport = .false.
    smear_type   = 'lorentzian'
    eta          = {eta}
/
"""
    with open(fname, "w") as f:
        f.write(content)
    print(f"[Perturbo] Generated: {fname}")
    return fname


# ============================================================
# Perturbo batch — 多温度扫描 (可选)
# ============================================================
def make_tdin_batch(temps, calc_dir="calc/perturbo"):
    """生成多温度 td.in 批处理"""
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "td_batch.in")
    temps_str = ", ".join(str(t) for t in temps)
    content = f"""&perturbo
    prefix = '{QE_PREFIX}'
    outdir = '../scf/out/'
    itype = 4
/

&cell
/

&system
/

&bands
/

&kpoints
/

&fermi_energy
    fermi_energy_type = 'from_energies'
/

&spin
    spin_qe_flag = .true.
/

&temp
    temperatures({len(temps)}) = {temps_str}
/

&deph
    deph_scatter = 'dephasing'
    wmin = {energy_min}
    wmax = {energy_max}
    deph_targ = 0.2
/

&scbt
    calc_cond       = .true.
    calc_spin       = .true.
    calc_transport  = .false.
    smear_type      = 'lorentzian'
    eta             = {eta}
/
"""
    with open(fname, "w") as f:
        f.write(content)
    print(f"[Perturbo] Generated: {fname}")
    return fname


if __name__ == "__main__":
    print("生成 Perturbo 输入文件 (真实结构)...")
    make_qe2pert_input()
    make_teout_input()
    make_tdin_input()
    # 多温度示例: 100K, 200K, 300K, 400K
    # make_tdin_batch([100, 200, 300, 400])
    print("完成！")
