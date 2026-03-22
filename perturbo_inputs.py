# Perturbo 输入文件生成 - W6CCl16 自旋寿命

import os
from params import *


# ============================================================
# Perturbo teout.in (计算电子-声子散射)
# ============================================================
def make_teout_input():
    """
    电子-声子散射计算
    teout = transport electron-outscattering
    """
    content = f"""&perturbo
    prefix = '{material}'
    outdir = './out/'
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
"""
    return content


# ============================================================
# Perturbo td(in) - 自旋动力学
# ============================================================
def make_tdin_input(calc_dir="calc/perturbo"):
    os.makedirs(calc_dir, exist_ok=True)
    content = f"""&perturbo
    prefix = '{material}'
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
    calc_cond = .true.
    calc_spin = .true.
    calc_transport = .false.
    smear_type = 'lorentzian'
    eta = {eta}
    /
"""
    fname = os.path.join(calc_dir, "td.in")
    with open(fname, "w") as f:
        f.write(content)
    print(f"[Perturbo] Generated: {fname}")
    return fname


# ============================================================
# Perturbo qe2pert.in - QE → Perturbo 数据转换
# ============================================================
def make_qe2pert_input(calc_dir="calc/perturbo"):
    os.makedirs(calc_dir, exist_ok=True)
    content = f"""&perturbo
    prefix = '{material}'
    outdir = '../scf/out/'
    itype = 0
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
"""
    fname = os.path.join(calc_dir, "qe2pert.in")
    with open(fname, "w") as f:
        f.write(content)
    print(f"[Perturbo] Generated: {fname}")
    return fname


if __name__ == "__main__":
    print("生成 Perturbo 输入文件...")
    make_tdin_input()
    make_qe2pert_input()
    print("完成！")
