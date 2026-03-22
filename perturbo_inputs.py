# Perturbo 输入文件生成 - W6CCl16 (真实结构)
# 完整流程: QE SCF → QE NSCF → QE DFPT(ph) → EPW → Perturbo ephmat → Perturbo td

import os
from params import *

QE_PREFIX = "wannier90"  # QE 计算的 prefix


# ============================================================
# Perturbo qe2pert.in — QE → Perturbo 数据转换
# ============================================================
def make_qe2pert_input(calc_dir="calc/perturbo"):
    """QE 输出转换为 Perturbo 可读格式"""
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
# Perturbo ephmat.in — 电子-声子散射矩阵
#   数据来源: QE DFPT 声子计算 + EPW
#   itype = 3
# ============================================================
def make_ephmat_input(calc_dir="calc/perturbo"):
    """
    计算电子-声子散射矩阵元 g(k,q)
    必需: QE DFPT 声子 (dynmat) + EPW el-ph 耦合
    也可直接从 EPW 输出读取
    """
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "ephmat.in")
    content = f"""&perturbo
    prefix = '{QE_PREFIX}'
    outdir = '../scf/out/'
    itype = 3
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

&ephmat
    ! 来自 EPW 计算的 elph 文件
    ! 如果用 EPW，设为 'epw'
    filelph = 'elph_from_epw'
    deph_smear = 'gaussian'
    degauss = 0.01
/

&temp
    temperatures(1) = {temp}
/

&elph_mat
/
"""
    with open(fname, "w") as f:
        f.write(content)
    print(f"[Perturbo] Generated: {fname}")
    return fname


# ============================================================
# Perturbo td.in — 自旋动力学 / 自旋寿命
#   itype = 4
# ============================================================
def make_tdin_input(calc_dir="calc/perturbo"):
    """
    自旋动力学: 计算自旋寿命 τ_s
    使用 el-ph 散射（来自 ephmat）
    """
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
    ! deph_scatter = 'elph'  ← 用 EPW/ephmat 的 el-ph 散射
    ! deph_scatter = 'dephasing' ← 用 dephasing 近似 (不需完整 eph)
    deph_scatter = 'elph'
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


# ============================================================
# Perturbo batch — 多温度扫描
# ============================================================
def make_tdin_batch(temps, calc_dir="calc/perturbo"):
    """多温度自旋寿命计算"""
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "td_batch.in")
    temps_str = ", ".join(str(t) for t in temps)
    ntemp = len(temps)
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
    temperatures({ntemp}) = {temps_str}
/

&deph
    deph_scatter = 'elph'
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
    print("生成 Perturbo 输入文件...")
    make_qe2pert_input()
    make_ephmat_input()
    make_tdin_input()
    # 多温度示例: 100K, 200K, 300K, 400K, 500K
    make_tdin_batch([100, 200, 300, 400, 500])
    print("完成！")
