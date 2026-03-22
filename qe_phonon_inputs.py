# QE 声子计算输入文件生成器 - W6CCl16
# 用于 Perturbo 电子-声子散射矩阵
# 流程: ph.x (DFPT) → dyncubes → q2r.x → matdyn → perturbo ephmat

import os
from params import *
from qe_inputs import ATOMS, _atoms_str, _cell_params, PP_W, PP_C, PP_Cl

# ============================================================
# QE ph.in — DFPT 声子计算 (q 点网格)
# ============================================================
def make_ph_input(q_mesh=[3, 3, 1], calc_dir="calc/phonon"):
    """
    生成每个 q 点的声子输入文件
    W6CCl16: 2D 结构，q 点只用 3x3x1
    """
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "ph.in")

    # q 点网格 (与 mp_grid 对齐)
    nq1, nq2, nq3 = q_mesh

    content = f"""&inputph
    tr2_ph = 1.0d-14
    prefix = '{QE_PREFIX}'
    outdir = '../scf/out/'
    fildyn = 'W6CCl16.dyn'
    ldisp = .true.
    nq1 = {nq1}
    nq2 = {nq2}
    nq3 = {nq3}
    diag_ndim = 4
    epsil = .true.
    elop = .true.
    ! 2D 材料需要忽略 z 方向的偶极子校正
    asr = 'crystal'
    amass(1) = 183.84   ! W
    amass(2) = 12.011    ! C
    amass(3) = 35.453   ! Cl
/

""".strip()

    with open(fname, "w") as f:
        f.write(content)
    print(f"[PH] Generated: {fname}")
    print(f"     q-grid: {nq1}x{nq2}x{nq3} = {nq1*nq2*nq3} q-points")
    return fname


# ============================================================
# QE q2r.in — 将 dynmat 转换到实空间
# ============================================================
def make_q2r_input(calc_dir="calc/phonon"):
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "q2r.in")
    content = f"""&input
    fildyn = 'W6CCl16.dyn',
    filfrc = 'W6CCl16.fc',
    loto_2d = .true.,
    asr = 'crystal',
    q_in_band_form = .false.,
/
"""
    with open(fname, "w") as f:
        f.write(content)
    print(f"[Q2R] Generated: {fname}")
    return fname


# ============================================================
# QE matdyn.in — 声子色散计算
# ============================================================
def make_matdyn_input(
    q_path="W6CCl16.dyn",
    fc_file="W6CCl16.fc",
   nk1=5, nk2=5, nk3=1,
    calc_dir="calc/phonon"
):
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "matdyn.in")

    # 高对称点路径 (2D hex BZ)
    # Γ-M-K-Γ for ~hexagonal lattice
    # a=12.997, b=10.251  → 用 Γ-M-Γ (矩形) 或 Γ-K-M-Γ
    # 根据实际对称性选择
    qpoints = """
0.0  0.0  0.0   50  ! Γ
0.5  0.0  0.0   30  ! M
0.0  0.5  0.0   30  ! X (if ortho) or skip
0.0  0.0  0.0   50  ! Γ back
"""
    content = f"""&input
    asr = 'crystal',
    loto_2d = .true.,
    flfrc = '{fc_file}',
    fldyn = '{q_path}',
    q_in_band_form = .true.,
/

{len(qpoints.strip().splitlines())}
{qpoints}
"""
    with open(fname, "w") as f:
        f.write(content)
    print(f"[MATDYN] Generated: {fname}")
    return fname


# ============================================================
# QE epw.in — EPW 接口 (生成 el-ph 矩阵元)
#    EPW 可直接生成 Perturbo 所需的 el-ph 数据
#    如果用 Perturbo 自带的 qe2pert+ephmat，跳过此步
# ============================================================
def make_epw_input(
    nband=512,
    nk1=6, nk2=6, nk3=1,
    nqf1=3, nqf2=3, nqf3=1,
    calc_dir="calc/phonon"
):
    """
    生成 EPW 输入文件，用于计算电声耦合
    EPW = QE + Wannier90 的电声耦合接口
    输出可直接被 Perturbo 使用
    """
    os.makedirs(calc_dir, exist_ok=True)
    fname = os.path.join(calc_dir, "epw.in")

    content = f"""&inputepw
    prefix = '{QE_PREFIX}',
    outdir = '../scf/out/',
    dvscf_dir = '../ph/out/',
    iverbosity = 1,
    eliashb = .true.,

    ! Wannierization
    nbndsub = {num_wann},
    num_kiter = 500,
    dis_froz_min = {dis_froz_min},
    dis_froz_max = {dis_froz_max},
    dis_win_max = {dis_win_max},

    ! Fine k/q grids
    nk1 = {nk1},  nk2 = {nk2},  nk3 = {nk3},
    nqf1 = {nqf1}, nqf2 = {nqf2}, nqf3 = {nqf3},
    ! 建议: nk >= 2*nq 确保收敛

    ! Band range for el-ph
    nbnd = {nband},

    ! Self-energy parameters
    degauss = 0.015,   ! Gaussian broaden (eV)
    delta_ene = 0.05,  ! Energy delta for derivative
    nstemp = 1,
    temps = {temp},
/

"""
    with open(fname, "w") as f:
        f.write(content)
    print(f"[EPW] Generated: {fname}")
    return fname


# ============================================================
# Perturbo ephmat.in — 读取 el-ph 矩阵元 (QE DFPT 路径)
# ============================================================
def make_ephmat_input(calc_dir="calc/perturbo"):
    """
    Perturbo 的电子-声子散射计算
    使用 QE DFPT 计算得到的 dynmat/fc 文件
    itype = 3: 计算 el-ph 矩阵
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
    ! QE DFPT 产生的文件路径 (相对于 outdir)
    fildyn = '../ph/W6CCl16.dyn'
    filfrc = '../ph/W6CCl16.fc'
    ! 2D 材料
    loto_2d = .true.
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
# Perturbo td.in — 自旋动力学 (使用 el-ph 矩阵)
# ============================================================
def make_tdin_eph(calc_dir="calc/perturbo"):
    """
    自旋动力学: 使用 el-ph 散射矩阵计算自旋寿命
    el-ph 数据来自 ephmat 计算结果
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
    deph_scatter = 'elph'
    ! elph 来源的散射: 来自 ephmat 计算
    wmin = {energy_min}
    wmax = {energy_max}
    deph_targ = 0.2
/

&scbt
    calc_cond       = .true.
    calc_spin      = .true.
    calc_transport = .false.
    smear_type     = 'lorentzian'
    eta            = {eta}
/
"""
    with open(fname, "w") as f:
        f.write(content)
    print(f"[Perturbo] Generated: {fname}")
    return fname


if __name__ == "__main__":
    print("生成声子计算输入文件...")
    make_ph_input()
    make_q2r_input()
    make_matdyn_input()
    make_epw_input()
    # Perturbo el-ph 矩阵
    make_ephmat_input()
    # Perturbo 自旋动力学
    make_tdin_eph()
    print("\n完成！")
    print("流程顺序:")
    print("  1. ph.in   → q点声子 (DFPT)")
    print("  2. q2r.in → 原子间力常数 (IFC)")
    print("  3. matdyn.in → 声子色散")
    print("  4. ephmat.in (Perturbo) → el-ph 矩阵")
    print("  5. td.in    (Perturbo) → 自旋寿命")
