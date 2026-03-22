# QE 输入文件生成器 - W6CCl16 (真实结构)
# 原子坐标/晶格均来自 wannier90.win

import os
from params import *

# ============================================================
# 原子列表 (cartesian, from wannier90.win)
# 格式: (元素, x, y, z)
# ============================================================
ATOMS = [
    # C (12 atoms)
    ("C",  0.0000000,  5.9773970, 14.9972048),
    ("C",  6.4986385,  0.8517268, 11.5420967),
    ("C",  3.2647194,  9.3415197, 14.4057503),
    ("C",  9.7633580,  4.2158496, 12.1335512),
    ("C",  4.8504085,  7.7908358, 11.8522434),
    ("C", 11.4754019,  6.9835595, 12.4911665),
    ("C",  8.0817698, 10.1160158,  9.0708896),
    ("C",  8.1350616,  3.4815273,  9.4940136),
    ("C",  9.6014269,  0.8460602, 11.5221718),
    ("C",  3.2339191,  4.2158496, 12.1335512),
    ("C",  8.1468686,  7.7908358, 11.8522434),
    ("C",  1.5218752,  6.9835595, 12.4911665),
    # Cl (21 atoms)
    ("Cl", 1.5831312,  4.9903456, 17.4684119),
    ("Cl", 1.6364231,  8.6071975, 17.0452880),
    ("Cl", 3.1027884,  5.9717304, 15.0171298),
    ("Cl", 9.7325577,  9.3415197, 14.4057503),
    ("Cl", 1.6482301,  2.6651657, 14.6870581),
    ("Cl", 8.0205137,  1.8578894, 14.0481351),
    ("Cl", 4.9155073, 10.1160158,  9.0708896),
    ("Cl", 4.8622155,  3.4815273,  9.4940136),
    ("Cl", 3.3958502,  0.8460602, 11.5221718),
    ("Cl", 8.1350616,  3.4815273,  9.4940136),
    ("Cl", 8.0817698, 10.1160158,  9.0708896),
    ("Cl",11.4141458,  4.9903456, 17.4684119),
    ("Cl",11.3608540,  8.6071975, 17.0452880),
    ("Cl", 9.8944887,  5.9717304, 15.0171298),
    ("Cl", 3.2647194,  9.3415197, 14.4057503),
    ("Cl",11.3490470,  2.6651657, 14.6870581),
    ("Cl", 4.9767634,  1.8578894, 14.0481351),
    ("Cl", 0.0000000,  2.1648266, 17.4461311),
    ("Cl", 0.0000000,  9.6379248, 14.3966443),
    ("Cl", 0.0000000,  7.0537076, 19.3883152),
    ("Cl", 6.4986385,  7.2904968,  9.0931705),
    ("Cl", 6.4986385,  4.5122547, 12.1426573),
    ("Cl", 6.4986385,  1.9280375,  7.1509864),
    ("Cl", 6.4986385,  8.8060069, 14.4696178),
    ("Cl", 0.0000000,  3.6803367, 12.0696837),
    # W (12 atoms)
    ("W",  1.3045012,  7.6706007, 14.8543560),
    ("W",  1.3074912,  4.8371057, 13.7343946),
    ("W",  5.1941374,  2.5449305, 11.6849456),
    ("W",  5.1911473,  9.9627759, 12.8049069),
    ("W",  7.8031397,  2.5449305, 11.6849456),
    ("W",  7.8061298,  9.9627759, 12.8049069),
    ("W", 11.6927759,  7.6706007, 14.8543560),
    ("W", 11.6897858,  4.8371057, 13.7343946),
    ("W",  0.0000000,  6.8260792, 17.0232223),
    ("W",  0.0000000,  3.9722982, 15.8905706),
    ("W",  6.4986385,  1.7004090,  9.5160793),
    ("W",  6.4986385,  9.0979684, 10.6487310),
]

# 赝势文件名 (!!! 根据你的 pseudo_dir 中的实际文件名修改 !!!)
PP_W  = "W.$pp_family.UPF"
PP_C  = "C.$pp_family.UPF"
PP_Cl = "Cl.$pp_family.UPF"


def _atoms_str():
    """生成 QE ATOMIC_POSITIONS 和 ATOMIC_SPECIES 行"""
    lines = []
    for elem, x, y, z in ATOMS:
        lines.append(f"  {elem:2s}  {x:12.7f}  {y:12.7f}  {z:12.7f}")
    return "\n".join(lines)


def _cell_params():
    return f"""CELL_PARAMETERS (angstrom)
 {a_lat:.7f}   0.0000000   0.0000000
  0.0000000  {b_lat:.7f}   0.0000000
  0.0000000   0.0000000  {c_lat:.7f}"""


def make_scf_input(calc_dir="calc/scf"):
    os.makedirs(calc_dir, exist_ok=True)
    content = f"""&control
    calculation = 'scf'
    prefix = '{material}'
    outdir = './out/'
    pseudo_dir = './pseudo/'
    tstress = .true.
    tprnfor = .true.
    wf_collect = .true.
/

&system
    ibrav = 0
    nat = {natom}
    ntyp = {ntyp}
    ecutwfc = {ecutwfc}
    ecutrho = {ecutrho}
    occupations = 'smearing'
    smearing = 'mp'
    degauss = 0.01
    lspinorb = .true.
    noncolin = .true.
/

&electrons
    diagonalization = 'davidson'
    mixing_beta = 0.7
    conv_thr = 1.0d-8
/

ATOMIC_SPECIES
  W  183.84  {PP_W}
  C   12.011  {PP_C}
  Cl  35.453  {PP_Cl}

ATOMIC_POSITIONS (angstrom)
{_atoms_str()}

{_cell_params()}

K_POINTS automatic
  {kpts_scf[0]} {kpts_scf[1]} {kpts_scf[2]}  0 0 0
"""
    path = f"{calc_dir}/scf.in"
    with open(path, "w") as f:
        f.write(content)
    print(f"[QE] Generated: {path}")
    return path


def make_nscf_input(calc_dir="calc/nscf"):
    os.makedirs(calc_dir, exist_ok=True)
    content = f"""&control
    calculation = 'nscf'
    prefix = '{material}'
    outdir = '../scf/out/'
    pseudo_dir = '../pseudo/'
    wf_collect = .true.
/

&system
    ibrav = 0
    nat = {natom}
    ntyp = {ntyp}
    ecutwfc = {ecutwfc}
    ecutrho = {ecutrho}
    occupations = 'smearing'
    smearing = 'mp'
    degauss = 0.01
    lspinorb = .true.
    noncolin = .true.
/

&electrons
    diagonalization = 'davidson'
    mixing_beta = 0.7
    conv_thr = 1.0d-8
/

ATOMIC_SPECIES
  W  183.84  {PP_W}
  C   12.011  {PP_C}
  Cl  35.453  {PP_Cl}

ATOMIC_POSITIONS (angstrom)
{_atoms_str()}

{_cell_params()}

K_POINTS automatic
  {kpts_nscf[0]} {kpts_nscf[1]} {kpts_nscf[2]}  0 0 0
"""
    path = f"{calc_dir}/nscf.in"
    with open(path, "w") as f:
        f.write(content)
    print(f"[QE] Generated: {path}")
    return path


def make_wannier90_input(calc_dir="calc/wannier"):
    os.makedirs(calc_dir, exist_ok=True)

    # 生成 k 点
    kpt_lines = []
    for i in range(kpts_wann[0]):
        for j in range(kpts_wann[1]):
            for k in range(kpts_wann[2]):
                kpt_lines.append(f"  {i/kpts_wann[0]:.9f}  {j/kpts_wann[1]:.9f}  {k/kpts_wann[2]:.9f}")

    content = f"""num_bands = {num_bands}
num_wann = {num_wann}
use_bloch_phases = {use_bloch_phases}
spinors = {spinors}

dis_num_iter = {dis_num_iter}
num_iter = {num_iter}
iprint = {3}
conv_tol = {conv_tol}

dis_win_min = {dis_win_min}
dis_win_max = {dis_win_max}

dis_froz_min = {dis_froz_min}
dis_froz_max = {dis_froz_max}

guiding_centres = .true.
write_tb = .true.

begin projections
C : s; px; py; pz
Cl : s; px; py; pz
W : s; px; py; pz; dxy; dyz; dz2; dxz; dx2-y2
end projections

mp_grid = {kpts_wann[0]}, {kpts_wann[1]}, {kpts_wann[2]}

begin kpoints
{chr(10).join(kpt_lines)}
end kpoints

begin unit_cell_cart
 {a_lat:.7f}  0.0000000  0.0000000
  0.0000000  {b_lat:.7f}  0.0000000
  0.0000000  0.0000000  {c_lat:.7f}
end unit_cell_cart

begin atoms_cart
{chr(10).join(f'{e}  {x:.7f}  {y:.7f}  {z:.7f}' for e,x,y,z in ATOMS)}
end atoms_cart
"""
    path = f"{calc_dir}/wannier90.win"
    with open(path, "w") as f:
        f.write(content)
    print(f"[W90] Generated: {path}")
    return path


if __name__ == "__main__":
    print("生成 QE + Wannier90 输入文件 (真实结构)...")
    make_scf_input()
    make_nscf_input()
    make_wannier90_input()
    print("完成！")
