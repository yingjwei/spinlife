# QE 输入文件生成器 - W6CCl16

import os
from params import *

# ============================================================
# QE SCF 输入模板
# ============================================================
def make_scf_input(calc_dir="calc/scf"):
    os.makedirs(calc_dir, exist_ok=True)
    content = f"""&control
    calculation = 'scf'
    prefix = '{material}'
    outdir = './out/'
    pseudo_dir = './pseudo/'
    tstress = .true.
    tprnfor = .true.
/

&system
    ibrav = 0
    nat = 26         ! W6 C8 Cl12 = 26 atoms
    ntyp = 3
    ecutwfc = {ecutwfc}
    ecutrho = {ecutrho}
    occupations = 'smearing'
    smearing = 'mp'
    degauss = 0.01
/

&electrons
    diagonalization = 'davidson'
    mixing_beta = 0.7
    conv_thr = 1.0d-8
/

ATOMIC_SPECIES
  W   183.84  W.pbe-{pp_family}.UPF
  C   12.011  C.pbe-{pp_family}.UPF
  Cl  35.453  Cl.pbe-{pp_family}.UPF

ATOMIC_POSITIONS (angstrom)
! W6CCl16 atomic positions
  W   3.333  1.924  12.500
  W   6.667  3.848  12.500
  W   3.333  5.772  12.500
  W   6.667  7.696  12.500
  W   3.333  9.620  12.500
  W   6.667 11.544  12.500
  C   2.000  1.154  11.920
  C   4.000  2.308  11.920
  C   6.000  3.462  11.920
  C   8.000  4.616  11.920
  C   2.000  6.540  11.920
  C   4.000  7.694  11.920
  C   6.000  8.848  11.920
  C   8.000 10.002  11.920
  Cl  1.000  0.577  11.250
  Cl  3.000  1.731  11.250
  Cl  5.000  2.885  11.250
  Cl  7.000  4.039  11.250
  Cl  9.000  5.193  11.250
  Cl  2.000  6.963  11.250
  Cl  4.000  8.117  11.250
  Cl  6.000  9.271  11.250
  Cl  8.000 10.425  11.250
  Cl 10.000 11.579  11.250
  Cl  1.000 13.077  11.250
  Cl  3.000 14.231  11.250

CELL_PARAMETERS (angstrom)
  {a_lat:.3f}  0.000  0.000
  {a_lat/2:.3f}  {b_lat*0.866:.3f}  0.000
  0.000  0.000  {c_lat:.3f}

K_POINTS automatic
  {kpts_scf[0]} {kpts_scf[1]} {kpts_scf[2]}  0 0 0
"""
    with open(f"{calc_dir}/scf.in", "w") as f:
        f.write(content)
    return f"{calc_dir}/scf.in"


# ============================================================
# QE NSCF 输入模板 (高密 k 点)
# ============================================================
def make_nscf_input(calc_dir="calc/nscf"):
    os.makedirs(calc_dir, exist_ok=True)
    content = f"""&control
    calculation = 'nscf'
    prefix = '{material}'
    outdir = '../scf/out/'
    pseudo_dir = '../pseudo/'
/

&system
    ibrav = 0
    nat = 26
    ntyp = 3
    ecutwfc = {ecutwfc}
    ecutrho = {ecutrho}
    occupations = 'smearing'
    smearing = 'mp'
    degauss = 0.01
/

&electrons
    diagonalization = 'davidson'
    mixing_beta = 0.7
    conv_thr = 1.0d-8
/

ATOMIC_SPECIES
  W   183.84  W.pbe-{pp_family}.UPF
  C   12.011  C.pbe-{pp_family}.UPF
  Cl  35.453  Cl.pbe-{pp_family}.UPF

ATOMIC_POSITIONS (angstrom)
  W   3.333  1.924  12.500
  W   6.667  3.848  12.500
  W   3.333  5.772  12.500
  W   6.667  7.696  12.500
  W   3.333  9.620  12.500
  W   6.667 11.544  12.500
  C   2.000  1.154  11.920
  C   4.000  2.308  11.920
  C   6.000  3.462  11.920
  C   8.000  4.616  11.920
  C   2.000  6.540  11.920
  C   4.000  7.694  11.920
  C   6.000  8.848  11.920
  C   8.000 10.002  11.920
  Cl  1.000  0.577  11.250
  Cl  3.000  1.731  11.250
  Cl  5.000  2.885  11.250
  Cl  7.000  4.039  11.250
  Cl  9.000  5.193  11.250
  Cl  2.000  6.963  11.250
  Cl  4.000  8.117  11.250
  Cl  6.000  9.271  11.250
  Cl  8.000 10.425  11.250
  Cl 10.000 11.579  11.250
  Cl  1.000 13.077  11.250
  Cl  3.000 14.231  11.250

CELL_PARAMETERS (angstrom)
  {a_lat:.3f}  0.000  0.000
  {a_lat/2:.3f}  {b_lat*0.866:.3f}  0.000
  0.000  0.000  {c_lat:.3f}

K_POINTS automatic
  {kpts_nscf[0]} {kpts_nscf[1]} {kpts_nscf[2]}  0 0 0
"""
    with open(f"{calc_dir}/nscf.in", "w") as f:
        f.write(content)
    return f"{calc_dir}/nscf.in"


# ============================================================
# Wannier90 输入模板
# ============================================================
def make_wannier90_input(calc_dir="calc/wannier"):
    os.makedirs(calc_dir, exist_ok=True)
    content = f"""num_wann = {num_wann}
num_iter = 1000
iprint = 1

guiding_centres = .true.
spinors = .true.

dis_froz_max = {dis_froz_max}
dis_win_max = {dis_win_max}
dis_num_iter = 200
dis_mix_ratio = 0.5

begin kpoint_path
  G  0.0  0.0  0.0   M  0.5  0.0  0.0
  M  0.5  0.0  0.0   K  0.333  0.333  0.0
  K  0.333  0.333  0.0  G  0.0  0.0  0.0
end kpoint_path

begin atoms_frac
  W   0.333  0.667  0.500
end atoms_frac

begin projections
  W: dw
  C : px, py, pz
  Cl : px, py, pz
end projections

mp_grid = {kpts_wann[0]}, {kpts_wann[1]}, {kpts_wann[2]}

begin kpoints
"""
    # 生成 k 点
    for i in range(kpts_wann[0]):
        for j in range(kpts_wann[1]):
            for k in range(kpts_wann[2]):
                content += f"  {i/kpts_wann[0]:.6f}  {j/kpts_wann[1]:.6f}  {k/kpts_wann[2]:.6f}\n"

    content += "end kpoints\n"
    with open(f"{calc_dir}/wannier90.win", "w") as f:
        f.write(content)
    return f"{calc_dir}/wannier90.win"


if __name__ == "__main__":
    print("生成 QE + Wannier90 输入文件...")
    make_scf_input()
    make_nscf_input()
    make_wannier90_input()
    print("完成！")
