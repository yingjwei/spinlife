#!/usr/bin/env python3
"""
声子计算提交脚本 - W6CCl16
流程: ph.x (DFPT) → q2r.x → matdyn → EPW / Perturbo ephmat
"""

import os
import sys
import argparse
from params import *

PH_BIN  = os.environ.get("PH_BIN",  "$HOME/qe/bin/ph.x")
Q2R_BIN = os.environ.get("Q2R_BIN","$HOME/qe/bin/q2r.x")
MAT_BIN = os.environ.get("MAT_BIN","$HOME/qe/bin/matdyn.x")
EPW_BIN = os.environ.get("EPW_BIN","$HOME/epw/src/epw.x")


def run_ph(q_mesh, calc_dir="calc/phonon"):
    """ph.x: DFPT 声子计算"""
    os.makedirs(calc_dir, exist_ok=True)
    from qe_phonon_inputs import make_ph_input
    make_ph_input(q_mesh=q_mesh, calc_dir=calc_dir)

    script = f"""#!/bin/bash
#SBATCH --job-name={material}_ph
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={email}

cd $SLURM_SUBMIT_DIR/{calc_dir}
module load quantum-espresso/7.3
export OMP_NUM_THREADS=1
mpirun -n $SLOT_NUM $PH_BIN < ph.in > ph.out
"""
    job_file = os.path.join(calc_dir, "ph.sh")
    with open(job_file, "w") as f:
        f.write(script)
    os.chmod(job_file, 0o755)
    print(f"[SLURM] Submitting: {job_file}")
    import subprocess
    result = subprocess.run(["sbatch", job_file], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode


def run_q2r(calc_dir="calc/phonon"):
    """q2r.x: dynmat → IFC"""
    script = f"""#!/bin/bash
#SBATCH --job-name={material}_q2r
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --partition={partition}
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={email}

cd $SLURM_SUBMIT_DIR/{calc_dir}
module load quantum-espresso/7.3
$Q2R_BIN < q2r.in > q2r.out
"""
    job_file = os.path.join(calc_dir, "q2r.sh")
    with open(job_file, "w") as f:
        f.write(script)
    os.chmod(job_file, 0o755)
    print(f"[SLURM] Submitting: {job_file}")
    import subprocess
    result = subprocess.run(["sbatch", job_file], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode


def run_matdyn(calc_dir="calc/phonon"):
    """matdyn.x: 声子色散"""
    script = f"""#!/bin/bash
#SBATCH --job-name={material}_matdyn
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --partition={partition}
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={email}

cd $SLURM_SUBMIT_DIR/{calc_dir}
module load quantum-espresso/7.3
$MAT_BIN < matdyn.in > matdyn.out
"""
    job_file = os.path.join(calc_dir, "matdyn.sh")
    with open(job_file, "w") as f:
        f.write(script)
    os.chmod(job_file, 0o755)
    print(f"[SLURM] Submitting: {job_file}")
    import subprocess
    result = subprocess.run(["sbatch", job_file], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode


def run_epw(nk, nqf, calc_dir="calc/phonon"):
    """EPW: 电子-声子耦合矩阵元 (推荐)"""
    from qe_phonon_inputs import make_epw_input
    make_epw_input(nk1=nk, nk2=nk, nk3=1,
                   nqf1=nqf, nqf2=nqf, nqf3=1,
                   calc_dir=calc_dir)
    script = f"""#!/bin/bash
#SBATCH --job-name={material}_epw
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks*2}
#SBATCH --partition={partition}
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={email}

cd $SLURM_SUBMIT_DIR/{calc_dir}
module load quantum-espresso/7.3 epw
export OMP_NUM_THREADS=1
mpirun -n $SLOT_NUM $EPW_BIN < epw.in > epw.out
"""
    job_file = os.path.join(calc_dir, "epw.sh")
    with open(job_file, "w") as f:
        f.write(script)
    os.chmod(job_file, 0o755)
    print(f"[SLURM] Submitting: {job_file}")
    import subprocess
    result = subprocess.run(["sbatch", job_file], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode


def run_perturbo_eph(calc_dir="calc/perturbo"):
    """Perturbo ephmat + td"""
    from run_perturbo import run_perturbo, run_all as run_perturbo_all
    return run_perturbo_all(calc_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run phonon / el-ph calculations")
    parser.add_argument("step", choices=[
        "ph", "q2r", "matdyn", "epw",
        "perturbo-eph", "eph", "all-ph",
        "generate"
    ], help="计算步骤")
    parser.add_argument("--calc-dir",  default="calc/phonon", help="声子计算目录")
    parser.add_argument("--nk",        type=int, default=6,   help="EPW fine k 网格")
    parser.add_argument("--nqf",       type=int, default=3,   help="EPW fine q 网格")
    parser.add_argument("--q-mesh",    nargs=3, type=int, default=[3,3,1],
                        metavar=("N1","N2","N3"), help="声子 q 点网格")
    args = parser.parse_args()

    if args.step == "generate":
        from qe_phonon_inputs import (
            make_ph_input, make_q2r_input,
            make_matdyn_input, make_epw_input,
            make_ephmat_input, make_tdin_eph
        )
        make_ph_input(q_mesh=args.q_mesh, calc_dir=args.calc_dir)
        make_q2r_input(calc_dir=args.calc_dir)
        make_matdyn_input(calc_dir=args.calc_dir)
        make_epw_input(calc_dir=args.calc_dir)
        make_ephmat_input()
        make_tdin_eph()
        print("全部生成完成！")

    elif args.step == "ph":
        run_ph(q_mesh=args.q_mesh, calc_dir=args.calc_dir)

    elif args.step == "q2r":
        run_q2r(calc_dir=args.calc_dir)

    elif args.step == "matdyn":
        run_matdyn(calc_dir=args.calc_dir)

    elif args.step == "epw":
        run_epw(nk=args.nk, nqf=args.nqf, calc_dir=args.calc_dir)

    elif args.step in ("eph", "perturbo-eph"):
        run_perturbo_eph()

    elif args.step == "all-ph":
        # 生成输入
        from qe_phonon_inputs import (
            make_ph_input, make_q2r_input,
            make_matdyn_input, make_epw_input,
            make_ephmat_input, make_tdin_eph
        )
        make_ph_input(q_mesh=args.q_mesh, calc_dir=args.calc_dir)
        make_q2r_input(calc_dir=args.calc_dir)
        make_matdyn_input(calc_dir=args.calc_dir)
        make_epw_input(calc_dir=args.calc_dir)
        make_ephmat_input()
        make_tdin_eph()
        print("全部输入生成完成！")
