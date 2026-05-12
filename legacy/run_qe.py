#!/usr/bin/env python3
"""
QE 计算提交脚本 - W6CCl16
支持 SLURM / PBS / 本地运行
"""

import os
import sys
import subprocess
import argparse
from params import *

# QE 可执行文件 (根据实际安装路径修改)
QE_BIN = os.environ.get("QE_BIN", "$HOME/qe/bin")  # pw.x 所在目录


def make_slurm_job(script_name, nodes=1, ntasks=24, partition="batch",
                   walltime="12:00:00", email="your@email.com", job_name=None):
    """生成 SLURM 脚本内容"""
    if job_name is None:
        job_name = f"{material}_{script_name}"
    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={email}

cd $SLURM_SUBMIT_DIR

module load quantum-espresso/7.3
export OMP_NUM_THREADS=1
export QE_BIN={QE_BIN}

mpirun -n $SLOT_NUM ${{QE_BIN}}/pw.x < {script_name}.in > {script_name}.out
"""
    return content


def run_local(cmd, calc_dir):
    """本地运行命令"""
    print(f"[LOCAL] Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=calc_dir,
                            capture_output=False, text=True)
    return result.returncode


def run_slurm(script_name, calc_dir, nodes=1, ntasks=24,
              partition="batch", walltime="12:00:00", email="your@email.com"):
    """提交 SLURM 任务"""
    job_content = make_slurm_job(
        script_name, nodes, ntasks, partition, walltime, email
    )
    job_file = os.path.join(calc_dir, f"{script_name}.sh")
    with open(job_file, "w") as f:
        f.write(job_content)
    os.chmod(job_file, 0o755)
    print(f"[SLURM] Submitting {job_file} ...")
    result = subprocess.run(["sbatch", job_file], cwd=calc_dir,
                            capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] Submit failed: {result.stderr}")
    return result.returncode


def run_step(step, calc_dir):
    """运行指定步骤"""
    steps = {
        "scf":     ("scf",     "pw.x < scf.in > scf.out",  False),
        "nscf":    ("nscf",    "pw.x < nscf.in > nscf.out", False),
        "wannier": ("wannier", "wannier90.x wannier90.win", False),
        "proj":    ("nscf",    "pw2w90dmi.x < pw2w90.in > pw2w90.out", False),
    }

    if step not in steps:
        print(f"[ERROR] Unknown step: {step}")
        print(f"Available: {list(steps.keys())}")
        return 1

    name, cmd, use_srun = steps[step]
    if queue_system == "slurm":
        return run_slurm(name, calc_dir, nodes, ntasks, partition, walltime, email)
    else:
        full_cmd = f"srun {cmd}" if use_srun else cmd
        return run_local(full_cmd, calc_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QE calculations")
    parser.add_argument("step", choices=["scf", "nscf", "wannier", "proj", "all"],
                        help="计算步骤")
    parser.add_argument("--calc-dir", default="calc", help="计算目录")
    args = parser.parse_args()

    if args.step == "all":
        for step in ["scf", "nscf", "wannier", "proj"]:
            print(f"\n{'='*50}")
            print(f"Step: {step}")
            print(f"{'='*50}")
            rc = run_step(step, args.calc_dir)
            if rc != 0:
                print(f"[FATAL] Step {step} failed with code {rc}")
                sys.exit(rc)
    else:
        rc = run_step(args.step, args.calc_dir)
        sys.exit(rc)

    print("\n✅ 计算流程完成！")
