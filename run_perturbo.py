#!/usr/bin/env python3
"""
Perturbo 计算脚本 - W6CCl16 自旋寿命
"""

import os
import sys
import subprocess
import argparse
from params import *


PERTURBO_BIN = os.environ.get("PERTURBO_BIN", "$HOME/perturbo/bin")


def run_perturbo(itype, input_file, calc_dir):
    """运行 Perturbo"""
    cmd = f"{PERTURBO_BIN}/perturbo.x < {input_file} > {input_file.replace('.in', '.out')}"
    print(f"[Perturbo] Running itype={itype}: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=calc_dir,
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed: {result.stderr}")
        return 1
    print(f"[OK] {input_file} 完成")
    return 0


def run_qe2pert(calc_dir):
    """步骤 1: QE → Perturbo 数据转换"""
    return run_perturbo(0, "qe2pert.in", calc_dir)


def run_teout(calc_dir):
    """步骤 2: 电子-声子散射矩阵"""
    return run_perturbo(1, "teout.in", calc_dir)


def run_tdin(calc_dir):
    """步骤 3: 自旋动力学 / 自旋寿命"""
    return run_perturbo(4, "td.in", calc_dir)


def run_all(calc_dir):
    """完整 Perturbo 流程"""
    steps = [
        (0, "qe2pert.in"),
        (1, "teout.in"),
        (4, "td.in"),
    ]
    for itype, fname in steps:
        rc = run_perturbo(itype, fname, calc_dir)
        if rc != 0:
            print(f"[FATAL] itype={itype} failed")
            sys.exit(rc)
    print("\n✅ Perturbo 流程完成！")
    print("自旋寿命结果: td_spinlife.yml 或 td.out")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Perturbo calculations")
    parser.add_argument("step", choices=["qe2pert", "teout", "td", "all"],
                       help="计算步骤")
    parser.add_argument("--calc-dir", default="calc/perturbo",
                       help="Perturbo 计算目录")
    args = parser.parse_args()

    if args.step == "all":
        rc = run_all(args.calc_dir)
    else:
        steps = {"qe2pert": (0, "qe2pert.in"),
                 "teout":    (1, "teout.in"),
                 "td":       (4, "td.in")}
        itype, fname = steps[args.step]
        rc = run_perturbo(itype, fname, args.calc_dir)

    sys.exit(rc)
