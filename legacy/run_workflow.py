#!/usr/bin/env python3
"""
W6CCl16 自旋寿命计算完整工作流
一键运行: QE结构优化 → SCF → NSCF → Wannier90 → Perturbo
"""

import os
import sys
import time

# 依次导入各模块
from qe_inputs import make_scf_input, make_nscf_input, make_wannier90_input
from perturbo_inputs import make_tdin_input, make_qe2pert_input
from run_qe import run_step as run_qe_step
from run_perturbo import run_all as run_perturbo_all

CALC_DIR = "calc"


def step(name, func, *args, **kwargs):
    print(f"\n{'#'*60}")
    print(f"  Step: {name}")
    print(f"{'#'*60}")
    t0 = time.time()
    rc = func(*args, **kwargs)
    elapsed = time.time() - t0
    if rc == 0:
        print(f"  ✅ {name} 完成 ({elapsed:.1f}s)")
    else:
        print(f"  ❌ {name} 失败 (exit code: {rc})")
        sys.exit(rc)


def main():
    print("="*60)
    print("  W6CCl16 自旋寿命计算工作流")
    print("="*60)

    # Step 1: 生成 QE 输入文件
    step("生成 QE + Wannier90 输入文件",
         lambda: (make_scf_input(), make_nscf_input(), make_wannier90_input()))

    # Step 2: SCF 自洽计算
    step("SCF 自洽计算", run_qe_step, "scf", CALC_DIR)

    # Step 3: NSCF 高密 k 点
    step("NSCF 高密 k 点计算", run_qe_step, "nscf", CALC_DIR)

    # Step 4: Wannier90 最大局域化 Wannier 函数
    step("Wannier90 MLWF", run_qe_step, "wannier", CALC_DIR)

    # Step 5: 生成 Perturbo 输入
    step("生成 Perturbo 输入文件",
         lambda: (make_qe2pert_input(), make_tdin_input()))

    # Step 6: Perturbo 自旋动力学
    step("Perturbo 自旋动力学", run_perturbo_all, f"{CALC_DIR}/perturbo")

    print("\n" + "="*60)
    print("  🎉 全部完成！")
    print("  下一步: python analyze_results.py")
    print("="*60)


if __name__ == "__main__":
    main()
