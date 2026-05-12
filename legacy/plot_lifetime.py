#!/usr/bin/env python3
"""
自旋寿命绘图脚本 - W6CCl16
绘制温度/能量依赖的自旋寿命曲线
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_energy_dependence(data_file="spinlifetime_results.json"):
    """绘制自旋寿命-能量曲线"""
    if not os.path.exists(data_file):
        print(f"[WARN] 数据文件不存在: {data_file}")
        return

    import json
    with open(data_file) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    energies = data.get("energy_eV", [])
    tau_s    = data.get("tau_s_fs", [])
    tau_phi  = data.get("tau_phi_fs", [])

    if energies and tau_s:
        ax.plot(energies, tau_s, "o-", color="steelblue",
                linewidth=2, markersize=6, label=r"$\tau_s$ (spin lifetime)")
    if energies and tau_phi:
        ax.plot(energies, tau_phi, "s--", color="coral",
                linewidth=2, markersize=5, label=r"$\tau_\phi$ (dephasing)")

    ax.set_xlabel("Energy relative to CBM (eV)", fontsize=12)
    ax.set_ylabel(r"Spin lifetime $\tau_s$ (fs)", fontsize=12)
    ax.set_title("W6CCl16 Spin Lifetime vs Energy", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    out = "spinlifetime_energy.png"
    fig.savefig(out, dpi=150)
    print(f"[OK] 保存: {out}")


def plot_temperature_dependence(temps, tau_s, out="spinlifetime_T.png"):
    """绘制自旋寿命-温度曲线"""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(temps, tau_s, "o-", color="steelblue", linewidth=2, markersize=7)
    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel(r"Spin lifetime $\tau_s$ (fs)", fontsize=12)
    ax.set_title("W6CCl16 Spin Lifetime vs Temperature", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[OK] 保存: {out}")


def demo_plot():
    """无数据时生成示例图"""
    energies = np.linspace(-2, 2, 50)
    tau_s    = 100 * np.exp(-(energies ** 2) / 0.5)
    tau_phi  = 50  * np.exp(-(energies ** 2) / 0.8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(energies, tau_s, "o-", color="steelblue", linewidth=2)
    axes[0].set_xlabel("E - CBM (eV)")
    axes[0].set_ylabel(r"$\tau_s$ (fs)")
    axes[0].set_title("W6CCl16: 自旋寿命 vs 能量 (示例)")
    axes[0].grid(True, alpha=0.3)

    temps = np.array([50, 100, 150, 200, 250, 300])
    tau_T = 500 * (300 / temps) ** 1.5
    axes[1].plot(temps, tau_T, "s-", color="coral", linewidth=2)
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel(r"$\tau_s$ (fs)")
    axes[1].set_title("W6CCl16: 自旋寿命 vs 温度 (示例)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    fig.suptitle("W6CCl16 自旋寿命计算结果 [示例图]", fontsize=14)
    fig.tight_layout()
    out = "spinlifetime_demo.png"
    fig.savefig(out, dpi=150)
    print(f"[OK] 示例图已保存: {out}")
    print("  (替换数据后重新运行 plot_lifetime.py 即可)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制自旋寿命图")
    parser.add_argument("--data",   default="spinlifetime_results.json",
                       help="结果 JSON 文件")
    parser.add_argument("--temps",  nargs="+", type=float,
                       help="温度列表 (K), e.g. --temps 100 200 300")
    parser.add_argument("--tau",    nargs="+", type=float,
                       help="对应的自旋寿命列表 (fs)")
    args = parser.parse_args()

    if args.temps and args.tau:
        plot_temperature_dependence(args.temps, args.tau)
    elif os.path.exists(args.data):
        plot_energy_dependence(args.data)
    else:
        print("[INFO] 无数据文件，生成示例图...")
        demo_plot()
