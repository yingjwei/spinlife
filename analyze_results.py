#!/usr/bin/env python3
"""
自旋寿命结果分析脚本 - W6CCl16
读取 Perturbo 输出，提取 τ_s, τ_φ 等关键数据
"""

import os
import re
import sys
import json
import argparse
import numpy as np


def parse_td_out(filepath="calc/perturbo/td.out"):
    """解析 Perturbo td.out 输出"""
    if not os.path.exists(filepath):
        print(f"[WARN] 文件不存在: {filepath}")
        return None

    with open(filepath, "r") as f:
        content = f.read()

    results = {}

    # 自旋寿命
    m = re.search(r"spin_lifetime\s*[=:]?\s*([\d\.e\-\+]+)\s*fs", content, re.IGNORECASE)
    if not m:
        m = re.search(r"tau[_\s]*spin\s*[=:]?\s*([\d\.e\-\+]+)\s*fs", content, re.IGNORECASE)
    if m:
        results["spin_lifetime_fs"] = float(m.group(1))

    # 退相干时间
    m = re.search(r"(?:dephasing|decoherence)[_\s]*time\s*[=:]?\s*([\d\.e\-\+]+)\s*fs", content, re.IGNORECASE)
    if not m:
        m = re.search(r"tau[_\s]*(?:phi|deph)\s*[=:]?\s*([\d\.e\-\+]+)\s*fs", content, re.IGNORECASE)
    if m:
        results["dephasing_time_fs"] = float(m.group(1))

    # 自旋扩散长度
    m = re.search(r"spin[_\s]*diffusion[_\s]*length\s*[=:]?\s*([\d\.e\-\+]+)\s*nm", content, re.IGNORECASE)
    if m:
        results["spin_diffusion_length_nm"] = float(m.group(1))

    # 费米能级
    m = re.search(r"fermi[_\s]*energy\s*[=:]?\s*([\d\.\-\+]+)\s*eV", content, re.IGNORECASE)
    if m:
        results["fermi_energy_eV"] = float(m.group(1))

    # 自旋霍尔电导
    m = re.search(r"spin[_\s]*hall[_\s]*conductance\s*[=:]?\s*([\d\.e\-\+]+)\s*(?:\(e|emu)", content, re.IGNORECASE)
    if m:
        results["spin_hall_conductance"] = float(m.group(1))

    # 温度
    m = re.search(r"temperature\s*[=:]?\s*([\d\.]+)\s*K", content, re.IGNORECASE)
    if m:
        results["temperature_K"] = float(m.group(1))

    return results


def parse_yml(filepath="calc/perturbo/td_spinlife.yml"):
    """解析 Perturbo YAML 输出"""
    if not os.path.exists(filepath):
        return None
    import yaml
    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"[WARN] YAML 解析失败: {e}")
        return None


def export_json(results, out="spinlifetime_results.json"):
    """导出为 JSON"""
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] 结果已保存: {out}")


def print_summary(results):
    """打印结果摘要"""
    if not results:
        print("[ERROR] 无法解析结果文件")
        print("请确认 Perturbo 已成功运行，输出文件存在:")
        print("  calc/perturbo/td.out")
        print("  calc/perturbo/td_spinlife.yml")
        return

    print("\n" + "="*50)
    print("  W6CCl16 自旋寿命计算结果摘要")
    print("="*50)

    keys = [
        ("温度",            "temperature_K",           "K"),
        ("自旋寿命 τ_s",    "spin_lifetime_fs",        "fs"),
        ("退相干时间 τ_φ",  "dephasing_time_fs",       "fs"),
        ("自旋扩散长度",    "spin_diffusion_length_nm","nm"),
        ("费米能级",        "fermi_energy_eV",         "eV"),
    ]

    for label, key, unit in keys:
        if key in results:
            print(f"  {label:<20s}: {results[key]:>10.4f}  [{unit}]")

    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 Perturbo 自旋寿命结果")
    parser.add_argument("--td-out",   default="calc/perturbo/td.out",
                       help="Perturbo td.out 文件路径")
    parser.add_argument("--td-yml",   default="calc/perturbo/td_spinlife.yml",
                       help="Perturbo td_spinlife.yml 文件路径")
    parser.add_argument("--json",     default="spinlifetime_results.json",
                       help="JSON 输出文件")
    args = parser.parse_args()

    results = parse_td_out(args.td_out)
    if results is None:
        results = parse_yml(args.td_yml)

    print_summary(results)
    if results:
        export_json(results, args.json)
