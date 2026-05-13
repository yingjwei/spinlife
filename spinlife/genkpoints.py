# -*- coding: utf-8 -*-
"""
生成 VASP KPOINTS 文件: 在指定 k 点附近生成密集网格 (列表格式).

典型场景:
  做完有效质量拟合后, 在 VBM/CBM 极值点附近做更精细的 VASP 计算.

用法:
  python -m spinlife.genkpoints                    交互模式
  python -m spinlife.genkpoints kx ky kz           指定中心点

输出: 当前目录下的 KPOINTS 文件 (同名文件会被覆盖, 请注意备份).
"""
import sys
import os
import numpy as np


def generate_grid(center, n_div, k_range=0.05, dim=2):
    """
    生成围绕中心点的均匀 k 点网格.

    Parameters
    ----------
    center : (3,) array-like — 中心 k 点 (分数坐标)
    n_div : int — 每个方向的网格点数
    k_range : float — 半宽 (分数坐标)
    dim : int — 2 或 3

    Returns
    -------
    kpoints : (N, 4) ndarray — kx ky kz weight
    """
    grid_1d = np.linspace(-k_range, k_range, n_div)
    if dim == 2:
        kx, ky = np.meshgrid(center[0] + grid_1d, center[1] + grid_1d)
        kpoints = np.column_stack([
            kx.ravel(), ky.ravel(),
            np.full(kx.size, center[2]),
            np.ones(kx.size, dtype=int),
        ])
    else:
        kx, ky, kz = np.meshgrid(
            center[0] + grid_1d,
            center[1] + grid_1d,
            center[2] + grid_1d,
        )
        kpoints = np.column_stack([
            kx.ravel(), ky.ravel(), kz.ravel(),
            np.ones(kx.size, dtype=int),
        ])
    return kpoints


def write_kpoints(filename, kpoints, comment="spinlife generated"):
    """写入 VASP KPOINTS (列表格式)."""
    nk = len(kpoints)
    with open(filename, 'w') as f:
        f.write(f"{comment}\n")
        f.write(f"  0\n")            # 0 = custom k-points
        f.write(f"  Reciprocal\n")
        f.write(f"  {nk}\n")
        for kp in kpoints:
            f.write(f"  {kp[0]:.10f}  {kp[1]:.10f}  {kp[2]:.10f}  {kp[3]:.0f}\n")
    print(f"\n  [KPOINTS -> {os.path.abspath(filename)}]  ({nk} k-points)")
    print(f"  中心: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
    print(f"  范围: ±{k_range}  ({dim}D, {n_div}点/方向)")
    print(f"  (注意: 同名文件会被覆盖. 若 KPOINTS 已有重要内容请先备份.)")
    return nk


def main():
    print("=" * 60)
    print("  KPOINTS 生成器 — 在指定 k 点附近生成密集网格")
    print("=" * 60)
    print()

    # 中心点
    if len(sys.argv) >= 4:
        center = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]
    else:
        center = [
            float(input("  Center kx (frac): ") or 0),
            float(input("  Center ky (frac): ") or 0),
            float(input("  Center kz (frac): ") or 0),
        ]

    # 范围
    global k_range, n_div, dim, out
    k_range = float(input("  Range (±, frac coords) [0.05]: ") or 0.05)
    n_div = int(input("  Divisions per direction [10]: ") or 10)
    dim = int(input("  Dimension (2/3) [2]: ") or 2)
    out = input("  Output filename [KPOINTS]: ").strip() or "KPOINTS"

    kpoints = generate_grid(center, n_div, k_range, dim)
    write_kpoints(out, kpoints, comment=f"K-mesh at ({center[0]:.4f},{center[1]:.4f},{center[2]:.4f})")


if __name__ == '__main__':
    main()
