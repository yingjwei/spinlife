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


def generate_grid(center, k_range, nx, ny, nz=1):
    """
    生成围绕中心点的均匀 k 点网格 (各方向密度可单独指定).

    Parameters
    ----------
    center : (3,) array-like — 中心 k 点 (分数坐标)
    k_range : float — 半宽 (分数坐标)
    nx, ny, nz : int — 各方向网格点数 (nz=1 时生成 2D 网格)

    Returns
    -------
    kpoints : (N, 4) ndarray — kx ky kz weight
    """
    gx = np.linspace(-k_range, k_range, nx)
    gy = np.linspace(-k_range, k_range, ny)
    if nz <= 1:
        kx, ky = np.meshgrid(center[0] + gx, center[1] + gy)
        kpoints = np.column_stack([
            kx.ravel(), ky.ravel(),
            np.full(kx.size, center[2]),
            np.ones(kx.size, dtype=int),
        ])
    else:
        gz = np.linspace(-k_range, k_range, nz)
        kx, ky, kz = np.meshgrid(center[0] + gx, center[1] + gy, center[2] + gz)
        kpoints = np.column_stack([
            kx.ravel(), ky.ravel(), kz.ravel(),
            np.ones(kx.size, dtype=int),
        ])
    return kpoints


def write_kpoints(filename, kpoints, comment="spinlife generated",
                  center=None, k_range=None, nx=None, ny=None, nz=None):
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
    if center is not None:
        print(f"  中心: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
    if k_range is not None:
        nz_str = f" x {nz}" if nz and nz > 1 else ""
        print(f"  网格: {nx}×{ny}{nz_str}, 范围 ±{k_range}")
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

    k_range = float(input("  Range (±, frac coords) [0.05]: ") or 0.05)
    nx = int(input("  kx 方向点数 [5]: ") or 5)
    ny = int(input("  ky 方向点数 [5]: ") or 5)
    nz = int(input("  kz 方向点数 [1]: ") or 1)
    out = input("  Output filename [KPOINTS]: ").strip() or "KPOINTS"

    kpoints = generate_grid(center, k_range, nx, ny, nz)
    write_kpoints(out, kpoints,
                  comment=f"K-mesh at ({center[0]:.4f},{center[1]:.4f},{center[2]:.4f})",
                  center=center, k_range=k_range, nx=nx, ny=ny, nz=nz)


if __name__ == '__main__':
    main()
