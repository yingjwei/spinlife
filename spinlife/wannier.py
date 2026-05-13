# -*- coding: utf-8 -*-
"""
spinlife.wannier — Wannier90 能带文件解析与 k 空间转换
"""
import numpy as np


def read_bands(filepath):
    """
    读 wannier90_band.dat

    支持两种格式:
      A) index  kx  ky  kz  E1  E2 ... Enbands  (有索引列)
      B) kx  ky  kz  E1  E2 ... Enbands          (无索引列)

    Returns
    -------
    k_frac : (nk,) ndarray — cumulative k-path distance in fractional coords
    energies : (nk, nbands) ndarray — band energies in eV
    nk : int
    nbands : int
    """
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]

    if ncols == 4:
        # 格式 B: kx ky kz E (单带, 无索引)
        kpts = data[:, :3]
        energies = data[:, 3:4]
    elif ncols > 4:
        # 判断第一列是否为整数索引 (格式 A)
        first_col = data[:, 0]
        if np.all(np.abs(first_col - np.round(first_col)) < 1e-6) and first_col[0] < 10:
            # 格式 A: index kx ky kz E1 E2 ...
            kpts = data[:, 1:4]
            energies = data[:, 4:]
        else:
            # 格式 B: kx ky kz E1 E2 ...
            kpts = data[:, :3]
            energies = data[:, 3:]
    else:
        raise ValueError(
            f"无法解析能带文件: {ncols} 列 (需要 ≥4 列)\n"
            f"期望格式: index kx ky kz E1 E2 ...")

    nk, nbands = energies.shape
    k_frac = np.zeros(nk)
    for i in range(1, nk):
        k_frac[i] = k_frac[i - 1] + np.sqrt(np.sum((kpts[i] - kpts[i - 1]) ** 2))

    return k_frac, energies, nk, nbands


def k_to_reciprocal(k_frac, lattice_a):
    """
    将 fractional k 距离转换为 Å⁻¹.

    k (Å⁻¹) = k_frac × 2π / a (Å)
    """
    return k_frac * 2.0 * np.pi / lattice_a


def find_extremum(k, energy, mode='max'):
    """找到能带极值位置的 k₀ 和索引."""
    idx = np.argmax(energy) if mode == 'max' else np.argmin(energy)
    return k[idx], idx


def band_slice(k, energy, k0, k_range):
    """截取 k₀ 附近 ±k_range 范围内的数据."""
    mask = np.abs(k - k0) <= k_range
    return k[mask], energy[mask]


def read_labelinfo(filepath):
    """
    读 wannier90_band.labelinfo.dat — 高对称点标签.

    Format: k_index  k_distance  label

    Returns
    -------
    labels : list of (k_index, k_distance, label_str)
        k_index is 0-based, k_distance is cumulative fractional k.
    """
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    kidx = int(parts[0]) - 1  # 1-based → 0-based
                    kdist = float(parts[1])
                    label = parts[2]
                    labels.append((kidx, kdist, label))
                except ValueError:
                    continue
    return labels
