# -*- coding: utf-8 -*-
"""
spinlife.wannier — Wannier90 能带文件解析与 k 空间转换
"""
import numpy as np


def read_bands(filepath):
    """
    读 wannier90_band.dat

    Format: index  kx  ky  kz  E1  E2 ... Enbands

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

    kpts = data[:, 1:4]
    energies = data[:, 4:]
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
