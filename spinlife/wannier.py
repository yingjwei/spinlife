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


def load_band_data(filename):
    """
    读 wannier90_band.dat, 自动判断格式 (参考 fit_effective_mass.py).

    支持三种格式:
      1. 空行分隔: 每条带一个数据块, 块间有空行
      2. # Band-Index 头分隔
      3. 纯两列堆叠: 所有带 (k, E) 直接拼接
      4. 多列格式: k  E1  E2 ...

    Returns
    -------
    k : (nk,) ndarray — k 坐标 (单位与文件一致, 一般是 Å⁻¹)
    energies : (nk, nbands) ndarray — 能带能量 (eV)
    nk : int
    nbands : int
    """
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')

    # 1) 空行分隔格式
    if '\n\n' in raw.strip():
        result = _parse_blank_separated(raw)
        if result is not None:
            k, E_all, nk_per = result
            return k, E_all, nk_per, E_all.shape[1]

    # 2) # 头分隔格式
    result = _parse_header_separated(raw)
    if result is not None:
        k, E_all, nk_per = result
        return k, E_all, nk_per, E_all.shape[1]

    # 3/4) 直接 loadtxt
    arr = np.loadtxt(filename)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.shape[1] >= 3:
        # 多列格式: k  E1  E2  ...
        return arr[:, 0], arr[:, 1:], arr.shape[0], arr.shape[1] - 1
    elif arr.shape[1] == 2:
        # 两列: k  E  (单带)
        return arr[:, 0], arr[:, 1:2], arr.shape[0], 1
    else:
        raise ValueError(f"无法解析: {arr.shape[1]} 列 (需要 ≥2 列)")


def _parse_blank_separated(raw_text):
    """解析空行分隔的 wannier90_band.dat 文本"""
    blocks = [b.strip() for b in raw_text.split('\n\n') if b.strip()]
    bands_list = []
    k_first = None
    for block in blocks:
        lines = block.split('\n')
        clean = [line.strip() for line in lines
                 if line.strip() and not line.strip().startswith('#')]
        if not clean:
            continue
        try:
            arr = np.loadtxt(clean)
        except Exception:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] < 2:
            continue
        k_this = arr[:, 0]
        e_this = arr[:, 1]
        if k_first is None:
            k_first = k_this
        elif not np.allclose(k_this, k_first, atol=1e-10):
            return None
        bands_list.append(e_this)
    if len(bands_list) < 1:
        return None
    return k_first, np.column_stack(bands_list), len(k_first)


def _parse_header_separated(raw_text):
    """解析 # Band-Index 类头分隔的格式"""
    lines = raw_text.split('\n')
    blocks = []
    cur = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith('#'):
            if cur:
                blocks.append('\n'.join(cur))
                cur = []
            continue
        cur.append(s)
    if cur:
        blocks.append('\n'.join(cur))
    bands_list = []
    k_first = None
    for block in blocks:
        try:
            arr = np.loadtxt(block.split('\n'))
        except Exception:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] < 2:
            continue
        k_this = arr[:, 0]
        e_this = arr[:, 1]
        if k_first is None:
            k_first = k_this
        elif len(k_this) != len(k_first) or not np.allclose(k_this, k_first, atol=1e-10):
            continue
        bands_list.append(e_this)
    if len(bands_list) < 2:
        return None
    return k_first, np.column_stack(bands_list), len(k_first)


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
