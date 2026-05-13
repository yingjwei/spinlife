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


def _detect_nk_per_band(k_raw):
    """从 k 值重复位置自动检测每条带点数 (两列堆叠格式用)."""
    n = len(k_raw)
    k0 = k_raw[0]
    for i in range(50, min(n, 5000)):
        if abs(k_raw[i] - k0) < 1e-14 and n % i == 0:
            return i
    return None


def load_band_data(filename, nk_expected=None):
    """
    读 wannier90_band.dat, 自动判断格式 (参考 fit_effective_mass.py).

    支持:
      1. 空行分隔: 每条带一个数据块, 块间有空行
      2. # Band-Index 头分隔
      3. 两列堆叠: (k, E) 按 (nbands, nk) 堆叠, 需要 nk_expected 或自动检测
      4. 多列格式: k  E1  E2 ...

    Parameters
    ----------
    nk_expected : int or None
        labelinfo 提供的每条带 k 点数 (两列堆叠格式必需).

    Returns
    -------
    k : (nk,) ndarray
    energies : (nk, nbands) ndarray
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

    total_rows = arr.shape[0]

    if arr.shape[1] >= 3:
        # 多列格式: k  E1  E2  ...
        return arr[:, 0], arr[:, 1:], total_rows, arr.shape[1] - 1

    if arr.shape[1] == 2:
        k_raw = arr[:, 0]

        # 两列堆叠: 用 labelinfo nk 拆出多条带
        if nk_expected and total_rows % nk_expected == 0:
            n_bands = total_rows // nk_expected
            print(f"  [两列堆叠: {n_bands} 条带 x {nk_expected} k点]")
            return k_raw[:nk_expected], arr[:, 1].reshape(n_bands, nk_expected).T, nk_expected, n_bands

        # 自动检测 nk
        auto_nk = _detect_nk_per_band(k_raw)
        if auto_nk and total_rows % auto_nk == 0:
            n_bands = total_rows // auto_nk
            print(f"  [两列堆叠: {n_bands} 条带 x {auto_nk} k点 (自动检测)]")
            return k_raw[:auto_nk], arr[:, 1].reshape(n_bands, auto_nk).T, auto_nk, n_bands

        # 回退: 单条带
        print(f"  [单条带, {total_rows} k点]")
        return k_raw, arr[:, 1:2], total_rows, 1

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

    Standard Wannier90 format: label  index(1-based)  k_distance  kx  ky  kz

    Returns
    -------
    labels : list of (k_index_0based, k_distance, label_str)
    """
    data = np.loadtxt(filepath, dtype=str)
    labels = []
    for row in data:
        label = row[0]
        idx = int(row[1]) - 1  # 1-based → 0-based
        kdist = float(row[2])
        labels.append((idx, kdist, label))
    return labels
