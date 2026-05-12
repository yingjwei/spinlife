# -*- coding: utf-8 -*-
"""
PROCAR 解析器 — 非共线自旋 (LSORBIT) 模式下读取电荷+自旋密度

PROCAR with LSORBIT / LNONCOLLINEAR 的每一条带包含 4 个数据块:
  块 1 (电荷):  tot = ρ₀  (电荷密度, 恒正)
  块 2 (磁化 x): tot = m_x → ⟨σ_x⟩ = m_x / ρ₀
  块 3 (磁化 y): tot = m_y → ⟨σ_y⟩ = m_y / ρ₀
  块 4 (磁化 z): tot = m_z → ⟨σ_z⟩ = m_z / ρ₀

自旋密度矩阵: ρ = ρ₀I + m_xσ_x + m_yσ_y + m_zσ_z
"""

import numpy as np
import re


class PROCAR:
    """
    PROCAR 数据容器

    属性:
        nk:        k 点数
        nbands:    能带数
        nions:     离子数
        kpoints:   [(kx, ky, kz, weight), ...]
        bands:     {(ik, ib): {energy, occ, rho, sx, sy, sz, mx, my, mz}}
    """

    def __init__(self, filename):
        self.filename = filename
        self.nk = 0
        self.nbands = 0
        self.nions = 0
        self.kpoints = []
        self.bands = {}

        self._parse()

    def _parse(self):
        """
        逐行扫描 PROCAR 文件

        PROCAR 文件结构:
        ┌─────────────────────────────────────────────────────┐
        │ Line   1: PROCAR lm decomposed                      │
        │ Line   2: # of k-points: Nk  # of bands: Nb  # ions: Ni │
        │ Line   3: (blank)                                   │
        │ Line   4: k-point  ik:  kx ky kz  weight = w       │
        │ Line   5: (blank)                                   │
        │ Line   6: band  ib  # energy  E  # occ.  occ       │
        │ Line   7: (blank)                                   │
        │ Line   8: ion   s   py   pz   px  dxy  dyz  dz2  dxz  x2-y2  tot │
        │ Lines 9..9+Ni-1: ion data (电荷块)                 │
        │ Line   9+Ni: tot  ...   (电荷块 tot)                 │
        │ Lines 10+Ni..10+2Ni-1: ion data (σ_x 块)           │
        │ Line   10+2Ni: tot  ...  (σ_x 块 tot)              │
        │ Lines 11+2Ni..11+3Ni-1: ion data (σ_y 块)          │
        │ Line   11+3Ni: tot  ...  (σ_y 块 tot)              │
        │ Lines 12+3Ni..12+4Ni-1: ion data (σ_z 块)          │
        │ Line   12+4Ni: tot  ...  (σ_z 块 tot)              │
        │ Line   13+4Ni: (blank)                              │
        │ Line   14+4Ni: band ib+1 ... (下一带)              │
        └─────────────────────────────────────────────────────┘
        """
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        # --- 第1-2行: 文件头 ---
        # L1: "PROCAR lm decomposed"
        # L2: "# of k-points:   81  # of bands:  80  # of ions:    5"
        header = lines[1].strip() if len(lines) > 1 else ''
        m = re.search(
            r'# of k-points:\s+(\d+).*# of bands:\s+(\d+).*# of ions:\s+(\d+)',
            header
        )
        if not m:
            raise ValueError(f"无法解析 PROCAR 头部: {header}")
        self.nk = int(m.group(1))
        self.nbands = int(m.group(2))
        self.nions = int(m.group(3))

        # --- 逐块解析 ---
        # 每个 band 块固定占用: 1(空) + 1(ion头) + 4*(nions+1) + 1(空)
        # band 标题行 + 1(空) 在上层处理
        band_block_lines = 4 * (self.nions + 1)  # 4 个 tot + nions 行/块
        band_header_size = 2  # "band X ..." 行 + 空行
        band_size = band_header_size + band_block_lines  # 从 "band X" 行开始

        line_idx = 2  # 跳过前 3 行 (头3行)
        assert lines[line_idx].strip() == '', f"L{line_idx+1} 应为空行"

        for ik in range(1, self.nk + 1):
            line_idx += 1
            kpt_line = lines[line_idx].strip()

            # --- 读取 k 点: L4 ---
            # "k-point     1 :    0.00000000 0.00000000 0.00000000     weight = 0.01234568"
            km = re.match(
                r'k-point\s+\d+\s*:\s+([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+weight\s*=\s*([\d.]+)',
                kpt_line
            )
            if not km:
                # repaired 格式: 负号前可能有空格丢失, 只补数字后的负号
                repaired = re.sub(r'(?<=[\d])(-)', r' \1', kpt_line)
                km = re.match(
                    r'k-point\s+\d+\s*:\s+(-?[\d.]+)\s*(-?[\d.]+)\s*(-?[\d.]+)\s+weight\s*=\s*([\d.]+)',
                    repaired
                )
            if not km:
                raise ValueError(f"无法解析 k 点行 (L{line_idx+1}): {kpt_line}")
            kx, ky, kz, wt = float(km.group(1)), float(km.group(2)), float(km.group(3)), float(km.group(4))
            self.kpoints.append((kx, ky, kz, wt))

            # --- 跳过 k 点行后的空行 (L5) ---
            line_idx += 1

            # --- 解析每个 band ---
            for ib in range(1, self.nbands + 1):
                line_idx += 1

                # --- band 标题: L6 ---
                # "band     1 # energy  -17.73509171 # occ.  1.00000000"
                band_line = lines[line_idx].strip()
                bm = re.match(
                    r'band\s+(\d+)\s+#\s*energy\s+([\-\d.]+)\s+#\s*occ\.\s+([\d.]+)',
                    band_line
                )
                if not bm:
                    raise ValueError(f"无法解析 band 行 (L{line_idx+1}): {band_line}")
                energy = float(bm.group(2))
                occ = float(bm.group(3))

                # --- 跳过空行 + ion 标题行: L7, L8 ---
                line_idx += 2

                # --- 读取 4 个数据块的 tot ---
                tot_vals = []
                for block in range(4):
                    # 跳过 nions 行 (各原子轨道投影)
                    line_idx += self.nions
                    # 跳过到 tot 行
                    line_idx += 1
                    tot_line = lines[line_idx].strip()
                    tot_parts = tot_line.split()
                    if tot_parts[0] != 'tot':
                        raise ValueError(f"期望 tot 行 (L{line_idx+1}): {tot_line}")
                    tot_vals.append(float(tot_parts[-1]))  # 最后一列是 tot

                # tot_vals[0] = charge(ρ₀), [1] = m_x, [2] = m_y, [3] = m_z
                rho = tot_vals[0]
                mx, my, mz = tot_vals[1], tot_vals[2], tot_vals[3]

                # 自旋期望值 = 磁化 / 电荷密度
                if abs(rho) > 1e-30:
                    sx = mx / rho
                    sy = my / rho
                    sz = mz / rho
                else:
                    sx = sy = sz = 0.0

                self.bands[(ik, ib)] = {
                    'energy': energy,
                    'occ': occ,
                    'rho': rho,
                    'mx': mx,
                    'my': my,
                    'mz': mz,
                    'sx': sx,
                    'sy': sy,
                    'sz': sz,
                }

                # 跳过 band 间空行 (最后一行不会有下一个 band 的空行)
                if ib < self.nbands:
                    line_idx += 1
                # 如果是最后一个 band, 且不是最后一个 k 点:
                # 下一行就是 "k-point ..." (会自动处理)

            # k 点结束后跳过所有空行 (可能有多行)
            if ik < self.nk:
                while line_idx + 1 < len(lines) and lines[line_idx + 1].strip() == '':
                    line_idx += 1

    def get_band_energy(self, iband):
        """获取指定 band 在所有 k 点的能量 E(k)"""
        return np.array([self.bands[(ik, iband)]['energy'] for ik in range(1, self.nk + 1)])

    def get_spin(self, iband):
        """获取指定 band 在所有 k 点的自旋 ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩"""
        sx = np.array([self.bands[(ik, iband)]['sx'] for ik in range(1, self.nk + 1)])
        sy = np.array([self.bands[(ik, iband)]['sy'] for ik in range(1, self.nk + 1)])
        sz = np.array([self.bands[(ik, iband)]['sz'] for ik in range(1, self.nk + 1)])
        return sx, sy, sz

    def get_kpoints_cart(self):
        """获取 k 点坐标数组 (nk, 3)"""
        return np.array([[k[0], k[1], k[2]] for k in self.kpoints])

    def get_kweight(self):
        """获取 k 点权重"""
        return np.array([k[3] for k in self.kpoints])

    def find_vbm_cbm(self):
        """
        根据占据数自动查找 VBM / CBM

        原理: SOC 下 Kramers 简并被 lifting,
        VBM 是占据数从 1→0 的边界,
        即最后一个完全占据的 band 是 VBM.
        """
        # 对第 1 个 k 点检查所有 band 的占据数
        occ_list = [self.bands[(1, ib)]['occ'] for ib in range(1, self.nbands + 1)]

        vbm_idx = None
        for ib in range(self.nbands, 0, -1):
            if occ_list[ib - 1] > 0.5:
                vbm_idx = ib
                break

        if vbm_idx is None:
            vbm_idx = self.nbands // 2

        cbm_idx = vbm_idx + 1
        return vbm_idx, cbm_idx

    def summary(self):
        """Print PROCAR summary"""
        print(f"PROCAR: {self.filename}")
        print(f"  k-points:  {self.nk}")
        print(f"  bands:     {self.nbands}")
        print(f"  ions:      {self.nions}")
        ik1 = 1
        b1 = self.bands[(1, 1)]
        print(f"  sample (k=1, band=1):")
        print(f"    energy: {b1['energy']:.4f} eV")
        print(f"    occ:    {b1['occ']}")
        print(f"    charge: {b1['rho']:.4f}")
        print(f"    <sigma>: ({b1['sx']:.4f}, {b1['sy']:.4f}, {b1['sz']:.4f})")
        vbm, cbm = self.find_vbm_cbm()
        print(f"  detected VBM: band {vbm}, CBM: band {cbm}")
