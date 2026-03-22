# ============================================================
# W6CCl16 自旋寿命计算参数
# ============================================================
# !!! 需要用户填写的参数，请修改以下值 !!!

# ---------- 基本信息 ----------
material   = "W6CCl16"
calc_dir   = "./calc"          # 计算工作目录

# ---------- 晶格参数 (单位: Angstrom) ----------
# !!! 替换为你的实际结构参数 !!!
a_lat = 10.0      # 晶格常数 a (Å)
b_lat = 10.0      # 晶格常数 b (Å)
c_lat = 25.0       # 层间距 c (Å)
alpha = 90.0       # α 角度 (°)
beta  = 90.0       # β 角度 (°)
gamma = 120.0      # γ 角度 (°)

# ---------- 原子位置 (单位: crystal) ----------
# !!! 替换为你的实际原子坐标 !!!
atoms = {
    "W":  ["0.333", "0.667", "0.500"],   # W 锚定中心
    "C":  ["0.200", "0.400", "0.480"],   # C 原子 (多个)
    "Cl": ["0.100", "0.500", "0.450"],    # Cl 原子 (多个)
}

# ----------赝势 ----------
pp_type   = "pbe"            # 赝势类型: pbe, pbesol, lda
pp_family = "slc"            # W: slc, C: carbon, Cl: chlorine

# ---------- 截断能 & k点 (SCF) ----------
ecutwfc = 80                 # 动能截断 (Ry)   !!! 根据赝势调整 !!!
ecutrho = 640                # 电荷密度截断 (Ry), 通常 = 8*ecutwfc
kpts_scf   = [6, 6, 1]       # SCF k点网格
kpts_nscf  = [12, 12, 1]    # NSCF k点网格 (建议 2x scf)
kpts_wann  = [12, 12, 1]    # Wannier90 展宽 k点

# ---------- Wannier90 ----------
num_wann = 20                # Wannier 函数数目  !!! 需根据能带调整 !!!
dis_froz_max = 2.0          # 冻结能量窗口上限 (eV)，相对 VBM
dis_win_max   = 5.0         # 能量窗口上限 (eV)

# ---------- Perturbo ----------
temp        = 300            # 温度 (K)
eta         = 0.01          # 展宽参数 (eV)
energy_min  = -5.0          # 能量范围 (eV)，相对 CBM
energy_max  =  5.0          # 能量范围 (eV)，相对 CBM
spin_qe_flag = True         # 是否使用 QE 计算的自旋本征态

# ---------- 提交系统 ----------
queue_system = "slurm"       # slurm / pbs / local
job_name     = "W6CCl16_spin"
email        = "your@email.com"  # !!! 替换为你的邮箱 !!!
nodes        = 1
ntasks       = 24
walltime     = "12:00:00"
partition    = "batch"
