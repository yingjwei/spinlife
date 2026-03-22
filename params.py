# ============================================================
# W6CCl16 自旋寿命计算参数  (真实结构)
# ============================================================
# !!! 以下为从 wannier90.win 提取的真实参数 !!!
# !!! 请根据需要修改部分已标注 !!!

# ---------- 基本信息 ----------
material   = "W6CCl16"
calc_dir   = "./calc"          # 计算工作目录

# ---------- 晶格参数 (单位: Angstrom) ----------
a_lat = 12.9972771      # a 轴 (Å)
b_lat = 10.2513403      # b 轴 (Å)
c_lat = 26.5393015      # c 轴 (Å)
alpha = 90.0            # α 角度 (°)
beta  = 90.0            # β 角度 (°)
gamma = 90.0            # γ 角度 (°) — 你的是正交?

# ---------- 原子信息 ----------
natom   = 45            # 总原子数: 12W + 12C + 21Cl
ntyp    = 3             # 元素种类: W, C, Cl
nat_w   = 12            # W 原子数
nat_c   = 12            # C 原子数
nat_cl  = 21            # Cl 原子数

# ----------赝势 ----------
pp_type   = "pbe"       # 赝势类型: pbe, pbesol, lda
pp_family = "slc"       # 赝势族 (根据你的 pseudo_dir 调整)

# ---------- 截断能 & k点 (SCF) ----------
# !!! 以下为估算值，请根据你的赝势调整 !!!
ecutwfc = 60            # 动能截断 (Ry)   !!! 需>=60，建议80-100 !!!
ecutrho = 480           # 电荷密度截断 (Ry), 通常 >= 8*ecutwfc
kpts_scf   = [4, 4, 1]     # SCF k点网格 (你已有 NSCF 数据可用更大)
kpts_nscf  = [6, 6, 1]    # NSCF k点网格 (>= SCF)
kpts_wann  = [3, 3, 1]    # Wannier90 k点网格 (= mp_grid)

# ---------- Wannier90 (来自 wannier90.win) ----------
num_bands    = 512
num_wann     = 488
spinors      = .true.       # 自旋轨道耦合 (spinor)
use_bloch_phases = .false.

dis_win_min  = -78.00       # 外窗口下限 (eV, 相对参考)
dis_win_max  =  3.10        # 外窗口上限 (eV)
dis_froz_min = -6.59        # 冻结窗口下限
dis_froz_max =  1.41        # 冻结窗口上限 (VBM ~ 0.0 eV)
dis_num_iter = 1000
num_iter     = 200
conv_tol     = 1e-10

# ---------- Perturbo ----------
temp        = 300            # 温度 (K)
eta         = 0.01          # 展宽参数 (eV)
energy_min  = -5.0          # 能量范围 (eV)
energy_max  =  3.5          # 能量范围上限 (eV)
spin_qe_flag = True          # 使用 QE 自旋本征态

# ---------- QE → Perturbo 路径 ----------
# 指向 QE 计算产生的 hdf5/wout 文件
qe_prefix      = "wannier90"   # QE 中的 prefix
perturbo_dir   = "calc/perturbo"
qe_scf_outdir  = "../scf/out/"  # QE outdir 相对路径

# ---------- 提交系统 ----------
queue_system = "slurm"       # slurm / pbs / local
job_name     = "W6CCl16_spin"
email        = "your@email.com"  # !!! 替换为你的邮箱 !!!
nodes        = 1
ntasks       = 48            # 建议用 48 核 (W12+C12+Cl21=45原子, 大计算)
walltime     = "24:00:00"
partition    = "compute"
