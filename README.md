# W6CCl16 Spin Lifetime Calculation

**W6CCl16** — 窄带隙半导体，二维结构，无磁。
自旋寿命主要由 **Elliott-Yafet (EY)** 和 **D'yakonov-Perel' (DP)** 机制决定。

---

## 结构参数 (来源: wannier90.win)

| 参数 | 值 |
|------|------|
| 晶格 a | 12.997 Å |
| 晶格 b | 10.251 Å |
| 晶格 c | 26.539 Å |
| 总原子数 | 46 (12W + 2C + 32Cl) |
| Wannier 函数 | 488 |
| 能带数 | 512 |
| mp_grid | 3×3×1 |
| 自旋 | SOC (spinor) |

**冻结窗口**: -6.59 → 1.41 eV (VBM ~ 0)  
**外窗口**: -78.0 → +3.10 eV  

---

## 完整计算流程

```
阶段一: QE 电子结构
├── 1_scf/       SCF 自洽 (SOC)
├── 2_nscf/      NSCF 高密 k 点
└── pw2wannier90 → wannier90.mmn/.amn/.eig

阶段二: Wannier90 MLWF
└── wannier90.win → wannier90.amn/.mmn/.eig

阶段三: QE 声子 (DFPT)  ← 电子-声子散射矩阵
├── ph.in        声子微扰 (q 点网格 3×3×1)
├── q2r.in       dynmat → 原子间力常数 (IFC)
├── matdyn.in    声子色散
└── dynmat 文件 → EPW / Perturbo elphmat

阶段四: EPW 电子-声子耦合 ← 推荐路径
└── epw.in       el-ph 矩阵元 (Wannier 插值)

阶段五: Perturbo 自旋动力学
├── qe2pert.in   QE → Perturbo 格式
├── ephmat.in    读取 el-ph 矩阵
└── td.in        自旋寿命 τ_s
```

**注意**: 声子计算 (阶段三) 是自旋寿命计算的必要输入！
不计算声子 → 无电子-声子散射 → 无法得到真实 τ_s

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `params.py` | 所有可调参数 |
| `qe_inputs.py` | QE SCF / NSCF / Wannier90 输入 |
| `run_qe.py` | 提交 QE 计算 (SLURM) |
| `qe_phonon_inputs.py` | QE ph / q2r / matdyn / EPW 输入 |
| `run_phonon.py` | 提交声子计算 (SLURM) |
| `perturbo_inputs.py` | Perturbo qe2pert / ephmat / td 输入 |
| `run_perturbo.py` | 运行 Perturbo 自旋寿命 |
| `analyze_results.py` | 解析 τ_s, τ_φ 等结果 |
| `plot_lifetime.py` | 绘图 (能量/温度依赖) |
| `run_workflow.py` | 一键运行完整流程 |

---

## 使用方法

```bash
# 1. 修改 params.py（赝势路径、邮箱等）
vim params.py

# 2. 生成所有输入文件
python qe_inputs.py           # 阶段一
python run_qe.py scf          # SCF
python run_qe.py nscf         # NSCF
python run_qe.py wannier      # Wannier90

# 3. 声子计算 (阶段三)
python run_phonon.py generate
python run_phonon.py ph        # ph.x (最长，约数小时)
python run_phonon.py q2r       # q2r.x
python run_phonon.py matdyn    # matdyn.x

# 4. EPW el-ph 耦合 (阶段四) — 推荐
python run_phonon.py epw       # epw.x (最长，约数天)

# 5. Perturbo 自旋寿命 (阶段五)
python perturbo_inputs.py      # 生成 Perturbo 输入
python run_perturbo.py all     # 运行完整 Perturbo

# 6. 分析结果
python analyze_results.py
python plot_lifetime.py
```

---

## 软件依赖

- Quantum ESPRESSO 7.x (with SOC + ph.x)
- Wannier90 3.x
- EPW (QE 配套)
- Perturbo (perturbopy)
- Python 3.9+ (numpy, matplotlib, scipy)

## 参考

- PERTURBO: https://perturbo-code.github.io/
- QE: https://www.quantum-espresso.org/
- Wannier90: https://wannier90.org/
- EPW: QE 配套包
