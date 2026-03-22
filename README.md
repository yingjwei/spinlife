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
| 总原子数 | 45 (12W + 12C + 21Cl) |
| Wannier 函数 | 488 |
| 能带数 | 512 |
| 自旋 | SOC (spinor) |

**冻结窗口**: VBM ~ -6.59 → -1.41 eV  
**外窗口**: -78.0 → +3.10 eV  
**Wannier90 k网格**: 3×3×1

---

## 计算流程

```
1. QE SCF        ← 自洽计算 (非磁性 SOC)
2. QE NSCF       ← 高密 k 点
3. QE pw2wannier90 ← 生成 .amn, .mmn, .eig
4. Wannier90     ← MLWF: num_wann=488, spinor=T
5. Perturbo      ← 自旋动力学 / τ_s
```

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `params.py` | 所有可调参数（截断能、k点、温度…） |
| `qe_inputs.py` | 生成 QE SCF / NSCF / Wannier90 输入 |
| `run_qe.py` | 提交 QE 计算 (SLURM) |
| `perturbo_inputs.py` | 生成 Perturbo 输入 (qe2pert / teout / td) |
| `run_perturbo.py` | 运行 Perturbo 自旋寿命 |
| `analyze_results.py` | 解析 τ_s, τ_φ 等结果 |
| `plot_lifetime.py` | 绘图 (能量/温度依赖) |
| `run_workflow.py` | 一键运行完整流程 |

---

## 使用方法

```bash
# 1. 修改 params.py 中的必要参数（赝势路径、邮箱等）

# 2. 生成输入文件
python qe_inputs.py

# 3. 运行 QE
python run_qe.py scf
python run_qe.py nscf
python run_qe.py wannier
python run_qe.py proj

# 4. 运行 Perturbo
python perturbo_inputs.py
python run_perturbo.py all

# 5. 分析结果
python analyze_results.py
python plot_lifetime.py
```

## 软件依赖

- Quantum ESPRESSO 7.x (with SOC)
- Wannier90 3.x
- Perturbo (perturbopy)
- Python 3.9+ (numpy, matplotlib, scipy)

## 参考

- PERTURBO: https://perturbo-code.github.io/
- QE: https://www.quantum-espresso.org/
- Wannier90: https://wannier90.org/
