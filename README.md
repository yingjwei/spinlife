# W6CCl16 Spin Lifetime Calculation

自旋寿命 (spin lifetime τ_s) 计算，基于 **QE + Wannier90 + Perturbo** 全流程。

> W6CCl16：窄带隙半导体，二维结构，无磁。
> 自旋寿命主要由 **Elliott-Yafet (EY)** 和 **D'yakonov-Perel' (DP)** 机制决定。

---

## 计算流程

```
1_struct_optimize    结构优化 (QE vc-relax)
2_scf               自洽计算 (QE scf)
3_nscf_dense        高密k点计算 (QE nscf)
4_wannier90         Wannier90 MLWF 拟合
5_perturbo          Perturbo 自旋动力学
```

## 主要脚本

| 脚本 | 功能 |
|------|------|
| `run_qe.py` | 提交 QE 计算任务 |
| `run_wannier90.py` | 运行 Wannier90 |
| `run_perturbo.py` | 运行 Perturbo 自旋寿命 |
| `analyze_results.py` | 分析自旋寿命结果 |
| `plot_lifetime.py` | 绘图可视化 |

## 软件依赖

- Quantum ESPRESSO 7.x
- Wannier90 3.x
- Perturbo (perturbopy)
- Python 3.9+ (numpy, matplotlib, scipy)

## 使用

```bash
# 1. 修改输入参数
vim params.py

# 2. 运行完整流程
python run_workflow.py

# 3. 分析结果
python analyze_results.py
python plot_lifetime.py
```

## 参考

- PERTURBO: https://perturbo-code.github.io/
- QE: https://www.quantum-espresso.org/
- Wannier90: https://wannier90.org/
