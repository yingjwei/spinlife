# spinlife — VASP 自旋寿命 + 载流子迁移率计算

```
spinlife/          VASP PROCAR → α, β, τ_s (k·p + DP 机制)
qe_perturbo/       QE → Phonon → Perturbo workflow (EY + DP 机制)
```

## 子命令

```bash
# 自旋寿命: 读 VASP LSORBIT PROCAR → α, β, τ_s
python -m spinlife.main PROCAR

# 载流子迁移率: 交互式输入应变数据 → C₂D, E₁, μ
python -m spinlife.main mobility
```

## 安装

```bash
pip install -e .
pip install git+https://github.com/yingjwei/spinlife.git
```

## 自旋寿命 (PROCAR)

```bash
# 基本用法 (交互式选 SOC 带对)
python -m spinlife.main PROCAR

# 导出 band 44 原始数据, 用于手算验证
python -m spinlife.main PROCAR --dump-band 44

# 命令行指定 SOC 带对
python -m spinlife.main PROCAR --soc-vbm 44 43 --soc-cbm 45 46

# 指定参数
python -m spinlife.main PROCAR --vbm 44 --tau-p 0.1 --k-range 0.05
```

### 输出

- `spinlife_report.txt` — m*, α, β, τ_s, L_PSH
- `spinlife_results.png` — SOC 能带 + ΔE² 拟合图 (VBM/CBM 双列)
- `band_N_data.txt` — `--dump-band N` 导出的能带原始数据

## 载流子迁移率

```bash
python -m spinlife.main mobility
```

交互式输入应变-能量数据 → 自动拟合 C₂D 和 E₁ → 输出 μ。

### 输出

- `mobility_report.txt` — C₂D, E₁, μ 报告
- `mobility_fit.png` — C₂D 二次拟合 + E₁ 线性拟合图

## 方法

- 自旋寿命: k·p 模型 → ΔE² vs k² 拟合 → √(α²+β²) → 自旋织构斜率 → α/β → DP τ_s
- 迁移率: 2D 形变势理论 μ = 2eℏ³C₂D / (3kBT|m*|²E₁²)

## QE → Perturbo (W6CCl16)

`qe_perturbo/` 包含 QE → Phonon → Perturbo 工作流, 覆盖 EY + DP 机制.
