# spinlife — VASP 自旋寿命 + 载流子迁移率计算

```
spinlife/          VASP PROCAR / Wannier90 → α, β, τ_s, μ
qe_perturbo/       QE → Phonon → Perturbo workflow (EY + DP 机制)
```

## 交互式菜单 (默认入口)

```bash
python -m spinlife.main
```

全功能 vaspkit 风格菜单:

```
  1)  载流子迁移率 (Mobility)           应变-总能量 → C₂D, E₁, μ, τ_p
  2)  有效质量 (Wannier 能带 → m*)       Wannier 密能带 → 抛物线拟合 → m*
  3)  SOC 参数 α/β                      Wannier ΔE² → √(α²+β²), PROCAR → α/β 比值
  4)  自旋寿命 (手工输入 → τ_s)          m* + α/β + τ_p → τ_s, L_PSH
  5)  导出能带数据                       PROCAR / Wannier 原始数据导出
```

## 快捷命令 (向后兼容)

```bash
# 自旋寿命 (PROCAR 传统模式)
python -m spinlife.main PROCAR [options]

# 载流子迁移率
python -m spinlife.main mobility

# 直接执行
python /path/to/spinlife/main.py
```

## 安装

```bash
pip install -e .
pip install git+https://github.com/yingjwei/spinlife.git
```

## 功能详解

### 1. 载流子迁移率

2D 形变势理论: μ = 2eℏ³C₂D / (3kBT|m*|²E₁²)

一次运行收集 X + Y 双方向数据:
- C₂D: 应变-总能量二次拟合
- E₁: 形变势线性拟合
- 各向异性 m*: x/y 分开输入
- 输出 μ 和 τ_p

### 2. 有效质量 (Wannier)

读 `wannier90_band.dat`, 沿高对称路径抛物线拟合:

```
E(k) = E₀ + A(k-k₀)²
m*/m₀ = 3.81 / |A|
```

需提供晶格常数 a (Å) 将 k 转换为 Å⁻¹。

### 3. SOC 参数 α/β

**√(α²+β²)**: Wannier 密能带 → ΔE² vs k² 拟合 (可靠)
**α/β 比值**: PROCAR 自旋期望 ⟨σ_x⟩/⟨σ_y⟩ Γ 附近平均值 (无需拟合)

⇒ 分离 α, β

### 4. 自旋寿命 (DP 机制)

手工输入 m*, α, β, τ_p → τ_s, L_PSH

### 5. PROCAR 传统模式

```bash
python -m spinlife.main PROCAR --soc-vbm 44 43 --soc-cbm 45 46
python -m spinlife.main PROCAR --dump-band 44
```

## 输出文件

| 文件 | 模块 | 内容 |
|------|------|------|
| `mobility_report.txt` | Mobility | C₂D, E₁, μ, τ_p |
| `mobility_fit.png` | Mobility | C₂D + E₁ 拟合图 |
| `effmass_report.txt` | 有效质量 | m* |
| `soc_report.txt` | α/β | √(α²+β²), α, β |
| `spinlife_report.txt` | 自旋寿命 | τ_s, L_PSH |
| `spinlife_results.png` | PROCAR | SOC 能带 + ΔE² 图 |
| `band_N_data.txt` | PROCAR --dump-band | k, E, ⟨σ⟩ |

## QE → Perturbo (W6CCl16)

`qe_perturbo/` 包含 QE → Phonon → Perturbo 工作流, 覆盖 EY + DP 机制.
