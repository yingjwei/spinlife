# spinlife — VASP 自旋寿命 + 载流子迁移率计算

```
spinlife/          VASP PROCAR / Wannier90 → α, β, τ_s, μ
qe_perturbo/       QE → Phonon → Perturbo workflow (EY + DP 机制)
```

## 交互式菜单 (默认入口, 推荐)

```bash
# 安装后直接启动
spinlife

# 或
python -m spinlife.main
```

所有输出文件 (报告/图片/数据) 统一保存在当前目录下的 `spinlife/` 文件夹中。

**会话上下文设计:** 各模块计算结果在内存中自动传递, 菜单顶部实时显示当前已计算的值.
进入自旋寿命选项时, 所有已知结果自动预填, 无需手工重复输入.

全功能 vaspkit 风格菜单:

```
  1)  载流子迁移率 (Mobility)           应变-总能量 → C₂D, E₁, μ, τ_p
  2)  有效质量 (Wannier 能带 → m*)       Wannier 密能带 → 抛物线拟合 → m*
  3)  SOC 参数 α/β                      自动扫描 Δk: ΔE² → √(α²+β²), 抛物线 → m*
  4)  自旋寿命 (结果整合)               自动装载 m*, α, β, τ_p → τ_s, L_PSH
  5)  导出能带数据                      PROCAR / Wannier 原始数据导出
  6)  生成 KPOINTS 文件                 读 Wannier → 选 SOC 带对 → 网格

  [m* = 4.3210 m0  |  a = 11.23 meV.A  |  b = 9.10 meV.A  |  tau_p = 0.1500 ps]
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

2D 形变势理论:

$$
C_{\rm 2D} = \frac{1}{A_0}\frac{\partial^2 E}{\partial\varepsilon^2}
\qquad
E_1 = \frac{dE_{\rm VBM}}{d\varepsilon}
$$

$$
\mu = \frac{2e\hbar^3 C_{\rm 2D}}{3k_B T\;|m^*|^2\;E_1^2}
\qquad
\tau_p = \frac{\mu\;m^*}{e}
$$

一次运行收集 X + Y 双方向数据:
- C₂D: 应变-总能量二次拟合
- E₁: 形变势线性拟合
- 各向异性 m*: x/y 分开输入 (若之前已通过 Wannier 算过 m*, 会显示参考值)
- 输出 μ 和 τ<sub>p</sub> (自动存入会话上下文)

### 2. 有效质量 (Wannier)

读 `wannier90_band.dat`, 沿高对称路径抛物线拟合:

$$
E(k) = E_0 + A\,(k - k_0)^2
\qquad\Longrightarrow\qquad
\frac{m^*}{m_0} = \frac{3.81}{|A|}
$$

需提供晶格常数 $a$ (Å) 将 $k$ 转换为 Å⁻¹。
结果自动保存至会话上下文, 供选项 4 使用。

### 3. SOC 参数 α/β

**√(α²+β²)**: Wannier 密能带 → ΔE² vs k² 拟合 (自动扫描最优 Δk, 独立选择)

$$
\Delta E^2 = 4\Delta^2 + 4(\alpha^2+\beta^2)\,k^2
\qquad\Longrightarrow\qquad
\sqrt{\alpha^2+\beta^2} = \frac{\sqrt{\text{slope}}}{2}
$$

**α/β 比值**: PROCAR Γ 点直接做比 — 找离 Γ 最近的可信 k 点 $(|S_x|,|S_y|>10^{-3})$, $\langle\sigma_x\rangle/\langle\sigma_y\rangle = \alpha/\beta$

**有效质量 m***: 单能带抛物线拟合 `fit_effmass()` 自动计算, **与 √(α²+β²) 各自独立扫描最优 Δk**, 与选项 2 统一.

$$
\frac{m^*}{m_0} = \frac{3.81}{|A|}
$$

⟹ 分离 α, β, 自动存入会话上下文。

### 4. 自旋寿命 (DP 机制 — 结果整合)

自动装载会话上下文中的 m*, α, β, τ<sub>p</sub>; 缺失项才要求手工输入.
τ<sub>p</sub> 可直接输入, 或通过 μ 反算 ($\tau_p = \mu \cdot m^* / e$).

$$
\alpha_{\rm eff} = |\alpha - \beta|
\qquad\text{(SU(2) 对称性破缺项)}
$$

$$
\tau_s = \frac{\hbar^2}{2\;m^*\;m_0\;\alpha_{\rm eff}^2\;\tau_p}
$$

$$
L_{\rm PSH} = \frac{\pi\hbar^2}{m^*\;m_0\;|\alpha|}
\approx \frac{2.39}{m^*\;|\alpha|_{\rm meV\cdot\AA}}\;(\mu{\rm m})
$$

### 5. PROCAR 传统模式

```bash
python -m spinlife.main PROCAR --soc-vbm 44 43 --soc-cbm 45 46
python -m spinlife.main PROCAR --dump-band 44
```

## 输出文件

| 文件 | 模块 | 内容 |
|------|------|------|
| `spinlife/mobility_report.txt` | Mobility | C₂D, E₁, μ, τ_p |
| `spinlife/mobility_fit.png` | Mobility | C₂D + E₁ 拟合图 |
| `spinlife/effmass_report.txt` | 有效质量 | m*, R² |
| `spinlife/effmass_fit.png` | 有效质量 | 抛物线拟合图 |
| `spinlife/soc_report.txt` | α/β | √(α²+β²), α, β, R² |
| `spinlife/soc_wannier_fit.png` | Wannier SOC | SOC 能带 + ΔE² 拟合图 |
| `spinlife/soc_spin_fit.png` | PROCAR 自旋 | ⟨σ⟩ vs k (含 Γ 点比值标记) |
| `spinlife/spinlife_report.txt` | 自旋寿命 | τ_s, L_PSH |
| `spinlife/spinlife_results.png` | PROCAR | SOC 能带 + ΔE² 图 |
| `spinlife/band_N_data.txt` | PROCAR --dump-band | k, E, ⟨σ⟩ |
| `KPOINTS` | genkpoints | 围绕 VBM/CBM 的密集 k 网格 |

## QE → Perturbo (W6CCl16)

`qe_perturbo/` 包含 QE → Phonon → Perturbo 工作流, 覆盖 EY + DP 机制.
