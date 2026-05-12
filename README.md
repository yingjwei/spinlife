# spinlife — VASP PROCAR Spin Lifetime Calculator

Extract SOC parameters (α, β) and calculate spin lifetime τ_s directly from VASP LSORBIT PROCAR files.

```
spinlife/          VASP PROCAR → α, β, τ_s (k·p + DP 机制)
qe_perturbo/       QE → Phonon → Perturbo workflow (EY + DP 机制)
```

## Quick Start

```bash
pip install -e .
python -m spinlife.main path/to/PROCAR
```

## Usage

```bash
# 先看能带表，决定用哪些带
python -m spinlife.main PROCAR --list-bands

# 交互式选择 SOC 带对（默认同时跑 VBM 和 CBM）
python -m spinlife.main PROCAR

# 命令行指定 SOC 带对（跳过交互）
python -m spinlife.main PROCAR --soc-vbm 44 43 --soc-cbm 45 46

# 只跑 VBM 或 CBM
python -m spinlife.main PROCAR --soc-vbm 44 43 --soc-cbm 0 0

# 指定参数
python -m spinlife.main PROCAR --vbm 44 --tau-p 0.1 --k-range 0.05 --T 300
```

## Options

| Flag | Description |
|------|-------------|
| `--vbm N` | VBM band index (default: auto-detect) |
| `--cbm N` | CBM band index (default: auto-detect) |
| `--soc-vbm U L` | VBM SOC pair: upper & lower bands |
| `--soc-cbm U L` | CBM SOC pair: upper & lower bands |
| `--list-bands` | Print band table at Γ and exit |
| `--tau-p N` | Momentum scattering time τ_p (ps), default 0.1 |
| `--k-range N` | Fitting range (Å⁻¹), default 0.05 |
| `--T N` | Temperature (K), default 300 |
| `--output-dir DIR` | Output directory, default `.` |

## Output

- Terminal report: m*, √(α²+β²), α, β, τ_s, L_PSH (VBM + CBM)
- `band_table.txt` — Γ-point band energies for band selection
- `spinlife_report.txt` — full text report
- `spinlife_results.png` — fitting figure (dual column for VBM & CBM)

## Method

k·p model → ΔE² vs k² fit → √(α²+β²) → spin texture slope → α/β ratio → DP spin lifetime

## QE → Perturbo (W6CCl16)

The `qe_perturbo/` directory contains a DFT → Phonon → Perturbo workflow for W6CCl16, covering both Elliott-Yafet and D'yakonov-Perel mechanisms.
