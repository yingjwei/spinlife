# spinlife — VASP PROCAR Spin Lifetime Calculator

Extract SOC parameters (α, β) and calculate spin lifetime τ_s directly from VASP LSORBIT PROCAR files.

```
spinlife/              # Main package: VASP PROCAR → α, β, τ_s
legacy/                # Legacy: QE + Phonon + Perturbo workflow for W6CCl16
```

## Quick Start

```bash
pip install -e .
python -m spinlife.main path/to/PROCAR
```

## Input

- `PROCAR`: VASP LSORBIT (non-collinear) calculation output
- `--tau-p`: momentum scattering time τ_p (ps), from experiment or calculation
- `--k-range`: fitting range (Å⁻¹), default 0.05
- `--vbm N`, `--cbm N`: override automatic VBM/CBM detection

## Output

- Terminal report: m*, √(α²+β²), α, β, τ_s, L_PSH
- `spinlife_report.txt`: full text report
- `spinlife_results.png`: 4-panel figure (SOC bands, ΔE² fit, spin texture, 2D map)

## Method

k·p model → ΔE² vs k² fit → √(α²+β²) → spin texture slope → α/β ratio → DP spin lifetime

## Legacy

The `legacy/` directory contains a QE → Phonon → Perturbo workflow for W6CCl16 spin lifetime calculation (Elliott-Yafet & D'yakonov-Perel mechanisms).
