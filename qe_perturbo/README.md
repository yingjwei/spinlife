# W6CCl16 Spin Lifetime Calculation (QE → Perturbo)

**W6CCl16** — 窄带隙半导体，二维结构，无磁。
自旋寿命主要由 **Elliott-Yafet (EY)** 和 **D'yakonov-Perel' (DP)** 机制决定。

## Workflow

1. `qe_inputs.py` — Generate QE input files
2. `run_qe.py` — Run QE SCF + NSCF
3. `qe_phonon_inputs.py` — Generate phonon input files
4. `run_phonon.py` — Run phonon calculation
5. `perturbo_inputs.py` — Generate Perturbo input files
6. `run_perturbo.py` — Run Perturbo
7. `analyze_results.py` — Extract spin lifetime
8. `plot_lifetime.py` — Plot results
9. `params.py` — Shared parameters
10. `run_workflow.py` — Full pipeline orchestrator
