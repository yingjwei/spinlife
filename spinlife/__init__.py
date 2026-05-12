# -*- coding: utf-8 -*-
"""
spinlife — VASP PROCAR 自旋寿命计算流水线
"""
from .read_procar import PROCAR
from .fit_soc import fit_effmass, fit_alpha_beta, calc_spin_lifetime
