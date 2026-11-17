"""Regenerate Fig. S6 (scaling ratio of the ACSNN interpolator) of the supplementary material.

Run from the project root: MODEL_SET=b100k python3 supplementary/make_zp2_figures.py
"""
import os, shutil, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import zp2_analysis as zp
from evaluate_holes import NAMELIST, read_all_databases

df = read_all_databases(NAMELIST)
df = df[df['Z_target'] == 1]
ref = df[df['Theory_ID'] == zp.THEORY].pivot_table(index='Z_ion', columns='E_proj[keV/u]', values='Cross_section[cm2]')
zp.fig_reference_and_acsnn(ref, zp.load_acsnn(), np.logspace(1, 5, 300))
shutil.copy(os.path.join(zp.OUT, 'acsnn_scaled_cross_section.png'), 'supplementary/figures/zp2_acsnn_scaled_cross_section.png')
