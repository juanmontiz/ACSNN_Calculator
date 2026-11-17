"""Exact predictions of the single-charge networks that exclude Z_P = 1 and Z_P = 9 (series b100k), on the reference grid.

Writes nn_edge_predictions.csv (hole, theory, E, log10_sigma_nn, log10_sigma_ref), used by make_figures.py.
Run from the ACSNN folder: MODEL_SET=b100k python3 supplementary/_extrepolation_ZP/export_nn_edge_predictions.py
"""
import os, sys
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MODEL_SET', 'b100k')
import numpy as np
import pandas as pd
from evaluate_holes import NAMELIST, read_all_databases, load_model, model_path

df = read_all_databases(NAMELIST)
df = df[df['Z_target'] == 1].reset_index(drop=True)
df['log_E'] = np.log10(df['E_proj[keV/u]'].values)
df['log_sigma'] = np.log10(df['Cross_section[cm2]'].values)
rows = []
for hole in (1, 9):
    m = load_model(model_path(f'{hole}{hole}{hole}'), df)
    for theory in ('CDW-EIS', 'CTMC'):
        ref = df[(df['Z_ion'] == hole) & (df['Theory_ID'] == theory)].sort_values('log_E')
        pred = m['predict'](ref[['Z_ion', 'Theory_ID', 'log_E']].reset_index(drop=True))
        rows += [{'hole': hole, 'theory': theory, 'E': e, 'log10_sigma_nn': p, 'log10_sigma_ref': r}
                 for e, p, r in zip(ref['E_proj[keV/u]'], pred, ref['log_sigma'])]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_edge_predictions.csv')
pd.DataFrame(rows).to_csv(out, index=False)
print('saved', out, len(rows), 'rows')
