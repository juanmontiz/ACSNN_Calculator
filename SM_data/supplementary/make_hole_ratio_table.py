"""Scaling ratio of the hole networks on their excluded charges (Table of Sec. S10 of the supplementary material).

For every case the ratio R_NN/R_ref = sigma_NN/sigma_ref is evaluated at the valid CDW-EIS energies; its median is taken in
each energy band, and the medians over the cases of each row are reported.
Run from the project root: MODEL_SET=b100k python3 supplementary/make_hole_ratio_table.py
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import zp2_analysis as zp
from evaluate_holes import NAMELIST, CDW_EMIN, read_all_databases, model_names

df = read_all_databases(NAMELIST); df = df[df['Z_target'] == 1]
theories_all = sorted(df['Theory_ID'].unique())
ref = df[df['Theory_ID'] == zp.THEORY].pivot_table(index='Z_ion', columns='E_proj[keV/u]', values='Cross_section[cm2]')
E = ref.columns.values; le = np.log10(E); valid = E >= CDW_EMIN
bands = {'50-500': valid & (E < 500), '500-5000': (E >= 500) & (E < 5000), '>=5000': E >= 5000}

rows = []
for name in model_names():
    excluded, net = zp.load_hole(name, theories_all)
    train = [t for t in range(1, 10) if t not in excluded]
    for z in excluded:
        lo, hi = [t for t in train if t < z], [t for t in train if t > z]
        kind = 'interpolated' if lo and hi else f'extrapolated, d={min(abs(z - t) for t in train)}'
        rho = 10**(net.log_sigma(np.full(len(E), z), le) - np.log10(ref.loc[z].values))
        row = {'model': name, 'Z': z, 'kind': kind, 'dev_%': 100 * np.median(np.abs(rho[valid] - 1))}
        row.update({b: np.median(rho[m]) for b, m in bands.items()})
        rows.append(row)
cases = pd.DataFrame(rows)
tab = cases.groupby(['Z', 'kind']).agg(cases=('model', 'size'), **{b: (b, 'median') for b in bands},
                                       dev=('dev_%', 'median')).reset_index()
out = os.path.join(zp.OUT, 'holes_ratio_by_charge.csv')
tab.to_csv(out, index=False); cases.to_csv(os.path.join(zp.OUT, 'holes_ratio_cases.csv'), index=False)
print(tab.round(3).to_string(index=False))
