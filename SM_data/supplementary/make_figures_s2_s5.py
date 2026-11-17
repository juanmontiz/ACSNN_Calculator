"""Regenerate Figs. S2 and S5 of the supplementary material from the b100k networks.

Run from the project root: MODEL_SET=b100k python3 supplementary/make_figures_s2_s5.py
"""
import os, shutil, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluate_holes import NAMELIST, read_all_databases, load_model, model_path
from evaluate_three_charges import OUT, plot_models

FIG_DIR = 'supplementary/figures'
CLASS_COLORS = {'interpolation': 'tab:blue', 'extrapolation': 'tab:red'}

df = read_all_databases(NAMELIST)
df = df[df['Z_target'] == 1].reset_index(drop=True)
df['log_E'] = np.log10(df['E_proj[keV/u]'].values)
df['log_sigma'] = np.log10(df['Cross_section[cm2]'].values)

# Fig. S2: predicted and reference cross section of each excluded charge
fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
for ax, z in zip(axes.ravel(), range(1, 10)):
    cls = 'extrapolation' if z in (1, 9) else 'interpolation'
    m = load_model(model_path(f'{z}{z}{z}'), df)
    for theory, mk, col in [('CDW-EIS', 'o', 'tab:green'), ('CTMC', 's', 'tab:purple')]:
        ref = df[(df['Z_ion'] == z) & (df['Theory_ID'] == theory)].sort_values('log_E')
        grid = pd.DataFrame({'Z_ion': z, 'Theory_ID': theory,
                             'log_E': np.linspace(ref['log_E'].min(), ref['log_E'].max(), 200)})
        ax.plot(10**grid['log_E'], 10**m['predict'](grid), color=col, label=f'NN {theory}')
        ax.plot(ref['E_proj[keV/u]'], ref['Cross_section[cm2]'], mk, color=col, mfc='none', ms=4,
                label=f'{theory} reference')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-19, 1e-14)
    ax.set_title(f'$Z_P$ = {z} excluded ({cls})', color=CLASS_COLORS[cls])
    ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.6)
for ax in axes[-1]:
    ax.set_xlabel('E [keV/u]')
for ax in axes[:, 0]:
    ax.set_ylabel('σ [cm²]')
axes[0, 0].legend(fontsize=8)
fig.suptitle('Single-charge exclusion: NN prediction on the unseen charge (target H)')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'one_curves_per_excluded_charge.png'), dpi=130)
plt.close(fig)

# Fig. S5: network ranking
plot_models(pd.read_csv(os.path.join(OUT, 'models.csv'), dtype={'model': str}))
shutil.copy(os.path.join(OUT, 'model_ranking.png'), os.path.join(FIG_DIR, 'three_model_ranking.png'))
