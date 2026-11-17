"""Combined figure replacing Figs. 5, 6 and 7 of the manuscript (MLP fit of CDW-EIS, CTMC and RSEM).

Top row: TCS of all projectile charges, one panel per theory. Bottom row: relative error, as a Z_P x E map for
CDW-EIS and CTMC and as a curve for RSEM. Model, palette, fonts and sizes are those of NN-ion-papiro/performance.ipynb.
Run from the ACSNN folder: python3 manuscript_figures/make_combined_fit_figure.py
"""
import os, sys, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, '.')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from evaluate_holes import NAMELIST, read_all_databases
from train_totals_NN_holes import FlexibleNN

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Palatino', 'serif']
plt.rcParams['font.size'] = 16

MODEL = 'best_statedict_gauss_128.pth'
THEORIES = ['CDW-EIS', 'CTMC', 'Semiempiric_1985Rudd']       # one-hot order of the model
ERROR_NORM = 'reference'                                     # 'reference': |NN - ref| / ref; 'prediction': |NN - ref| / NN
OUT = 'manuscript_figures'
E_AU = 24.80                                                 # keV/u; eta = Z_P / v = 1 at E = E_AU Z_P^2

# Palette of the manuscript (colorlist3)
INK, TEAL, CYAN, AQUA, VANILLA, GOLD, CARAMEL, RUSTY, IRON, BROWN = [
    '#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
Z_COLORS = dict(zip(range(1, 10), [INK, TEAL, CYAN, AQUA, GOLD, CARAMEL, RUSTY, IRON, BROWN]))
ERROR_CMAP = LinearSegmentedColormap.from_list('error', ['#ffffff', AQUA, CYAN, TEAL, INK])
ERROR_NORM_SCALE = Normalize(vmin=0, vmax=30)


def load_model():
    state = torch.load(MODEL, map_location='cpu')
    model = FlexibleNN(state['lin1.weight'].shape[1], state['lin1.weight'].shape[0], 'gauss', 0.0)
    model.load_state_dict(state)
    return model.eval()


def predict(model, theory, z, energies):
    onehot = np.tile(np.eye(3)[THEORIES.index(theory)], (len(energies), 1))
    x = np.column_stack([np.full(len(energies), z), np.log10(energies), onehot])
    with torch.no_grad():
        return 10**model(torch.tensor(x, dtype=torch.float32)).numpy().ravel().astype(float)


def rel_error(pred, ref):
    return 100 * np.abs(pred - ref) / (ref if ERROR_NORM == 'reference' else pred)


def style(ax):
    ax.tick_params(axis='both', which='both', direction='in', length=4, width=2, labelsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def cell_edges(energies):
    """Cell limits in E for a map: geometric midpoints between grid energies."""
    mid = np.sqrt(energies[1:] * energies[:-1])
    return np.r_[energies[0]**2 / mid[0], mid, energies[-1]**2 / mid[-1]]


def panel_label(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=28, va='top', ha='left')


if __name__ == '__main__':
    df = read_all_databases(NAMELIST)
    df = df[df['Z_target'] == 1]
    model = load_model()

    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], width_ratios=[1, 0.8, 0.8])
    top = [fig.add_subplot(gs[0, k]) for k in range(3)]
    bottom = [fig.add_subplot(gs[1, k], sharex=top[k]) for k in range(3)]

    for k, theory in enumerate(['CDW-EIS', 'CTMC']):
        sub = df[df['Theory_ID'] == theory]
        ref = sub.pivot_table(index='Z_ion', columns='E_proj[keV/u]', values='Cross_section[cm2]')
        energies = ref.columns.values
        grid = np.logspace(np.log10(energies[0]), np.log10(energies[-1]), 1000)
        err = np.zeros(ref.shape)
        for i, z in enumerate(ref.index):
            top[k].loglog(grid, predict(model, theory, z, grid), lw=3, color=Z_COLORS[z])
            top[k].loglog(energies, ref.loc[z], '*', ms=10, color=Z_COLORS[z], mec='w', mew=0.6, zorder=3)
            err[i] = rel_error(predict(model, theory, z, energies), ref.loc[z].values)
        mesh = bottom[k].pcolormesh(cell_edges(energies), np.arange(0.5, 10), err, cmap=ERROR_CMAP,
                                    norm=ERROR_NORM_SCALE, edgecolors='none', rasterized=True)
        bottom[k].set_xscale('log')
        bottom[k].set_yticks(range(1, 10))
        print(theory, 'error median %.2f max %.2f' % (np.median(err), err.max()))

    # Sommerfeld limit eta = Z_P / v = 1 on the CDW-EIS map
    z_curve = np.linspace(0.5, 9.5, 200)
    limits = bottom[0].get_xlim()
    bottom[0].plot(E_AU * z_curve**2, z_curve, ':', lw=2, color=RUSTY)
    bottom[0].set_xlim(limits)
    bottom[0].text(130, 4.6, r'$\eta>1$', fontsize=20, color=RUSTY, ha='center', va='center')
    bottom[0].text(8000, 4.6, r'$\eta<1$', fontsize=20, color=RUSTY, ha='center', va='center')

    sub = df[df['Theory_ID'] == 'Semiempiric_1985Rudd'].sort_values('E_proj[keV/u]').iloc[2:]   # as in Fig. 7
    energies, sigma = sub['E_proj[keV/u]'].values, sub['Cross_section[cm2]'].values
    grid = np.logspace(np.log10(energies[0]), np.log10(energies[-1]), 1000)
    top[2].loglog(energies, sigma, '*', ms=10, color=GOLD)
    top[2].loglog(grid, predict(model, 'Semiempiric_1985Rudd', 1, grid), lw=3, color=TEAL)
    top[2].text(0.5, 0.2, r'$Z_P$=1', transform=top[2].transAxes, fontsize=20)
    err = rel_error(predict(model, 'Semiempiric_1985Rudd', 1, energies), sigma)
    bottom[2].semilogx(energies, err, c=CYAN, lw=2)
    bottom[2].fill_between(energies, err, color=AQUA, alpha=0.5)
    bottom[2].set_ylim(0, 2)
    bottom[2].tick_params(axis='y', labelcolor=CYAN)
    bottom[2].set_ylabel(r'relative error (\%)', color=CYAN, fontsize=18)
    print('RSEM error median %.2f max %.2f' % (np.median(err), err.max()))

    for ax in top[:2]:
        ax.set_ylim(1e-19, 1e-14)
    top[0].set_ylabel(r'TCS ($cm^2$)', fontsize=22)
    top[2].set_ylabel(r'TCS ($cm^2$)', fontsize=22)
    bottom[0].set_ylabel(r'$Z_P$', fontsize=22)
    for ax in bottom:
        ax.set_xlabel('Projectile Energy (keV/u)', fontsize=22)
    for ax in top + bottom:
        style(ax)
    for ax in top:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax, label in zip(top, ['(a)', '(b)', '(c)']):
        panel_label(ax, label)
    for ax, label in zip(bottom, ['(d)', '(e)', '(f)']):
        panel_label(ax, label)

    handles = [Line2D([], [], color=Z_COLORS[z], lw=3, marker='*', ms=10, mec='w', mew=0.6, label=str(z))
               for z in range(1, 10)]
    top[0].legend(handles=handles, title=r'$Z_P$', ncol=3, fontsize=14, title_fontsize=16, frameon=False,
                  loc='lower left', bbox_to_anchor=(0.2, 0.0), columnspacing=0.8, handlelength=1.6, handletextpad=0.4)
    cbar = fig.colorbar(mesh, ax=bottom[:2], location='right', pad=0.01, aspect=18)
    cbar.set_label(r'relative error (\%)', color=CYAN, fontsize=18)
    cbar.ax.tick_params(labelsize=18, direction='in', width=2, length=4, labelcolor=CYAN)
    cbar.outline.set_linewidth(1.5)
    cbar.set_ticks([0, 5, 10, 15, 20, 25, 30])

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT, f'performance_combined.{ext}'), dpi=200)
