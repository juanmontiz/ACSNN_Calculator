"""All figures of the supplementary material, in the style of the manuscript figures.

Fonts, tick style and palette follow manuscript_figures/make_combined_fit_figure.py (NN-ion-papiro/performance.ipynb).
Hole networks: series b100k. ACSNN interpolator: the model of the main article.
Run from the ACSNN folder:  MODEL_SET=b100k python3 supplementary/make_sm_figures.py [S1 S2 ...]
"""
import os, sys, glob, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MODEL_SET', 'b100k')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import zp2_analysis as zp
from evaluate_holes import NAMELIST, CDW_EMIN, read_all_databases, load_model, model_path, model_names

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Palatino', 'serif']
plt.rcParams['font.size'] = 16

FIG = os.path.join('supplementary', 'figures')
INK, TEAL, CYAN, AQUA, VANILLA, GOLD, CARAMEL, RUSTY, IRON, BROWN = [
    '#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
Z_COLORS = dict(zip(range(1, 10), [INK, TEAL, CYAN, AQUA, GOLD, CARAMEL, RUSTY, IRON, BROWN]))
THEORY_STYLE = {'CDW-EIS': dict(color=TEAL, ls='-', marker='o'), 'CTMC': dict(color=RUSTY, ls='--', marker='s')}
CASE_COLORS = {'interp. gap 1': TEAL, 'interp. gap 2': CYAN, 'interp. gap 3': AQUA,
               'extrap. d=1': GOLD, 'extrap. d=2': RUSTY, 'extrap. d=3': BROWN}
CASE_LABELS = {'interp. gap 1': 'gap 1', 'interp. gap 2': 'gap 2', 'interp. gap 3': 'gap 3',
               'extrap. d=1': r'$d=1$', 'extrap. d=2': r'$d=2$', 'extrap. d=3': r'$d=3$'}
NEXT_COLORS = {0: TEAL, 1: GOLD, 2: RUSTY, 3: INK}
ORDER_COLORS = [TEAL, GOLD, IRON]          # excluded charges of one network, in increasing order
EXCL_COLORS = {1: TEAL, 2: GOLD, 3: IRON}
SIGMA_LABEL = r'TCS (cm$^2$)'
E_LABEL = r'Projectile Energy (keV/u)'
LETTERS = 'abcdefghi'


def style(ax, labelsize=16):
    ax.tick_params(axis='both', which='both', direction='in', length=4, width=2, labelsize=labelsize, top=True,
                   right=True)
    ax.tick_params(axis='both', which='minor', length=2, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def label(ax, letter, text=None, x=0.04, y=0.95, color='k'):
    s = f'({letter})' + (f' {text}' if text else '')
    ax.text(x, y, s, transform=ax.transAxes, fontsize=18, va='top', ha='left', color=color)


def data():
    df = read_all_databases(NAMELIST)
    df = df[df['Z_target'] == 1].reset_index(drop=True)
    df['log_E'] = np.log10(df['E_proj[keV/u]'].values)
    df['log_sigma'] = np.log10(df['Cross_section[cm2]'].values)
    return df


def curve(m, df, z, theory, n=200):
    ref = df[(df['Z_ion'] == z) & (df['Theory_ID'] == theory)]
    grid = pd.DataFrame({'Z_ion': z, 'Theory_ID': theory, 'log_E': np.linspace(ref['log_E'].min(), ref['log_E'].max(), n)})
    return 10**grid['log_E'].values, 10**m['predict'](grid)


def reference(ax, df, z, theory, color, ms=5):
    ref = df[(df['Z_ion'] == z) & (df['Theory_ID'] == theory)]
    ax.plot(ref['E_proj[keV/u]'], ref['Cross_section[cm2]'], THEORY_STYLE[theory]['marker'], color=color, mfc='none',
            ms=ms, mew=1.2, ls='none')


def sigma_axes(axes, ncols):
    for k, ax in enumerate(axes.ravel()):
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-19, 1e-14)
        style(ax)
        if k % ncols:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(SIGMA_LABEL, fontsize=20)
    for ax in axes[-1]:
        ax.set_xlabel(E_LABEL, fontsize=20)


# ------------------------------------------------------------------------------------------------ S1
def fig_s1():
    # full histories if present (after retraining), otherwise the reduced ones in histories/ (see history_io.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from history_io import load_histories
    hist = load_histories()
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)
    for k, col in EXCL_COLORS.items():
        H = hist[k]
        ep = H[0]['epoch']
        tr = np.array([h['train'] for h in H]); va = np.array([h['val'] for h in H])
        best = np.array([h['best'] for h in H])
        q = np.percentile(best, [25, 50, 75], axis=0)
        ax[0].plot(ep, q[1], color=col, lw=2.5, label=f'{k} excluded ({len(H)} networks)')
        ax[0].fill_between(ep, q[0], q[2], color=col, alpha=0.25, lw=0)
        ax[0].plot(np.median([h['argmin_epoch'] for h in H]), np.median([h['min_val'] for h in H]), 'v', color=col,
                   ms=11, mec='k')
        ax[1].plot(ep, np.median(tr, axis=0), color=col, lw=2, label=f'{k} excluded')
        ax[1].plot(ep, np.median(va, axis=0), color=col, lw=2, ls='--')
    for a, letter in zip(ax, 'ab'):
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('Epoch', fontsize=20); style(a)
        label(a, letter, y=0.06, x=0.04)
    ax[0].set_ylabel('Lowest validation loss so far', fontsize=20)
    ax[1].set_ylabel('Loss per epoch (median)', fontsize=20)
    ax[0].legend(frameon=False, fontsize=15)
    handles = [Line2D([], [], color=col, lw=2, label=f'{k} excluded') for k, col in EXCL_COLORS.items()]
    handles += [Line2D([], [], color='k', lw=2, label='training (dropout on)'),
                Line2D([], [], color='k', lw=2, ls='--', label='validation (dropout off)')]
    ax[1].legend(handles=handles, frameon=False, fontsize=15)
    fig.savefig(os.path.join(FIG, 'training_convergence.png'), dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------------ S2
def fig_s2(df):
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=True, constrained_layout=True)
    for k, (ax, z) in enumerate(zip(axes.ravel(), range(1, 10))):
        m = load_model(model_path(f'{z}{z}{z}'), df)
        for theory, st in THEORY_STYLE.items():
            e, s = curve(m, df, z, theory)
            ax.plot(e, s, color=st['color'], ls=st['ls'], lw=2.5)
            reference(ax, df, z, theory, st['color'])
        kind = 'extrapolated' if z in (1, 9) else 'interpolated'
        label(ax, LETTERS[k], rf'$Z_P={z}$, {kind}')
    sigma_axes(axes, 3)
    handles = [Line2D([], [], color=st['color'], ls=st['ls'], lw=2.5, marker=st['marker'], mfc='none', ms=6,
                      label=f'{t}') for t, st in THEORY_STYLE.items()]
    axes[0, 1].legend(handles=handles, frameon=False, fontsize=16, loc='lower center')
    fig.savefig(os.path.join(FIG, 'one_curves_per_excluded_charge.png'), dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------------ S3, S4
def selected_curves(df, networks, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), sharex=True, constrained_layout=True)
    theory_handles = [Line2D([], [], color='k', ls=st['ls'], lw=2, marker=st['marker'], mfc='none', ms=6, label=t)
                      for t, st in THEORY_STYLE.items()]
    for k, (ax, (excluded, name)) in enumerate(zip(axes.ravel(), networks)):
        m = load_model(model_path(name), df)
        handles = []
        for z, col in zip(excluded, ORDER_COLORS):
            for theory, st in THEORY_STYLE.items():
                e, s = curve(m, df, z, theory)
                ax.plot(e, s, color=col, ls=st['ls'], lw=2.5 if theory == 'CDW-EIS' else 1.8)
                reference(ax, df, z, theory, col)
            handles.append(Line2D([], [], color=col, lw=2.5, label=rf'$Z_P={z}$'))
        label(ax, LETTERS[k], '(' + ','.join(str(z) for z in excluded) + ')')
        ax.legend(handles=(theory_handles if k == 0 else []) + handles, frameon=False, fontsize=14, loc='upper right')
    sigma_axes(axes, 3)
    fig.savefig(os.path.join(FIG, out), dpi=200); plt.close(fig)


def pair_model(a, b):
    for name in (f'{a}{a}{b}', f'{a}{b}{b}'):
        if os.path.exists(model_path(name)):
            return name
    raise FileNotFoundError(f'no model for pair ({a},{b})')


def fig_s3(df):
    pairs = [(4, 5), (3, 7), (2, 3), (1, 2), (8, 9), (1, 9)]
    selected_curves(df, [((a, b), pair_model(a, b)) for a, b in pairs], 'two_curves_selected_pairs.png')


def fig_s4(df):
    triples = ['357', '456', '234', '123', '789', '159']
    selected_curves(df, [(tuple(int(c) for c in t), t) for t in triples], 'three_curves_selected_triples.png')


# ------------------------------------------------------------------------------------------------ S5
def fig_s5():
    tab = pd.read_csv(os.path.join('eval_holes_b100k', 'three_charges', 'models.csv'), dtype={'model': str})
    s = tab.sort_values('MAE_log').reset_index(drop=True)
    pos = np.arange(len(s))
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), sharey=True, gridspec_kw={'width_ratios': [2, 1]},
                           constrained_layout=True)
    ax[0].bar(pos, s['MAE_log'], width=0.8, color=[NEXT_COLORS[n] for n in s['n_extrapolated']])
    ax[0].set_xticks(pos); ax[0].set_xticklabels([','.join(m) for m in s['model']], rotation=90, fontsize=10)
    ax[0].set_xlim(-1, len(s))
    for k, tick in enumerate(ax[0].xaxis.get_major_ticks()):
        tick.set_pad(3 if k % 2 == 0 else 40)
        if k % 2:
            ax[0].annotate('', xy=(k, 0), xycoords=ax[0].get_xaxis_transform(), xytext=(0, -41),
                           textcoords='offset points', annotation_clip=False,
                           arrowprops=dict(arrowstyle='-', lw=0.5, color='0.45', shrinkA=0, shrinkB=0))
    ax[0].set_yscale('log'); ax[0].set_ylabel(r'Network MAE ($\log_{10}\sigma$)', fontsize=20)
    ax[0].set_xlabel('Excluded charges of the network, sorted by MAE', fontsize=20)
    ax[0].legend(handles=[Patch(color=c, label=f'{n} extrapolated') for n, c in NEXT_COLORS.items()], frameon=False,
                 fontsize=15, loc='upper left', bbox_to_anchor=(0.06, 1.0))
    rng = np.random.default_rng(2)
    for n in sorted(tab['n_extrapolated'].unique()):
        sub = tab[tab['n_extrapolated'] == n]
        ax[1].scatter(n + rng.normal(0, 0.07, len(sub)), sub['MAE_log'], s=28, color=NEXT_COLORS[n], zorder=3)
        ax[1].plot(n, sub['MAE_log'].median(), '_', color='k', ms=30, mew=2.5, zorder=4)
    ax[1].set_xticks(sorted(tab['n_extrapolated'].unique())); ax[1].set_xlim(-0.5, 3.5)
    ax[1].set_xlabel('Extrapolated charges', fontsize=20)
    for a, letter in zip(ax, 'ab'):
        style(a); label(a, letter, x=0.02 if letter == 'a' else 0.05)
    ax[0].tick_params(axis='x', which='both', top=False)
    fig.savefig(os.path.join(FIG, 'three_model_ranking.png'), dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------------ S6, S7
def cdw_reference(df):
    return df[df['Theory_ID'] == 'CDW-EIS'].pivot_table(index='Z_ion', columns='E_proj[keV/u]',
                                                         values='Cross_section[cm2]')


def grey_band(ax, ref):
    ax.axvspan(ref.columns.min() * 0.8, CDW_EMIN, color='0.9', zorder=0, lw=0)


def fig_s6(df):
    ref = cdw_reference(df); sp = ref.loc[1]; E = ref.columns.values
    acs = zp.load_acsnn()
    grid = np.logspace(1, 5, 300); lg = np.log10(grid)
    sp_grid = 10**np.interp(lg, np.log10(E), np.log10(sp.values))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    for z in range(1, 10):
        s_nn = 10**acs.log_sigma(np.full_like(lg, z), lg)
        ax[0].plot(grid, s_nn / z**2, color=Z_COLORS[z], lw=2)
        ax[0].plot(E, ref.loc[z] / z**2, 'o', color=Z_COLORS[z], mfc='none', ms=5, mew=1.2)
        ax[1].plot(grid, s_nn / (z**2 * sp_grid), color=Z_COLORS[z], lw=2)
        ax[1].plot(E, ref.loc[z] / (z**2 * sp), 'o', color=Z_COLORS[z], mfc='none', ms=5, mew=1.2)
    ax[0].set_yscale('log'); ax[0].set_ylim(1e-21, 1e-15); ax[0].set_ylabel(r'$\sigma/Z_P^2$ (cm$^2$)', fontsize=20)
    ax[1].set_yscale('log'); ax[1].set_ylim(0.03, 2); ax[1].axhline(1, color='k', lw=1)
    ax[1].set_ylabel(r'$R=\sigma(Z_P)\,/\,[Z_P^2\,\sigma_p]$', fontsize=20)
    for a, letter in zip(ax, 'ab'):
        a.set_xscale('log'); a.set_xlim(8, 1.25e5); grey_band(a, ref); a.set_xlabel(E_LABEL, fontsize=20); style(a)
        label(a, letter, x=0.04 if letter == 'a' else 0.9, y=0.95 if letter == 'a' else 0.08)
    handles = [Line2D([], [], color=Z_COLORS[z], lw=2, marker='o', mfc='none', ms=5, label=str(z)) for z in range(1, 10)]
    ax[0].legend(handles=handles, title=r'$Z_P$', ncol=3, frameon=False, fontsize=14, title_fontsize=16,
                 loc='upper right', columnspacing=0.8)
    fig.savefig(os.path.join(FIG, 'zp2_acsnn_scaled_cross_section.png'), dpi=200); plt.close(fig)


def fig_s7(df):
    ref = cdw_reference(df); sp = ref.loc[1]; E = ref.columns.values; le = np.log10(E)
    theories_all = sorted(df['Theory_ID'].unique())
    curves = {}
    for name in model_names():
        excluded, net = zp.load_hole(name, theories_all)
        train = [t for t in range(1, 10) if t not in excluded]
        for z in excluded:
            curves.setdefault(z, []).append((zp.classify(z, train), 10**net.log_sigma(np.full(len(E), z), le)))
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=True, sharey=True, constrained_layout=True)
    for k, (ax, z) in enumerate(zip(axes.ravel(), range(1, 10))):
        for c in CASE_COLORS:
            for cls, s in curves[z]:
                if cls == c:
                    ax.plot(E, s / (z**2 * sp.values), color=CASE_COLORS[c], lw=1.2, alpha=0.75)
        ax.plot(E, ref.loc[z] / (z**2 * sp), 'o', color=INK, mfc='w', ms=5, mew=1.2, zorder=5)
        ax.axhline(1, color='k', lw=1); grey_band(ax, ref)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.03, 3); ax.set_xlim(8, 1.25e5)
        style(ax); label(ax, LETTERS[k], rf'$Z_P={z}$', x=0.66, y=0.1)
    for ax in axes[-1]:
        ax.set_xlabel(E_LABEL, fontsize=20)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$R=\sigma\,/\,[Z_P^2\,\sigma_p]$', fontsize=20)
    handles = [Line2D([], [], color=c, lw=2, label=CASE_LABELS[k]) for k, c in CASE_COLORS.items()]
    handles.append(Line2D([], [], color=INK, marker='o', mfc='w', ls='none', ms=6, label='CDW-EIS'))
    fig.legend(handles=handles, loc='outside upper center', ncol=7, frameon=False, fontsize=16)
    fig.savefig(os.path.join(FIG, 'zp2_holes_scaling_ratio.png'), dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------------ S8, S9 (Sec. S12)
EXTRA_DIR = os.path.join('supplementary', '_extrepolation_ZP')


def log_table(df, theory):
    sub = df[df['Theory_ID'] == theory]
    return np.log10(sub.pivot_table(index='E_proj[keV/u]', columns='Z_ion', values='Cross_section[cm2]').sort_index())


def edge_bias(L):
    """Bias (prediction - reference) in log10 sigma of three neighbour-based extrapolations to Z_P = 1 and 9."""
    Y = L.values; lnZ = np.log(np.arange(1, 10))

    def lin_ln(i, j, k):
        return Y[:, i] + (Y[:, i] - Y[:, j]) * (lnZ[k] - lnZ[i]) / (lnZ[i] - lnZ[j])

    return {'constant': (Y[:, 1] - Y[:, 0], Y[:, 7] - Y[:, 8]),
            'linear in Z': (2 * Y[:, 1] - Y[:, 2] - Y[:, 0], 2 * Y[:, 7] - Y[:, 6] - Y[:, 8]),
            'linear in ln Z': (lin_ln(1, 2, 0) - Y[:, 0], lin_ln(7, 6, 8) - Y[:, 8])}


def fig_s8(df):
    L = log_table(df, 'CDW-EIS'); Z = np.arange(1, 10)
    energies = [50, 100, 500, 5000, 100000]
    colors = [RUSTY, GOLD, CYAN, TEAL, INK]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True, constrained_layout=True)
    for E, col in zip(energies, colors):
        y = L.loc[E].values - L.loc[E].values[0]
        lab = f'{E / 1000:g} MeV/u' if E >= 1000 else f'{E:g} keV/u'
        for a in ax:
            a.plot(Z, y, 'o-', color=col, lw=2.5, ms=7, label=lab)
    ax[1].plot(Z, 2 * np.log10(Z), ':', color='k', lw=2, label=r'$Z_P^2$ law')
    ax[0].set_xticks(Z); ax[0].set_xlabel(r'$Z_P$ (linear axis, as seen by the network)', fontsize=20)
    ax[1].set_xscale('log'); ax[1].minorticks_off()
    ax[1].set_xticks([1, 2, 3, 4, 5, 6, 9]); ax[1].set_xticklabels(['1', '2', '3', '4', '5', '6', '9'])
    ax[1].set_xlabel(r'$Z_P$ (logarithmic axis)', fontsize=20)
    ax[0].set_ylabel(r'$\log_{10}[\sigma(Z_P)/\sigma(1)]$', fontsize=20)
    for a, letter in zip(ax, 'ab'):
        style(a); label(a, letter, x=0.04, y=0.95)
    fig.legend(*ax[1].get_legend_handles_labels(), loc='outside upper center', ncol=6, frameon=False, fontsize=16)
    fig.savefig(os.path.join(FIG, 'extrap_charge_dependence.png'), dpi=200); plt.close(fig)


def fig_s9(df):
    L = {t: log_table(df, t) for t in ('CDW-EIS', 'CTMC')}
    B = {t: edge_bias(L[t]) for t in L}
    nn = pd.read_csv(os.path.join(EXTRA_DIR, 'nn_edge_predictions.csv'))
    nn['bias'] = nn['log10_sigma_nn'] - nn['log10_sigma_ref']
    colors = {'constant': IRON, 'linear in Z': TEAL, 'linear in ln Z': GOLD}
    labels = {'constant': 'constant (copy neighbour)', 'linear in Z': r'linear in $Z_P$',
              'linear in ln Z': r'linear in $\log_{10}Z_P$'}
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True, constrained_layout=True)
    for i, hole in enumerate((1, 9)):
        for k, col in colors.items():
            ax[i].plot(L['CDW-EIS'].index, B['CDW-EIS'][k][i], '-', color=col, lw=2.5)
            ax[i].plot(L['CTMC'].index, B['CTMC'][k][i], 's', color=col, mfc='none', ms=7, mew=1.5)
        for theory, mk in (('CDW-EIS', 'o'), ('CTMC', '^')):
            g = nn[(nn['hole'] == hole) & (nn['theory'] == theory)].sort_values('E')
            ax[i].plot(g['E'], g['bias'], mk, color=INK, ms=7, mfc=INK if theory == 'CDW-EIS' else 'none', mew=1.5,
                       ls='-' if theory == 'CDW-EIS' else 'none', lw=1.2)
        ax[i].axhline(0, color='k', lw=1); ax[i].axvspan(8, 50, color='0.9', zorder=0, lw=0)
        ax[i].set_xscale('log'); ax[i].set_xlim(8, 1.25e5); ax[i].set_ylim(-0.3, 0.8)
        ax[i].set_xlabel(E_LABEL, fontsize=20); style(ax[i])
    label(ax[0], 'a', r'$Z_P=1$ from $Z_P=2,3$', x=0.45, y=0.1)
    label(ax[1], 'b', r'$Z_P=9$ from $Z_P=8,7$', x=0.45, y=0.1)
    ax[0].set_ylabel(r'Bias in $\log_{10}\sigma$ (prediction $-$ reference)', fontsize=20)
    handles = [Line2D([], [], color=c, lw=2.5, label=labels[k]) for k, c in colors.items()]
    handles += [Line2D([], [], color=INK, marker='o', ms=7, lw=1.2, label='network, CDW-EIS'),
                Line2D([], [], color=INK, marker='^', mfc='none', ms=7, mew=1.5, ls='none', label='network, CTMC'),
                Line2D([], [], color='k', lw=2.5, label='lines: CDW-EIS'),
                Line2D([], [], color='k', marker='s', mfc='none', ms=7, mew=1.5, ls='none', label='squares: CTMC')]
    ax[1].legend(handles=handles, frameon=False, fontsize=14, loc='upper right', ncol=1)
    fig.savefig(os.path.join(FIG, 'extrap_edge_bias.png'), dpi=200); plt.close(fig)


if __name__ == '__main__':
    which = set(sys.argv[1:]) or {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'}
    df = data()
    if 'S1' in which: fig_s1()
    if 'S2' in which: fig_s2(df)
    if 'S3' in which: fig_s3(df)
    if 'S4' in which: fig_s4(df)
    if 'S5' in which: fig_s5()
    if 'S6' in which: fig_s6(df)
    if 'S7' in which: fig_s7(df)
    if 'S8' in which: fig_s8(df)
    if 'S9' in which: fig_s9(df)
    print('done:', ' '.join(sorted(which)))
