"""Three-charge exclusion analysis.

Models best_statedict_gauss_128_hole_abc.pth with three distinct charges a < b < c (84 models).
Each (model, evaluated excluded charge) is a "case" (252 cases), classified by the local
structure of the training charges around the evaluated charge z:
  - interpolation, local gap width 1 (isolated: z-1 and z+1 are training charges),
    gap 2 (one neighbour also excluded), gap 3 edge (d=1) or gap 3 centre (d=2: z-1 and z+1 excluded)
  - extrapolation d=1 / d=2 / d=3 (distance to the nearest training charge)
and compared with the single-charge (zzz) and two-charge models that exclude the same charge.
"""
import os, glob, itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
import matplotlib.patches
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from evaluate_holes import (NAMELIST, CDW_EMIN, read_all_databases, load_model, is_valid,
                            OUT_DIR, model_path, model_names)
from evaluate_two_charges import add_errors, case_metrics, run_models

OUT = os.path.join(OUT_DIR, 'three_charges')
CASE_ORDER = ['interp. gap 1', 'interp. gap 2', 'interp. gap 3 edge', 'interp. gap 3 centre',
              'extrap. d=1', 'extrap. d=2', 'extrap. d=3']
CASE_COLORS = dict(zip(CASE_ORDER, ['tab:blue', 'tab:cyan', 'tab:green', 'olive',
                                    'tab:orange', 'tab:red', 'darkred']))
MODEL_COLORS = {'interpolation': 'tab:blue', 'mixed': 'tab:purple', 'extrapolation': 'tab:red'}
E_BANDS = [(0, 50, 'E < 50'), (50, 500, '50 ≤ E < 500'), (500, np.inf, 'E ≥ 500')]


def case_attributes(model, z):
    excluded = sorted({int(c) for c in model})
    train = [c for c in range(1, 10) if c not in excluded]
    d = min(abs(z - t) for t in train)
    if z < min(train) or z > max(train):
        cls, gap = f'extrap. d={d}', 0  # 0 = not applicable (groupby drops NaN keys)
    else:
        gap = min(t for t in train if t > z) - max(t for t in train if t < z) - 1
        cls = f'interp. gap {gap}' + ('' if gap < 3 else (' edge' if d == 1 else ' centre'))
    partners = [c for c in excluded if c != z]
    return {'n_excluded': len(excluded), 'partners': ','.join(map(str, partners)),
            'distance': d, 'local_gap': gap, 'case_class': cls,
            'min_partner_dist': min((abs(z - p) for p in partners), default=0)}


def describe(points):
    keys = points[['model', 'Z_ion']].drop_duplicates()
    attrs = pd.DataFrame([case_attributes(m, z) for m, z in keys.values], index=keys.index)
    keys = pd.concat([keys, attrs], axis=1)
    out = points.drop(columns=['n_excluded']).merge(keys, on=['model', 'Z_ion'])
    out['E_band'] = pd.cut(out['E_proj[keV/u]'], [b[0] for b in E_BANDS] + [np.inf],
                           labels=[b[2] for b in E_BANDS], right=False)
    return out


def stat_tests(cases, prog):
    lines = []
    def kw(groups, label):
        groups = {k: v for k, v in groups.items() if len(v) > 1}
        h, p = stats.kruskal(*groups.values())
        lines.append(f'{label}: Kruskal-Wallis H={h:.2f}, p={p:.2e} ({", ".join(f"{k}: n={len(v)}" for k, v in groups.items())})')
    def mw(x, y, label):
        u, p = stats.mannwhitneyu(x, y, alternative='two-sided')
        lines.append(f'{label}: Mann-Whitney U={u:.1f}, p={p:.2e} (medians {np.median(x):.4f} vs {np.median(y):.4f}, n={len(x)}/{len(y)})')
    def c(cls):
        return cases.loc[cases['case_class'] == cls, 'MAE_log']

    kw({k: c(k).values for k in CASE_ORDER}, 'Case MAE_log across case classes')
    kw({k: cases.loc[cases['model_class'] == k, 'MAE_log'].values for k in MODEL_COLORS},
       'Case MAE_log across model classes')
    inter = cases[cases['charge_class'] == 'interpolation']
    ext = cases[cases['charge_class'] == 'extrapolation']
    mw(inter['MAE_log'], ext['MAE_log'], 'Interpolation vs extrapolation')
    mw(c('interp. gap 1'), c('interp. gap 2'), 'Interp. gap 1 vs gap 2')
    mw(c('interp. gap 2'), pd.concat([c('interp. gap 3 edge'), c('interp. gap 3 centre')]), 'Interp. gap 2 vs gap 3')
    mw(c('interp. gap 3 edge'), c('interp. gap 3 centre'), 'Interp. gap 3 edge vs centre')
    rho, p = stats.spearmanr(inter['local_gap'], inter['MAE_log'])
    lines.append(f'Interpolation: Spearman(local gap width, MAE_log) rho={rho:.3f}, p={p:.2e}')
    inner = inter[inter['Z_ion'].between(3, 8)]
    rho, p = stats.spearmanr(inner['local_gap'], inner['MAE_log'])
    lines.append(f'Interpolation, charges 3-8: Spearman(local gap width, MAE_log) rho={rho:.3f}, p={p:.2e}')
    rho, p = stats.spearmanr(ext['distance'], ext['MAE_log'])
    lines.append(f'Extrapolation: Spearman(distance d, MAE_log) rho={rho:.3f}, p={p:.2e}')

    for base, label in [('single_MAE_log', 'single-charge'), ('two_MAE_log', 'two-charge (mean of models sharing z and one partner)')]:
        diff = cases['MAE_log'] - cases[base]
        w = stats.wilcoxon(diff)
        ratio = cases['MAE_log'] / cases[base]
        lines.append(f'Three-charge vs {label} MAE_log (paired): Wilcoxon p={w.pvalue:.2e}; median ratio={np.median(ratio):.2f}, '
                     f'{100 * np.mean(ratio > 1):.0f}% of cases worse')
    # Progression 1 -> 2 -> 3 excluded charges for interior isolated interpolation (gap 1 only)
    g = prog[(prog['case_class'] == 'interp. gap 1') & prog['Z_ion'].between(3, 7)]
    kw({n: g.loc[g['n_excluded'] == n, 'MAE_log'].values for n in [1, 2, 3]},
       'Isolated interpolation, Z=3-7: MAE_log for 1 / 2 / 3 excluded charges')
    rho, p = stats.spearmanr(g['n_excluded'], g['MAE_log'])
    lines.append(f'  Spearman(n_excluded, MAE_log) rho={rho:.3f}, p={p:.2e}')
    return lines


def plot_partner_matrices(cases):
    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    vmin, vmax = np.log10(cases['MAE_log'].min()), np.log10(cases['MAE_log'].max())
    for ax, z in zip(axes.ravel(), range(1, 10)):
        M = np.full((9, 9), np.nan)
        for _, r in cases[cases['Z_ion'] == z].iterrows():
            p1, p2 = map(int, r['partners'].split(','))
            M[p2 - 1, p1 - 1] = r['MAE_log']
        im = ax.imshow(np.log10(M), cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        for i, j in itertools.product(range(9), range(9)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=6, color='w')
        ax.set_xticks(range(9)); ax.set_xticklabels(range(1, 10)); ax.set_yticks(range(9)); ax.set_yticklabels(range(1, 10))
        ax.set_xlabel('Lower partner charge'); ax.set_ylabel('Higher partner charge')
        ax.set_title(f'Evaluated charge Z = {z}')
        ax.axvline(z - 1, color='w', lw=0.5, alpha=0.4); ax.axhline(z - 1, color='w', lw=0.5, alpha=0.4)
    cb = fig.colorbar(im, ax=axes, shrink=0.6); cb.set_label('log10(MAE [log10 σ]) on the evaluated charge')
    fig.suptitle(f'Three-charge exclusion: error on charge Z depending on the two other excluded charges\n'
                 f'(valid points: no CDW-EIS E<{CDW_EMIN:g} keV/u)', fontsize=14)
    fig.savefig(os.path.join(OUT, 'partner_matrices.png'), dpi=110, bbox_inches='tight'); plt.close(fig)


def plot_ratio_grid(points):
    models = sorted(points['model'].unique())
    ncol = 7
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(24, 3.0 * nrow), sharex=True, sharey=True)
    colors = ['tab:blue', 'tab:green', 'tab:red']
    for ax, m in zip(axes.ravel(), models):
        sub = points[points['model'] == m]
        for z, col, role in zip(sorted({int(c) for c in m}), colors, ['lowest', 'middle', 'highest']):
            s = sub[sub['Z_ion'] == z]
            cdw = s[s['Theory_ID'] == 'CDW-EIS'].sort_values('log_E')
            ctmc = s[s['Theory_ID'] == 'CTMC'].sort_values('log_E')
            ax.plot(cdw['E_proj[keV/u]'], cdw['ratio'], '-', color=col, lw=1.1,
                    label=f'{role} charge, CDW-EIS' if m == models[0] else None)
            ax.plot(ctmc['E_proj[keV/u]'], ctmc['ratio'], 's', color=col, ms=2.5, mfc='none',
                    label=f'{role} charge, CTMC' if m == models[0] else None)
        ax.axhline(1, color='k', lw=0.7); ax.axhspan(0.9, 1.1, color='0.85', zorder=0)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.25, 4)
        ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(['0.25', '0.5', '1', '2', '4'])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        cls = sub['model_class'].iloc[0]
        ax.set_title(f'({", ".join(m)}) – {cls}', color=MODEL_COLORS[cls], fontsize=9)
    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('E [keV/u]')
    for ax in axes[:, 0]:
        ax.set_ylabel('σ_NN / σ_ref')
    fig.suptitle('Three-charge exclusion: prediction ratio on the three excluded charges (grey band = ±10%)',
                 y=0.998, fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(os.path.join(OUT, 'ratio_grid_all_models.png'), dpi=90); plt.close(fig)


def plot_selected_curves(df):
    selected = ['357', '456', '234', '123', '789', '159']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True)
    colors = ['tab:blue', 'tab:green', 'tab:red']
    for ax, name in zip(axes.ravel(), selected):
        m = load_model(model_path(name), df)
        for z, col in zip(m['excluded'], colors):
            for theory, mk, lw in [('CDW-EIS', 'o', 1.5), ('CTMC', 's', 0.9)]:
                ref = df[(df['Z_ion'] == z) & (df['Theory_ID'] == theory)]
                grid = pd.DataFrame({'Z_ion': z, 'Theory_ID': theory,
                                     'log_E': np.linspace(ref['log_E'].min(), ref['log_E'].max(), 200)})
                ax.plot(10**grid['log_E'], 10**m['predict'](grid), '-' if theory == 'CDW-EIS' else '--',
                        color=col, lw=lw)
                ax.plot(ref['E_proj[keV/u]'], ref['Cross_section[cm2]'], mk, color=col, mfc='none', ms=4,
                        label=f'Z={z} {theory} ref.')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-19, 1e-14)
        ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.6); ax.legend(fontsize=7, ncol=2)
        ax.set_title(f'Excluded ({", ".join(name)})')
    for ax in axes[-1]:
        ax.set_xlabel('E [keV/u]')
    for ax in axes[:, 0]:
        ax.set_ylabel('σ [cm²]')
    fig.suptitle('Selected triples: NN (solid = CDW-EIS, dashed = CTMC) vs reference (markers)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'curves_selected_triples.png'), dpi=130); plt.close(fig)


def plot_case_classes(cases):
    present = [c for c in CASE_ORDER if c in cases['case_class'].values]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    for k, (col, lab) in enumerate([('MAE_log', 'MAE [log10 σ]'), ('sigma_MedRE_%', 'Median relative error in σ [%]'),
                                    ('MBE_log', 'MBE [log10 σ]')]):
        data = [cases.loc[cases['case_class'] == c, col] for c in present]
        bp = ax[k].boxplot(data, tick_labels=present, patch_artist=True)
        for patch, c in zip(bp['boxes'], present):
            patch.set_facecolor(CASE_COLORS[c]); patch.set_alpha(0.5)
        rng = np.random.default_rng(0)
        for i, d in enumerate(data):
            ax[k].scatter(rng.normal(i + 1, 0.05, len(d)), d, s=8, color='k', alpha=0.5, zorder=3)
        if col == 'MBE_log':
            ax[k].axhline(0, color='k', lw=0.8)
        else:
            ax[k].set_yscale('log')
        ax[k].set_ylabel(lab); ax[k].tick_params(axis='x', rotation=35)
    ax[1].set_title('Per-case error by class (gap = width of the missing-charge run around z; d = distance to nearest training charge)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'box_by_case_class.png'), dpi=130); plt.close(fig)


def plot_progression(prog):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    offsets = {1: -0.25, 2: 0.0, 3: 0.25}
    markers = {1: '*', 2: 's', 3: 'o'}
    rng = np.random.default_rng(1)
    for n in [1, 2, 3]:
        sub = prog[prog['n_excluded'] == n]
        x = sub['Z_ion'] + offsets[n] + rng.normal(0, 0.03 if n > 1 else 0, len(sub))
        ax.scatter(x, sub['MAE_log'], c=[CASE_COLORS.get(c, 'k') for c in sub['case_class']],
                   marker=markers[n], s=60 if n == 1 else 14, alpha=0.75, edgecolor='k', linewidth=0.3)
        med = sub.groupby('Z_ion')['MAE_log'].median()
        ax.plot(med.index + offsets[n], med.values, '_', color='k', ms=18, mew=2)
    for n, lab in [(1, '1 excluded charge (zzz)'), (2, '2 excluded charges'), (3, '3 excluded charges')]:
        ax.scatter([], [], marker=markers[n], color='0.5', label=lab)
    for c in CASE_ORDER:
        ax.scatter([], [], marker='o', color=CASE_COLORS[c], label=c)
    ax.plot([], [], '_', color='k', ms=18, mew=2, label='median per charge and n')
    ax.set_yscale('log'); ax.set_xticks(range(1, 10)); ax.set_xlabel('Evaluated (excluded) charge')
    ax.set_ylabel('MAE [log10 σ]'); ax.legend(fontsize=7.5, ncol=2, loc='upper center')
    ax.set_title('Error on each charge for 1, 2 and 3 excluded charges (left/centre/right of each tick)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'error_progression_1_2_3.png'), dpi=130); plt.close(fig)


def plot_vs_baselines(cases):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for k, (base, lab) in enumerate([('single_MAE_log', 'single-charge model (zzz)'),
                                     ('two_MAE_log', 'two-charge models (z with one of the partners)')]):
        for c in CASE_ORDER:
            s = cases[cases['case_class'] == c]
            ax[k].scatter(s[base], s['MAE_log'], s=16, color=CASE_COLORS[c], label=c, alpha=0.8)
        lim = [cases[['MAE_log', base]].min().min() * 0.8, cases[['MAE_log', base]].max().max() * 1.2]
        ax[k].plot(lim, lim, 'k-', lw=0.8)
        ax[k].plot(lim, [2 * l for l in lim], 'k:', lw=0.7)
        ax[k].set_xscale('log'); ax[k].set_yscale('log'); ax[k].set_xlim(lim); ax[k].set_ylim(lim)
        ax[k].set_xlabel(f'MAE of the {lab} [log10 σ]'); ax[k].set_ylabel('MAE three-charge [log10 σ]')
    ax[0].set_title('Three-charge vs single-charge (dotted = 2×)'); ax[1].set_title('Three-charge vs two-charge (dotted = 2×)')
    ax[0].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'three_vs_one_and_two.png'), dpi=130); plt.close(fig)


def plot_energy(points):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(points['log_E'].min(), points['log_E'].max(), 15)
    for c in CASE_ORDER:
        sub = points[points['case_class'] == c]
        if sub.empty:
            continue
        g = sub.groupby(pd.cut(sub['log_E'], bins, include_lowest=True), observed=True)
        mid = [iv.mid for iv in g.size().index]
        ax[0].plot(mid, 100 * (g['ratio'].median() - 1).values, 'o-', ms=3, color=CASE_COLORS[c], label=c)
        ax[1].plot(mid, 100 * g['rel_sigma'].median().values, 'o-', ms=3, color=CASE_COLORS[c], label=c)
    ax[0].axhline(0, color='k', lw=0.8); ax[0].set_yscale('symlog', linthresh=10)
    ax[0].set_ylabel('Median signed error (σ_NN/σ_ref − 1) [%]'); ax[0].set_title('Bias vs energy')
    ax[1].set_yscale('log'); ax[1].set_ylabel('Median |relative error| in σ [%]'); ax[1].set_title('Error magnitude vs energy')
    for a in ax:
        a.set_xlabel('log10 E [keV/u]'); a.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'error_vs_energy_by_case_class.png'), dpi=130); plt.close(fig)


def plot_models(model_tab):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, gridspec_kw={'width_ratios': [2, 1]})
    s = model_tab.sort_values('MAE_log').reset_index(drop=True)
    next_colors = {0: 'tab:blue', 1: 'tab:purple', 2: 'tab:orange', 3: 'tab:red'}
    pos = np.arange(len(s))
    ax[0].bar(pos, s['MAE_log'], width=0.8, color=[next_colors[n] for n in s['n_extrapolated']])
    # Excluded charges of each network, staggered in two rows so the labels can be large enough to read
    ax[0].set_xticks(pos); ax[0].set_xticklabels([','.join(str(m)) for m in s['model']], rotation=90, fontsize=9)
    ax[0].set_xlim(-1, len(s))
    for k, tick in enumerate(ax[0].xaxis.get_major_ticks()):
        tick.set_pad(2 if k % 2 == 0 else 36)
        if k % 2:  # leader line from the axis to the lower row of labels
            ax[0].annotate('', xy=(k, 0), xycoords=ax[0].get_xaxis_transform(), xytext=(0, -38),
                           textcoords='offset points', annotation_clip=False,
                           arrowprops=dict(arrowstyle='-', lw=0.5, color='0.45', shrinkA=0, shrinkB=0))
    ax[0].set_yscale('log'); ax[0].set_ylabel('Network MAE [log10 σ]', fontsize=12)
    ax[0].set_xlabel('Excluded charges of the network, sorted by MAE', fontsize=12)
    ax[0].set_title('Network ranking (valid points)')
    ax[0].legend(handles=[matplotlib.patches.Patch(color=col, label=f'{n} extrapolated') for n, col in next_colors.items()],
                 fontsize=10)
    rng = np.random.default_rng(2)
    for n in sorted(model_tab['n_extrapolated'].unique()):
        sub = model_tab[model_tab['n_extrapolated'] == n]
        ax[1].scatter(n + rng.normal(0, 0.06, len(sub)), sub['MAE_log'], s=20,
                      color=next_colors[n])
        ax[1].plot(n, sub['MAE_log'].median(), '_', color='k', ms=25, mew=2)
    ax[1].set_yscale('log'); ax[1].set_xticks(sorted(model_tab['n_extrapolated'].unique()))
    ax[1].set_xlabel('Extrapolated charges in the network', fontsize=12)
    ax[1].set_title('Error vs extrapolated charges')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'model_ranking.png'), dpi=200); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    df = read_all_databases(NAMELIST)
    df = df[df['Z_target'] == 1].reset_index(drop=True)
    df['log_E'] = np.log10(df['E_proj[keV/u]'].values)
    df['log_sigma'] = np.log10(df['Cross_section[cm2]'].values)

    models, pts3_all = run_models(df, model_names(3))
    _, pts2_all = run_models(df, model_names(2))
    _, pts1_all = run_models(df, [f'{z}{z}{z}' for z in range(1, 10)])
    pts3_all, pts2_all, pts1_all = describe(pts3_all), describe(pts2_all), describe(pts1_all)
    pts3 = pts3_all[is_valid(pts3_all)]

    keys = ['model', 'Z_ion', 'partners', 'n_excluded', 'distance', 'local_gap', 'charge_class',
            'case_class', 'model_class', 'min_partner_dist']
    cases = case_metrics(pts3, keys)
    cases_all = case_metrics(pts3_all, keys)
    c2 = case_metrics(pts2_all[is_valid(pts2_all)], keys)
    c1 = case_metrics(pts1_all[is_valid(pts1_all)], keys)

    # Baselines for the same evaluated charge
    cases = cases.merge(c1[['Z_ion', 'MAE_log']].rename(columns={'MAE_log': 'single_MAE_log'}), on='Z_ion')
    def two_baseline(r):
        ps = [int(p) for p in r['partners'].split(',')]
        sel = c2[(c2['Z_ion'] == r['Z_ion']) & c2['partners'].astype(int).isin(ps)]
        return sel['MAE_log'].mean()
    cases['two_MAE_log'] = cases.apply(two_baseline, axis=1)
    cases['ratio_vs_single'] = cases['MAE_log'] / cases['single_MAE_log']
    cases['ratio_vs_two'] = cases['MAE_log'] / cases['two_MAE_log']

    prog = pd.concat([c1, c2, cases[c1.columns]], ignore_index=True)
    model_tab = case_metrics(pts3, ['model', 'model_class'])
    n_ext = pts3.groupby('model').apply(lambda g: g.loc[g['charge_class'] == 'extrapolation', 'Z_ion'].nunique(),
                                        include_groups=False).rename('n_extrapolated')
    model_tab = model_tab.merge(n_ext, on='model')
    models = models.merge(n_ext, on='model')

    tables = {
        'cases': cases,
        'cases_all_points': cases_all,
        'models': model_tab,
        'by_case_class': case_metrics(pts3, ['case_class']),
        'by_case_class_all_points': case_metrics(pts3_all, ['case_class']),
        'by_model_class': case_metrics(pts3, ['model_class']),
        'by_n_extrapolated': case_metrics(pts3.merge(n_ext, on='model'), ['n_extrapolated']),
        'by_charge_and_case_class': case_metrics(pts3, ['Z_ion', 'case_class']),
        'by_case_class_and_theory': case_metrics(pts3, ['case_class', 'Theory_ID']),
        'by_case_class_and_energy_band': case_metrics(pts3, ['case_class', 'E_band']),
        'progression_by_n_excluded_and_case_class': prog.groupby(['n_excluded', 'case_class'])[
            ['MAE_log', 'MBE_log', 'sigma_MedRE_%', 'sigma_P90_%']].median().join(
            prog.groupby(['n_excluded', 'case_class']).size().rename('n_cases')).reset_index(),
        'progression_by_charge': prog.pivot_table(index='Z_ion', columns='n_excluded', values='MAE_log',
                                                  aggfunc='median').add_prefix('median_MAE_log_n').reset_index(),
    }
    for name, t in tables.items():
        t.to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
    models.to_csv(os.path.join(OUT, 'metrics_per_model.csv'), index=False)
    pts3_all.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
    desc = cases.groupby('case_class')[['MAE_log', 'RMSE_log', 'MBE_log', 'MaxAE_log', 'sigma_MedRE_%',
                                        'sigma_P90_%', 'frac_lt10%', 'ratio_vs_single', 'ratio_vs_two']].describe().T
    desc.to_csv(os.path.join(OUT, 'case_statistics_by_case_class.csv'))

    tests = stat_tests(cases, prog)
    with open(os.path.join(OUT, 'statistical_tests.txt'), 'w') as fp:
        fp.write('\n'.join(tests) + '\n')

    plot_partner_matrices(cases)
    plot_ratio_grid(pts3_all)
    plot_selected_curves(df)
    plot_case_classes(cases)
    plot_progression(prog)
    plot_vs_baselines(cases)
    plot_energy(pts3)
    plot_models(model_tab)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print('Models:', len(models), '| cases:', len(cases))
    print(models['class'].value_counts().to_string())
    print(cases['case_class'].value_counts().to_string())
    for name in ['by_case_class', 'by_case_class_all_points', 'by_model_class', 'by_n_extrapolated',
                 'by_charge_and_case_class', 'by_case_class_and_theory', 'by_case_class_and_energy_band',
                 'progression_by_n_excluded_and_case_class', 'progression_by_charge']:
        print(f'\n=== {name} ===\n', tables[name].round(4).to_string(index=False))
    print('\n=== models (sorted) ===\n', model_tab.sort_values('MAE_log')[
        ['model', 'model_class', 'n_extrapolated', 'MAE_log', 'MBE_log', 'sigma_MedRE_%', 'sigma_P90_%']].round(4).to_string(index=False))
    print('\n=== case statistics by case class ===\n', desc.round(4).to_string())
    print('\n=== statistical tests ===\n' + '\n'.join(tests))
