"""Two-charge exclusion analysis.

Models best_statedict_gauss_128_hole_XYZ.pth whose name contains exactly two distinct
charges {a, b}. Every pair a < b has two models (aab and abb) that exclude the same charges.

Each (model, evaluated charge) is a "case". Cases are classified by:
  - charge class: interpolation / extrapolation (w.r.t. the training charge range)
  - case class: interpolation "isolated" (both neighbouring charges z±1 are in training) or
    "adjacent" (the partner is z±1, so the training gap around z is 3 charges wide);
    extrapolation d=1 / d=2 (distance to the nearest training charge)
  - pair class: interpolation / mixed / extrapolation (both charges together)
and compared with the single-charge baseline (model zzz) for the same charge.
"""
import os, glob, itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from evaluate_holes import (NAMELIST, CDW_EMIN, read_all_databases, evaluate_model, load_model,
                            summarize, is_valid, OUT_DIR, model_path, model_names, MODEL_SET)

OUT = os.path.join(OUT_DIR, 'two_charges')
CASE_ORDER = ['interp. isolated', 'interp. adjacent', 'extrap. d=1', 'extrap. d=2']
CASE_COLORS = {'interp. isolated': 'tab:blue', 'interp. adjacent': 'tab:cyan',
               'extrap. d=1': 'tab:orange', 'extrap. d=2': 'tab:red'}
PAIR_COLORS = {'interpolation': 'tab:blue', 'mixed': 'tab:purple', 'extrapolation': 'tab:red'}
# The original set has two models per pair (aab, abb); the _b set has one
ONE_MODEL_PER_PAIR = MODEL_SET.startswith('b')
E_BANDS = [(0, 50, 'E < 50'), (50, 500, '50 ≤ E < 500'), (500, np.inf, 'E ≥ 500')]


def add_errors(p):
    p = p.copy()
    p['err'] = p['pred'] - p['log_sigma']
    p['abs_err'] = p['err'].abs()
    p['sq_err'] = p['err'] ** 2
    p['rel_sigma'] = np.abs(10**p['pred'] - 10**p['log_sigma']) / 10**p['log_sigma']
    p['ratio'] = 10**p['err']
    return p


def case_metrics(p, keys):
    g = p.groupby(keys, observed=True)
    return pd.DataFrame({
        'n_points': g.size(),
        'MAE_log': g['abs_err'].mean(),
        'RMSE_log': g['sq_err'].mean() ** 0.5,
        'MBE_log': g['err'].mean(),
        'MaxAE_log': g['abs_err'].max(),
        'sigma_MRE_%': 100 * g['rel_sigma'].mean(),
        'sigma_MedRE_%': 100 * g['rel_sigma'].median(),
        'sigma_P90_%': 100 * g['rel_sigma'].quantile(0.9),
        'sigma_MaxRE_%': 100 * g['rel_sigma'].max(),
        'frac_lt5%': 100 * g['rel_sigma'].apply(lambda r: np.mean(r < 0.05)),
        'frac_lt10%': 100 * g['rel_sigma'].apply(lambda r: np.mean(r < 0.10)),
    }).reset_index()


def run_models(df, names):
    rows, pts = [], []
    for name in names:
        row, p = evaluate_model(model_path(name), df)
        rows.append(row); pts.append(p)
    return pd.DataFrame(rows), add_errors(pd.concat(pts, ignore_index=True))


def describe_cases(points):
    def attrs(r):
        a, b = sorted({int(c) for c in r['model']})
        z = r['Z_ion']
        partner = b if z == a else a
        train = [c for c in range(1, 10) if c not in (a, b)]
        dist = min(abs(z - c) for c in train)
        return pd.Series({'pair': f'{a},{b}', 'a': a, 'b': b, 'partner': partner,
                          'gap': b - a, 'adjacent': b - a == 1, 'distance': dist,
                          'variant': 'aab' if r['model'] == f'{a}{a}{b}' else 'abb'})
    keys = points[['model', 'Z_ion']].drop_duplicates()
    keys = pd.concat([keys.reset_index(drop=True), keys.apply(attrs, axis=1).reset_index(drop=True)], axis=1)
    out = points.merge(keys, on=['model', 'Z_ion'])
    out['case_class'] = np.where(out['charge_class'] == 'extrapolation',
                                 'extrap. d=' + out['distance'].astype(int).astype(str),
                                 np.where(out['adjacent'], 'interp. adjacent', 'interp. isolated'))
    out['E_band'] = pd.cut(out['E_proj[keV/u]'], [b[0] for b in E_BANDS] + [np.inf],
                           labels=[b[2] for b in E_BANDS], right=False)
    return out


def stat_tests(cases, single):
    lines = []
    def kw(groups, label):
        groups = {k: v for k, v in groups.items() if len(v) > 1}
        h, p = stats.kruskal(*groups.values())
        lines.append(f'{label}: Kruskal-Wallis H={h:.2f}, p={p:.2e} ({", ".join(f"{k}: n={len(v)}" for k, v in groups.items())})')
    def mw(x, y, label):
        u, p = stats.mannwhitneyu(x, y, alternative='two-sided')
        lines.append(f'{label}: Mann-Whitney U={u:.1f}, p={p:.2e} (medians {np.median(x):.4f} vs {np.median(y):.4f}, n={len(x)}/{len(y)})')

    kw({c: cases.loc[cases['case_class'] == c, 'MAE_log'].values for c in CASE_ORDER},
       'Case MAE_log across case classes')
    kw({c: cases.loc[cases['pair_class'] == c, 'MAE_log'].values for c in PAIR_COLORS},
       'Case MAE_log across pair classes')
    inter = cases[cases['charge_class'] == 'interpolation']
    mw(inter.loc[~inter['adjacent'], 'MAE_log'], inter.loc[inter['adjacent'], 'MAE_log'],
       'Interpolation isolated vs adjacent pair (MAE_log)')
    ext = cases[cases['charge_class'] == 'extrapolation']
    mw(ext.loc[ext['distance'] == 1, 'MAE_log'], ext.loc[ext['distance'] == 2, 'MAE_log'],
       'Extrapolation d=1 vs d=2 (MAE_log)')
    mw(inter['MAE_log'], ext['MAE_log'], 'Interpolation vs extrapolation (MAE_log)')

    # Degradation with respect to the single-charge model for the same charge
    w = stats.wilcoxon(cases['MAE_log'] - cases['single_MAE_log'])
    lines.append(f'Two-charge vs single-charge MAE_log (paired by charge): Wilcoxon W={w.statistic:.1f}, p={w.pvalue:.2e}; '
                 f'median ratio = {np.median(cases["degradation"]):.2f}, '
                 f'{100 * np.mean(cases["degradation"] > 1):.0f}% of cases worse than single')
    inner = inter[inter['Z_ion'].between(3, 8)]
    rho, p = stats.spearmanr(inner['gap'], inner['MAE_log'])
    lines.append(f'Interior charges 3-8 (interpolation): Spearman(|a-b|, MAE_log) rho={rho:.3f}, p={p:.2e}')
    rho, p = stats.spearmanr(inner['partner_dist'], inner['degradation'])
    lines.append(f'Interior charges 3-8: Spearman(|z-partner|, degradation) rho={rho:.3f}, p={p:.2e}')
    return lines


def plot_pair_matrices(cases, single):
    mean = cases.groupby(['Z_ion', 'partner'])[['MAE_log', 'sigma_MedRE_%', 'degradation']].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    specs = [('MAE_log', 'MAE [log10 σ] on evaluated charge', 'viridis', True),
             ('sigma_MedRE_%', 'Median relative error in σ [%]', 'viridis', True),
             ('degradation', 'MAE ratio: two-charge / single-charge', 'RdBu_r', False)]
    for ax, (col, title, cmap, use_log) in zip(axes, specs):
        M = np.full((9, 9), np.nan)
        for _, r in mean.iterrows():
            M[int(r['Z_ion']) - 1, int(r['partner']) - 1] = r[col]
        if col != 'degradation':
            for _, r in single.iterrows():
                M[int(r['Z_ion']) - 1, int(r['Z_ion']) - 1] = r[col]
        if use_log:
            im = ax.imshow(np.log10(M), cmap=cmap, origin='lower')
            cb = fig.colorbar(im, ax=ax); cb.set_label(f'log10({col})')
        else:
            im = ax.imshow(np.log2(M), cmap=cmap, origin='lower', vmin=-4, vmax=4)
            cb = fig.colorbar(im, ax=ax); cb.set_label('log2(ratio)')
        for i, j in itertools.product(range(9), range(9)):
            if not np.isnan(M[i, j]):
                txt = f'{M[i, j]:.3f}' if col == 'MAE_log' else f'{M[i, j]:.1f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=6.5,
                        color='w' if (col != 'degradation') else 'k')
        ax.set_xticks(range(9)); ax.set_xticklabels(range(1, 10))
        ax.set_yticks(range(9)); ax.set_yticklabels(range(1, 10))
        ax.set_xlabel('Partner charge excluded with it' + ('' if col == 'degradation' else ' (diagonal = single)'))
        ax.set_ylabel('Evaluated charge'); ax.set_title(title)
    fig.suptitle(f'Two-charge exclusion, ' + ('one model per pair' if ONE_MODEL_PER_PAIR else 'mean over the two models of each pair') + f' (valid points: no CDW-EIS E<{CDW_EMIN:g} keV/u)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pair_matrices.png'), dpi=130); plt.close(fig)


def plot_ratio_grid(points):
    pairs = sorted(points['pair'].unique(), key=lambda s: tuple(map(int, s.split(','))))
    fig, axes = plt.subplots(6, 6, figsize=(22, 19), sharex=True, sharey=True)
    for ax, pair in zip(axes.ravel(), pairs):
        a, b = map(int, pair.split(','))
        sub = points[points['pair'] == pair]
        for z, col, role in [(a, 'tab:blue', 'lower'), (b, 'tab:red', 'higher')]:
            for var, ls in [('aab', '-'), ('abb', '--')]:
                s = sub[(sub['Z_ion'] == z) & (sub['variant'] == var)]
                if s.empty:
                    continue
                cdw = s[s['Theory_ID'] == 'CDW-EIS'].sort_values('log_E')
                ctmc = s[s['Theory_ID'] == 'CTMC'].sort_values('log_E')
                ax.plot(cdw['E_proj[keV/u]'], cdw['ratio'], ls, color=col, lw=1.1,
                        label=f'{role} charge, CDW-EIS' + ('' if ONE_MODEL_PER_PAIR else f', model {var}') if pair == pairs[0] else None)
                ax.plot(ctmc['E_proj[keV/u]'], ctmc['ratio'], 's' if var == 'aab' else 'x', color=col, ms=3, mfc='none',
                        label=f'{role} charge, CTMC' + ('' if ONE_MODEL_PER_PAIR else f', model {var}') if pair == pairs[0] else None)
        ax.axhline(1, color='k', lw=0.7); ax.axhspan(0.9, 1.1, color='0.85', zorder=0)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.25, 4)
        ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(['0.25', '0.5', '1', '2', '4'])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        cls = sub['pair_class'].iloc[0]
        ax.set_title(f'({a}, {b}) – {cls}', color=PAIR_COLORS[cls], fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('E [keV/u]')
    for ax in axes[:, 0]:
        ax.set_ylabel('σ_NN / σ_ref')
    fig.suptitle('Two-charge exclusion (a, b): prediction ratio on both excluded charges (grey band = ±10%)',
                 y=0.995, fontsize=14)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(OUT, 'ratio_grid_all_pairs.png'), dpi=110); plt.close(fig)


def plot_selected_curves(df):
    selected = [(4, 5), (3, 7), (2, 3), (1, 2), (8, 9), (1, 9)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True)
    for ax, (a, b) in zip(axes.ravel(), selected):
        for z, col in [(a, 'tab:blue'), (b, 'tab:red')]:
            for ls, name in [('-', f'{a}{a}{b}'), ('--', f'{a}{b}{b}')]:
                if not os.path.exists(model_path(name)):
                    continue
                m = load_model(model_path(name), df)
                for theory, lw in [('CDW-EIS', 1.4), ('CTMC', 0.9)]:
                    ref = df[(df['Z_ion'] == z) & (df['Theory_ID'] == theory)]
                    grid = pd.DataFrame({'Z_ion': z, 'Theory_ID': theory,
                                         'log_E': np.linspace(ref['log_E'].min(), ref['log_E'].max(), 200)})
                    ax.plot(10**grid['log_E'], 10**m['predict'](grid), ls, color=col, lw=lw,
                            alpha=1 if theory == 'CDW-EIS' else 0.6)
            for theory, mk in [('CDW-EIS', 'o'), ('CTMC', 's')]:
                ref = df[(df['Z_ion'] == z) & (df['Theory_ID'] == theory)]
                ax.plot(ref['E_proj[keV/u]'], ref['Cross_section[cm2]'], mk, color=col, mfc='none', ms=4,
                        label=f'Z={z} {theory} ref.')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-19, 1e-14)
        ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.6); ax.legend(fontsize=7)
        ax.set_title(f'Excluded ({a}, {b})')
    for ax in axes[-1]:
        ax.set_xlabel('E [keV/u]')
    for ax in axes[:, 0]:
        ax.set_ylabel('σ [cm²]')
    fig.suptitle('Selected pairs: NN (' + ('' if ONE_MODEL_PER_PAIR else 'solid = model aab, dashed = model abb; ') + 'thick = CDW-EIS, thin = CTMC) vs reference')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'curves_selected_pairs.png'), dpi=130); plt.close(fig)


def plot_case_classes(cases):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    present = [c for c in CASE_ORDER if c in cases['case_class'].values]
    for k, (col, lab) in enumerate([('MAE_log', 'MAE [log10 σ]'), ('sigma_MedRE_%', 'Median relative error in σ [%]'),
                                    ('MBE_log', 'MBE [log10 σ]')]):
        data = [cases.loc[cases['case_class'] == c, col] for c in present]
        bp = ax[k].boxplot(data, tick_labels=present, patch_artist=True)
        for patch, c in zip(bp['boxes'], present):
            patch.set_facecolor(CASE_COLORS[c]); patch.set_alpha(0.5)
        for i, d in enumerate(data):
            ax[k].scatter(np.random.default_rng(0).normal(i + 1, 0.05, len(d)), d, s=10, color='k', alpha=0.6, zorder=3)
        if col != 'MBE_log':
            ax[k].set_yscale('log')
        else:
            ax[k].axhline(0, color='k', lw=0.8)
        ax[k].set_ylabel(lab)
    ax[1].set_title('Per-case error by class (d = distance to nearest training charge)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'box_by_case_class.png'), dpi=130); plt.close(fig)


def plot_per_charge(cases, single):
    fig, ax = plt.subplots(figsize=(11, 5))
    for z in range(1, 10):
        sub = cases[cases['Z_ion'] == z]
        x = z + np.random.default_rng(z).normal(0, 0.08, len(sub))
        ax.scatter(x, sub['MAE_log'], c=[CASE_COLORS[c] for c in sub['case_class']], s=22, alpha=0.8,
                   edgecolor='k', linewidth=0.3)
    s = single.set_index('Z_ion')['MAE_log']
    ax.plot(s.index, s.values, 'k*-', ms=11, label='single-charge baseline (zzz)')
    for c in CASE_ORDER:
        ax.scatter([], [], color=CASE_COLORS[c], label=c)
    ax.set_yscale('log'); ax.set_xticks(range(1, 10)); ax.set_xlabel('Evaluated (excluded) charge')
    ax.set_ylabel('MAE [log10 σ]'); ax.legend(fontsize=8)
    ax.set_title('Two-charge exclusion: error on each excluded charge (each dot = one model)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'error_per_charge_vs_single.png'), dpi=130); plt.close(fig)


def plot_gap(cases):
    inner = cases[(cases['charge_class'] == 'interpolation')]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for z in sorted(inner['Z_ion'].unique()):
        sub = inner[inner['Z_ion'] == z].groupby('partner_dist')['degradation'].mean()
        ax[0].plot(sub.index, sub.values, 'o-', label=f'Z={z}', color=plt.get_cmap('viridis', 9)(z - 1))
    ax[0].axhline(1, color='k', lw=0.8); ax[0].set_yscale('log')
    ax[0].set_xlabel('|evaluated charge − partner charge|'); ax[0].set_ylabel('MAE ratio two-charge / single')
    ax[0].set_title('Interpolation cases: degradation vs distance to the partner'); ax[0].legend(ncol=2, fontsize=8)
    g = cases.groupby('pair')[['MAE_log']].mean().join(cases.groupby('pair')[['gap', 'pair_class']].first())
    for cls, col in PAIR_COLORS.items():
        s = g[g['pair_class'] == cls]
        ax[1].scatter(s['gap'], s['MAE_log'], color=col, label=cls, s=30)
        for name, r in s.iterrows():
            ax[1].annotate(name, (r['gap'], r['MAE_log']), fontsize=6.5, xytext=(3, 2), textcoords='offset points')
    ax[1].set_yscale('log'); ax[1].set_xlabel('|a − b|'); ax[1].set_ylabel('Mean MAE over both charges [log10 σ]')
    ax[1].set_title('Pair-level error vs separation of the excluded charges'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'error_vs_separation.png'), dpi=130); plt.close(fig)


def plot_energy(points):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    bins = np.linspace(points['log_E'].min(), points['log_E'].max(), 15)
    for c in CASE_ORDER:
        sub = points[points['case_class'] == c]
        if sub.empty:
            continue
        g = sub.groupby(pd.cut(sub['log_E'], bins, include_lowest=True), observed=True)
        mid = [iv.mid for iv in g.size().index]
        signed = 100 * (g['ratio'].median() - 1)
        ax[0].plot(mid, signed.values, 'o-', ms=3, color=CASE_COLORS[c], label=c)
        q1, q3 = 100 * (g['ratio'].quantile(0.25) - 1), 100 * (g['ratio'].quantile(0.75) - 1)
        ax[0].fill_between(mid, q1.values, q3.values, color=CASE_COLORS[c], alpha=0.15)
        ax[1].plot(mid, 100 * g['rel_sigma'].median().values, 'o-', ms=3, color=CASE_COLORS[c], label=c)
    ax[0].axhline(0, color='k', lw=0.8); ax[0].set_yscale('symlog', linthresh=10)
    ax[0].set_ylabel('Median signed error (σ_NN/σ_ref − 1) [%], IQR band'); ax[0].set_title('Bias vs energy')
    ax[1].set_yscale('log'); ax[1].set_ylabel('Median |relative error| in σ [%]'); ax[1].set_title('Error magnitude vs energy')
    for a in ax:
        a.set_xlabel('log10 E [keV/u]'); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'error_vs_energy_by_case_class.png'), dpi=130); plt.close(fig)




if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    df = read_all_databases(NAMELIST)
    df = df[df['Z_target'] == 1].reset_index(drop=True)
    df['log_E'] = np.log10(df['E_proj[keV/u]'].values)
    df['log_sigma'] = np.log10(df['Cross_section[cm2]'].values)

    pair_names = model_names(2)
    models, points_all = run_models(df, pair_names)
    _, single_all = run_models(df, [f'{z}{z}{z}' for z in range(1, 10)])
    points_all = describe_cases(points_all)
    points_all['pair_class'] = points_all['model_class']
    points = points_all[is_valid(points_all)]
    single = case_metrics(single_all[is_valid(single_all)], ['Z_ion'])

    # Case-level metrics (valid points, and all points for reference)
    case_keys = ['model', 'pair', 'variant', 'Z_ion', 'partner', 'gap', 'adjacent', 'distance',
                 'charge_class', 'case_class', 'pair_class']
    cases = case_metrics(points, case_keys)
    cases_all = case_metrics(points_all, case_keys)
    cases = cases.merge(single[['Z_ion', 'MAE_log', 'sigma_MedRE_%']].rename(
        columns={'MAE_log': 'single_MAE_log', 'sigma_MedRE_%': 'single_sigma_MedRE_%'}), on='Z_ion')
    cases['degradation'] = cases['MAE_log'] / cases['single_MAE_log']
    cases['partner_dist'] = (cases['Z_ion'] - cases['partner']).abs()

    # Pair-level and grouped summaries
    pair_tab = case_metrics(points, ['pair', 'pair_class', 'gap'])
    tables = {
        'cases': cases,
        'cases_all_points': cases_all,
        'pairs': pair_tab,
        'by_case_class': case_metrics(points, ['case_class']),
        'by_case_class_all_points': case_metrics(points_all, ['case_class']),
        'by_pair_class': case_metrics(points, ['pair_class']),
        'by_charge_and_case_class': case_metrics(points, ['Z_ion', 'case_class']),
        'by_case_class_and_theory': case_metrics(points, ['case_class', 'Theory_ID']),
        'by_case_class_and_energy_band': case_metrics(points, ['case_class', 'E_band']),
        'by_adjacency_interpolation': case_metrics(points[points['charge_class'] == 'interpolation'], ['adjacent']),
        'single_charge_baseline': single,
    }
    for name, t in tables.items():
        t.to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
    models.to_csv(os.path.join(OUT, 'metrics_per_model.csv'), index=False)
    points_all.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)

    # Per-case statistics of the per-case metrics
    desc = cases.groupby('case_class')[['MAE_log', 'RMSE_log', 'MBE_log', 'MaxAE_log', 'sigma_MedRE_%',
                                        'sigma_P90_%', 'frac_lt10%', 'degradation']].describe().T
    desc.to_csv(os.path.join(OUT, 'case_statistics_by_case_class.csv'))

    tests = stat_tests(cases, single)
    with open(os.path.join(OUT, 'statistical_tests.txt'), 'w') as fp:
        fp.write('\n'.join(tests) + '\n')

    plot_pair_matrices(cases, single)
    plot_ratio_grid(points_all)
    plot_selected_curves(df)
    plot_case_classes(cases)
    plot_per_charge(cases, single)
    plot_gap(cases)
    plot_energy(points)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print('Models:', len(models), '| pairs:', points['pair'].nunique(), '| cases:', len(cases))
    print(models['class'].value_counts().to_string())
    for name in ['by_case_class', 'by_case_class_all_points', 'by_pair_class', 'by_charge_and_case_class',
                 'by_case_class_and_theory', 'by_case_class_and_energy_band', 'by_adjacency_interpolation',
                 'single_charge_baseline']:
        print(f'\n=== {name} ===\n', tables[name].round(4).to_string(index=False))
    print('\n=== pairs (mean over both models and both charges) ===\n',
          pair_tab.sort_values('MAE_log').round(4).to_string(index=False))
    print('\n=== cases ===\n', cases.sort_values(['pair', 'Z_ion', 'variant'])[
        ['pair', 'variant', 'Z_ion', 'case_class', 'MAE_log', 'MBE_log', 'sigma_MedRE_%', 'sigma_P90_%',
         'degradation']].round(3).to_string(index=False))
    print('\n=== case statistics by case class ===\n', desc.round(4).to_string())
    print('\n=== statistical tests ===\n' + '\n'.join(tests))
