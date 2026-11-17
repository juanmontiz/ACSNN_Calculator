"""CDW-EIS errors as a function of the Sommerfeld parameter eta = Z_P / v.

CDW-EIS is a perturbative (high-velocity) approximation, reliable when eta = Z_P / v < 1, with v the projectile
velocity in atomic units. With E in keV/u, v = sqrt(E / E_au), E_au = m_u c^2 alpha^2 / 2 = 24.80 keV/u, so
eta < 1  <=>  E > 24.80 Z_P^2 keV/u (25 keV/u for Z_P = 1, 2.0 MeV/u for Z_P = 9).

The CDW-EIS points are split into two regimes:
  - 'eta < 1'           : perturbative regime (all energies above 24.80 Z_P^2 keV/u),
  - 'eta >= 1, E >= 50' : the remaining valid points (50 keV/u <= E <= 24.80 Z_P^2 keV/u).
CDW-EIS points with eta >= 1 and E < 50 keV/u are left out, as in the rest of the evaluation.

Analysed: the ACSNN interpolator (all charges in training, in-sample) and the hole models on their excluded
charges (predictions from eval_holes_<set>/test_predictions.csv), plus the Z_P^2 baselines of zp2_analysis.py
restricted to eta < 1.
Run: MODEL_SET=b100k python3 sommerfeld_analysis.py
"""
import os, logging
logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches
import matplotlib.pyplot as plt
from scipy import stats

from evaluate_holes import NAMELIST, CDW_EMIN, MODEL_SET, read_all_databases
from zp2_analysis import load_acsnn, classify, baselines

E_AU = 24.80                       # keV/u per (atomic unit of velocity)^2 / 2, with m_u
PRED = os.path.join('eval_holes' + (f'_{MODEL_SET}' if MODEL_SET else ''), 'test_predictions.csv')
OUT = 'eval_sommerfeld' + (f'_{MODEL_SET}' if MODEL_SET else '')
THEORY = 'CDW-EIS'
REGIMES = ['eta < 1', 'eta >= 1, E >= 50']
REG_COLORS = {'eta < 1': 'tab:blue', 'eta >= 1, E >= 50': 'tab:orange'}
CASE_ORDER = ['interp. gap 1', 'interp. gap 2', 'interp. gap 3', 'extrap. d=1', 'extrap. d=2', 'extrap. d=3']
CASE_COLORS = dict(zip(CASE_ORDER, ['tab:blue', 'tab:cyan', 'tab:green', 'tab:orange', 'tab:red', 'darkred']))


def sommerfeld(z, e):
    return z / np.sqrt(np.asarray(e, dtype=float) / E_AU)


def regime(z, e):
    eta = sommerfeld(z, e)
    return np.where(eta < 1, REGIMES[0], np.where(np.asarray(e) >= CDW_EMIN, REGIMES[1], 'excluded'))


def metrics(err):
    rel = np.abs(10**err - 1)
    return pd.Series({'n_points': len(err), 'MAE_log': np.mean(np.abs(err)), 'MBE_log': np.mean(err),
                      'RMSE_log': np.sqrt(np.mean(err**2)), 'sigma_MedRE_%': 100 * np.median(rel),
                      'sigma_P90_%': 100 * np.quantile(rel, 0.9), 'frac_lt10%': 100 * np.mean(rel < 0.1)})


def case_class(model, z):
    excluded = {int(c) for c in model}
    c = classify(z, [t for t in range(1, 10) if t not in excluded])
    return 'interp. gap 3' if c.startswith('interp. gap 3') else c


# ---------------------------------------------------------------- figures
def fig_regimes(ref_pts):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r, col, mk in [(REGIMES[0], 'tab:blue', 'o'), (REGIMES[1], 'tab:orange', 's'), ('excluded', '0.6', 'x')]:
        s = ref_pts[ref_pts['regime'] == r]
        ax.scatter(s['E'], s['Z'], color=col, marker=mk, s=18, label={'excluded': 'η ≥ 1, E < 50 keV/u (not evaluated)'}.get(
            r, r.replace('eta', 'η').replace('>=', '≥').replace('E ≥ 50', 'E ≥ 50 keV/u')))
    zz = np.linspace(0.8, 9.3, 200)
    ax.plot(E_AU * zz**2, zz, 'k-', lw=1, label='η = 1  (E = 24.80 Z_P² keV/u)')
    ax.axvline(CDW_EMIN, color='k', ls=':', lw=0.8)
    ax.set_xscale('log'); ax.set_xlabel('E [keV/u]'); ax.set_ylabel('Z_P'); ax.set_yticks(range(1, 10))
    ax.set_title('CDW-EIS grid points by Sommerfeld regime'); ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'sommerfeld_regimes.png'), dpi=130); plt.close(fig)


def fig_error_vs_eta(holes, acs):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.logspace(np.log10(0.08), np.log10(9 / np.sqrt(CDW_EMIN / E_AU)) + 1e-6, 16)
    mid = np.sqrt(bins[1:] * bins[:-1])
    for c in CASE_ORDER:
        s = holes[holes['case_class'] == c]
        if s.empty:
            continue
        g = s.groupby(pd.cut(s['eta'], bins), observed=False)
        ok = g.size().values >= 5
        ax[0].plot(mid[ok], 100 * (g['ratio'].median().values[ok] - 1), 'o-', ms=3, color=CASE_COLORS[c], label=c)
        ax[1].plot(mid[ok], g['abs_err'].median().values[ok], 'o-', ms=3, color=CASE_COLORS[c], label=c)
    g = acs.groupby(pd.cut(acs['eta'], bins), observed=False)
    ok = g.size().values >= 5
    ax[0].plot(mid[ok], 100 * (10**g['err'].median().values[ok] - 1), 'k*--', ms=6, label='ACSNN (in-sample)')
    ax[1].plot(mid[ok], g['abs_err'].median().values[ok], 'k*--', ms=6, label='ACSNN (in-sample)')
    ax[0].set_yscale('symlog', linthresh=10); ax[0].axhline(0, color='k', lw=0.8)
    ax[0].set_ylabel('Median signed error (σ_NN/σ_CDW-EIS − 1) [%]'); ax[0].set_title('Bias vs Sommerfeld parameter')
    ax[1].set_yscale('log'); ax[1].set_ylabel('Median |log10 σ_NN − log10 σ_CDW-EIS|')
    ax[1].set_title('Error magnitude vs Sommerfeld parameter')
    for a in ax:
        a.set_xscale('log'); a.axvline(1, color='k', ls='--', lw=1); a.set_xlabel('η = Z_P / v')
        a.grid(True, which='both', ls='--', lw=0.4, alpha=0.5); a.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'error_vs_sommerfeld.png'), dpi=130); plt.close(fig)


def fig_box_regimes(cases):
    present = [c for c in CASE_ORDER if c in cases['case_class'].values]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(present))
    for k, r in enumerate(REGIMES):
        data = [cases.loc[(cases['case_class'] == c) & (cases['regime'] == r), 'MAE_log'].dropna() for c in present]
        pos = x + (k - 0.5) * 0.36
        keep = [i for i, d in enumerate(data) if len(d)]
        bp = ax.boxplot([data[i] for i in keep], positions=pos[keep], widths=0.32, patch_artist=True, showfliers=False)
        for p in bp['boxes']:
            p.set_facecolor(REG_COLORS[r]); p.set_alpha(0.55)
        rng = np.random.default_rng(k)
        for i in keep:
            ax.scatter(pos[i] + rng.normal(0, 0.03, len(data[i])), data[i], s=6, color='k', alpha=0.4, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(present); ax.set_yscale('log'); ax.set_ylabel('Per-case MAE, CDW-EIS [log10 σ]')
    ax.legend(handles=[matplotlib.patches.Patch(color=REG_COLORS[r], alpha=0.55,
                                                label=r.replace('eta', 'η').replace('>=', '≥') + (' keV/u' if 'E' in r else ''))
                       for r in REGIMES]); ax.set_title('Hole models: CDW-EIS error on the excluded charge by Sommerfeld regime')
    ax.grid(True, axis='y', which='both', ls='--', lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'box_by_case_class_and_regime.png'), dpi=130); plt.close(fig)


def fig_baselines(cases_b):
    present = [c for c in CASE_ORDER if c in cases_b['case_class'].values]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(present)); w = 0.26
    for j, (col, lab, colr) in enumerate([('MAE_NN', 'hole network (NN)', 'tab:purple'),
                                          ('MAE_Born', 'Born: z² σ_p (Z = 1 in training)', 'tab:brown'),
                                          ('MAE_power', 'power law in Z', 'tab:olive')]):
        med = [cases_b.loc[cases_b['case_class'] == c, col].median() for c in present]
        ax.bar(x + (j - 1) * w, med, w, color=colr, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(present); ax.set_yscale('log')
    ax.set_ylabel('Median per-case MAE, CDW-EIS η < 1 [log10 σ]'); ax.legend(fontsize=8)
    ax.set_title('Perturbative regime (η < 1): hole networks vs scaling baselines')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'baselines_eta_lt1.png'), dpi=130); plt.close(fig)


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    df = read_all_databases(NAMELIST)
    df = df[(df['Z_target'] == 1) & (df['Theory_ID'] == THEORY)]
    ref = df.pivot_table(index='Z_ion', columns='E_proj[keV/u]', values='Cross_section[cm2]')
    energies = ref.columns.values

    ref_pts = pd.DataFrame([(z, e) for z in ref.index for e in energies], columns=['Z', 'E'])
    ref_pts['eta'] = sommerfeld(ref_pts['Z'], ref_pts['E'])
    ref_pts['regime'] = regime(ref_pts['Z'], ref_pts['E'])
    thresholds = pd.DataFrame({'Z': range(1, 10), 'E_eta1_keV/u': [E_AU * z**2 for z in range(1, 10)],
                               'n_eta_lt_1': [int(((ref_pts['Z'] == z) & (ref_pts['regime'] == REGIMES[0])).sum()) for z in range(1, 10)],
                               'n_eta_ge_1_E_ge_50': [int(((ref_pts['Z'] == z) & (ref_pts['regime'] == REGIMES[1])).sum()) for z in range(1, 10)]})
    thresholds.to_csv(os.path.join(OUT, 'sommerfeld_thresholds.csv'), index=False)
    fig_regimes(ref_pts)

    # ---------------- ACSNN interpolator, in-sample on CDW-EIS
    acs_net = load_acsnn()
    acs = ref_pts.copy()
    acs['pred'] = acs_net.log_sigma(acs['Z'].values.astype(float), np.log10(acs['E'].values))
    acs['log_sigma'] = np.log10([ref.loc[z, e] for z, e in zip(acs['Z'], acs['E'])])
    acs['err'] = acs['pred'] - acs['log_sigma']; acs['abs_err'] = acs['err'].abs()
    acs = acs[acs['regime'] != 'excluded']
    acs_tab = acs.groupby(['Z', 'regime'])['err'].apply(metrics).unstack().reset_index()
    acs_tab_all = acs.groupby('regime')['err'].apply(metrics).unstack().reset_index()
    acs_tab.to_csv(os.path.join(OUT, 'acsnn_by_charge_and_regime.csv'), index=False)
    acs_tab_all.to_csv(os.path.join(OUT, 'acsnn_by_regime.csv'), index=False)

    # ---------------- hole models on their excluded charges
    p = pd.read_csv(PRED, dtype={'model': str})
    p = p[p['Theory_ID'] == THEORY].copy()
    p['E'] = p['E_proj[keV/u]']
    p['eta'] = sommerfeld(p['Z_ion'], p['E'])
    p['regime'] = regime(p['Z_ion'], p['E'])
    p = p[p['regime'] != 'excluded']
    p['ratio'] = 10**p['err']
    keys = p[['model', 'Z_ion']].drop_duplicates()
    keys['case_class'] = [case_class(m, z) for m, z in zip(keys['model'], keys['Z_ion'])]
    p = p.merge(keys, on=['model', 'Z_ion'])
    p.to_csv(os.path.join(OUT, 'cdweis_predictions_with_eta.csv'), index=False)

    cases = p.groupby(['model', 'n_excluded', 'Z_ion', 'case_class', 'regime'])['err'].apply(metrics).unstack().reset_index()
    cases.to_csv(os.path.join(OUT, 'cases_by_regime.csv'), index=False)
    by_class = p.groupby(['case_class', 'regime'])['err'].apply(metrics).unstack().reset_index()
    by_n_class = p.groupby(['n_excluded', 'case_class', 'regime'])['err'].apply(metrics).unstack().reset_index()
    by_z = p.groupby(['Z_ion', 'regime'])['err'].apply(metrics).unstack().reset_index()
    for nm, t in [('holes_by_case_class_and_regime', by_class), ('holes_by_n_excluded_case_class_and_regime', by_n_class),
                  ('holes_by_charge_and_regime', by_z)]:
        t.to_csv(os.path.join(OUT, f'{nm}.csv'), index=False)

    # per-case medians and paired test eta < 1 vs eta >= 1 (cases that have both regimes, Z >= 2)
    wide = cases.pivot_table(index=['model', 'Z_ion', 'case_class'], columns='regime', values='MAE_log').reset_index()
    rows = []
    for c in [c for c in CASE_ORDER if c in wide['case_class'].values]:
        s = wide[wide['case_class'] == c]
        both = s.dropna(subset=REGIMES)
        rows.append({'case_class': c, 'cases': len(s), 'median_MAE_eta<1': both[REGIMES[0]].median(),
                     'median_MAE_eta>=1_E>=50': both[REGIMES[1]].median(), 'cases_with_both': len(both),
                     'median_ratio_eta>=1/eta<1': (both[REGIMES[1]] / both[REGIMES[0]]).median(),
                     'pct_cases_worse_at_eta>=1': 100 * (both[REGIMES[1]] > both[REGIMES[0]]).mean(),
                     'wilcoxon_p': stats.wilcoxon(both[REGIMES[0]], both[REGIMES[1]]).pvalue if len(both) >= 6 else np.nan})
    paired = pd.DataFrame(rows)
    paired.to_csv(os.path.join(OUT, 'per_case_regime_comparison.csv'), index=False)

    # ---------------- Z_P^2 baselines in the perturbative regime
    lsig = np.log10(ref)
    base_rows = []
    for (m, z, c), _ in p.groupby(['model', 'Z_ion', 'case_class']):
        excluded = {int(ch) for ch in m}
        train = [t for t in range(1, 10) if t not in excluded]
        born, z2, power, _ = baselines(z, train, ref)
        msk = sommerfeld(z, energies) < 1
        lr = lsig.loc[z].values
        nn = p[(p['model'] == m) & (p['Z_ion'] == z) & (p['regime'] == REGIMES[0])].sort_values('E')
        assert np.allclose(nn['E'].values, energies[msk])
        base_rows.append({'model': m, 'Z_ion': z, 'case_class': c,
                          'MAE_NN': np.mean(np.abs(nn['err'].values)),
                          'MAE_Born': np.mean(np.abs(born[msk] - lr[msk])) if 1 in train else np.nan,
                          'MAE_Z2': np.mean(np.abs(z2[msk] - lr[msk])),
                          'MAE_power': np.mean(np.abs(power[msk] - lr[msk]))})
    cases_b = pd.DataFrame(base_rows)
    base_tab = cases_b.groupby('case_class')[['MAE_NN', 'MAE_Born', 'MAE_Z2', 'MAE_power']].median().reindex(
        [c for c in CASE_ORDER if c in cases_b['case_class'].values]).reset_index()
    for base in ['Born', 'Z2', 'power']:
        base_tab[f'pct_NN_better_than_{base}'] = [
            100 * (g['MAE_NN'] < g[f'MAE_{base}']).mean() if g[f'MAE_{base}'].notna().any() else np.nan
            for g in (cases_b[cases_b['case_class'] == c].dropna(subset=[f'MAE_{base}']) for c in base_tab['case_class'])]
    cases_b.to_csv(os.path.join(OUT, 'baselines_eta_lt1_cases.csv'), index=False)
    base_tab.to_csv(os.path.join(OUT, 'baselines_eta_lt1_by_case_class.csv'), index=False)

    fig_error_vs_eta(p, acs)
    fig_box_regimes(cases)
    fig_baselines(cases_b)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    with open(os.path.join(OUT, 'console_output.txt'), 'w') as fp:
        for title, t in [('thresholds', thresholds), ('ACSNN by regime', acs_tab_all), ('ACSNN by charge and regime', acs_tab),
                         ('holes by case class and regime', by_class), ('holes by n excluded, case class and regime', by_n_class),
                         ('holes by charge and regime', by_z), ('per-case comparison', paired),
                         ('baselines eta<1 by case class', base_tab)]:
            txt = f'\n=== {title} ===\n' + t.round(4).to_string(index=False)
            print(txt); fp.write(txt + '\n')
