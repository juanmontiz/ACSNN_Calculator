"""Evaluation of the best_statedict_gauss_128_hole_XYZ.pth models.

Each model was trained (train_totals_NN_holes.py) on Z_target == 1 data with the
projectile charges Z_ion = X, Y, Z removed. This script rebuilds the one-hot encoder
of every model, evaluates it on the excluded charges (hole test set) and on the
charges it saw (in-sample), and classifies each case as interpolation /
extrapolation / mixed with respect to the charge.
A dedicated analysis is done for the single-charge models (111, 222, ..., 999).
"""
import os, re, glob, logging
# Configure logging before importing the training script so it does not write to logs.log
logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from train_totals_NN_holes import (FlexibleNN, read_all_databases,
                                   MAE, RMSE, MBE, MRE, MaxAE, MaxRE)

# Model set: MODEL_SET='' -> best_statedict_gauss_128_hole_XYZ.pth (CDW-EIS + CTMC + Rudd);
# MODEL_SET='b', 'b100k', ... -> best_statedict_gauss_128_hole_XYZ_<set>.pth (CDW-EIS + CTMC, unique holes, no Rudd)
MODEL_SET = os.environ.get('MODEL_SET', '')
SUFFIX = f'_{MODEL_SET}' if MODEL_SET else ''
OUT_DIR = 'eval_holes' + SUFFIX
SINGLE_DIR = os.path.join(OUT_DIR, 'single_charge')
# CDW-EIS drops to ~1e-28 cm2 at low energy (outside its validity range); those points dominate
# the linear relative errors, so an analysis without them is also reported.
CDW_EMIN = 50.0
NAMELIST = [os.path.join('.', 'dbs', 'Semiempiric_db', 'semiempiric_TCS_database_single_ionization_atoms_1985Rudd_H+_+_H.csv'),
            os.path.join('.', 'dbs', 'ctmc', 'ctmc_TCS_database_single_ionization_hydrogen.csv'),
            os.path.join('.', 'dbs', 'CDWEIS', 'cdw-eis_TCS_database_single_ionization_atoms_except_Kr.csv')]
if MODEL_SET.startswith('b'):
    NAMELIST = [p for p in NAMELIST if 'Rudd' not in p]
CLASS_ORDER = ['interpolation', 'mixed', 'extrapolation']


def model_path(name):
    return f'best_statedict_gauss_128_hole_{name}{SUFFIX}.pth'


def model_names(n_distinct=None):
    """3-digit names of the models of the current set, optionally with n_distinct excluded charges."""
    pattern = re.compile(rf'best_statedict_gauss_128_hole_(\d{{3}}){SUFFIX}\.pth$')
    names = sorted(m.group(1) for m in map(pattern.match, os.listdir('.')) if m)
    return [n for n in names if n_distinct is None or len(set(n)) == n_distinct]


def metrics_log(pred, true):
    """Same metrics used during training, on log10(sigma)."""
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(true, dtype=torch.float32)
    return {'MAE': MAE(p, t), 'RMSE': RMSE(p, t), 'MBE': MBE(p, t),
            'MRE': MRE(p, t), 'MaxAE': MaxAE(p, t), 'MaxRE': MaxRE(p, t)}


def metrics_sigma(pred, true):
    """Relative error on the cross section in linear scale."""
    rel = np.abs(10**pred - 10**true) / 10**true
    return {'sigma_MRE_%': 100 * rel.mean(), 'sigma_MedRE_%': 100 * np.median(rel),
            'sigma_MaxRE_%': 100 * rel.max(),
            'sigma_frac_lt10%': 100 * np.mean(rel < 0.10)}


def classify_charge(z, train_charges):
    return 'interpolation' if min(train_charges) < z < max(train_charges) else 'extrapolation'


def classify_model(per_charge):
    kinds = set(per_charge.values())
    return kinds.pop() if len(kinds) == 1 else 'mixed'


def is_valid(df):
    return ~((df['Theory_ID'] == 'CDW-EIS') & (df['E_proj[keV/u]'] < CDW_EMIN))


def load_model(path, df):
    hole = re.search(r'hole_(\d+)(?:_\w+)?\.pth', path).group(1)
    excluded = sorted({int(c) for c in hole})

    # Same reconstruction as in training
    train_df = df[~df['Z_ion'].isin(excluded)].reset_index(drop=True)
    encoder = OneHotEncoder(sparse_output=False).fit(train_df[['Theory_ID']].values)

    state = torch.load(path, map_location='cpu')
    model = FlexibleNN(2 + len(encoder.categories_[0]), state['lin1.weight'].shape[0], 'gauss', 0.0)
    model.load_state_dict(state)
    model.eval()

    def predict(sub):
        X = np.hstack([sub[['Z_ion', 'log_E']].values,
                       encoder.transform(sub[['Theory_ID']].values)])
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()

    return {'hole': hole, 'excluded': excluded, 'train_df': train_df,
            'known': set(encoder.categories_[0]), 'predict': predict}


def evaluate_model(path, df):
    m = load_model(path, df)
    hole, excluded, train_df, known, predict = m['hole'], m['excluded'], m['train_df'], m['known'], m['predict']
    train_charges = sorted(train_df['Z_ion'].unique())

    test_df = df[df['Z_ion'].isin(excluded)].copy()
    # Theories missing from training (e.g. Rudd only has Z_ion=1) cannot be encoded
    n_skipped = int((~test_df['Theory_ID'].isin(known)).sum())
    test_df = test_df[test_df['Theory_ID'].isin(known)].copy()
    test_df['pred'] = predict(test_df)
    train_df['pred'] = predict(train_df)

    per_charge = {z: classify_charge(z, train_charges) for z in excluded}
    info = {'model': hole, 'excluded': ','.join(map(str, excluded)),
            'n_excluded': len(excluded), 'class': classify_model(per_charge),
            'n_test': len(test_df), 'n_test_skipped': n_skipped}

    row = dict(info)
    row.update({f'test_{k}': v for k, v in metrics_log(test_df['pred'].values, test_df['log_sigma'].values).items()})
    row.update({f'test_{k}': v for k, v in metrics_sigma(test_df['pred'].values, test_df['log_sigma'].values).items()})
    row.update({f'train_{k}': v for k, v in metrics_log(train_df['pred'].values, train_df['log_sigma'].values).items()})
    row.update({f'train_{k}': v for k, v in metrics_sigma(train_df['pred'].values, train_df['log_sigma'].values).items()})

    test_df['model'] = hole
    test_df['model_class'] = info['class']
    test_df['charge_class'] = test_df['Z_ion'].map(per_charge)
    test_df['n_excluded'] = len(excluded)
    return row, test_df


def summarize(points, keys):
    g = points.groupby(keys)
    out = pd.DataFrame({
        'n_points': g.size(),
        'n_models': g['model'].nunique(),
        'MAE_log': g['abs_err'].mean(),
        'RMSE_log': g['sq_err'].mean() ** 0.5,
        'MBE_log': g['err'].mean(),
        'MaxAE_log': g['abs_err'].max(),
        'sigma_MRE_%': 100 * g['rel_sigma'].mean(),
        'sigma_MedRE_%': 100 * g['rel_sigma'].median(),
        'sigma_P90_%': 100 * g['rel_sigma'].quantile(0.9),
        'sigma_MaxRE_%': 100 * g['rel_sigma'].max(),
    })
    return out.reset_index()


def make_plots(models, points):
    present = [c for c in CLASS_ORDER if c in models['class'].values]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].boxplot([models.loc[models['class'] == c, 'test_MAE'] for c in present], tick_labels=present)
    ax[0].set_yscale('log'); ax[0].set_ylabel('Test MAE [log10 σ]'); ax[0].set_title('MAE per model')
    ax[1].boxplot([models.loc[models['class'] == c, 'test_sigma_MRE_%'] for c in present], tick_labels=present)
    ax[1].set_yscale('log'); ax[1].set_ylabel('Mean relative error in σ [%]'); ax[1].set_title('Relative error per model')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'box_by_class.png'), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    zs = sorted(points['Z_ion'].unique())
    ax.boxplot([100 * points.loc[points['Z_ion'] == z, 'rel_sigma'] for z in zs], tick_labels=zs, showfliers=False)
    ax.set_yscale('log'); ax.set_xlabel('Excluded Z_ion'); ax.set_ylabel('Relative error in σ [%]')
    ax.set_title('Error on the excluded charge (all models)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'box_by_charge.png'), dpi=130); plt.close(fig)

    # Heatmap: error on excluded charge z depending on which other charge was excluded with it
    fig, ax = plt.subplots(figsize=(7, 5.5))
    piv = (points.groupby(['model', 'Z_ion'])['rel_sigma'].median() * 100).unstack()
    heat = np.full((9, 9), np.nan)
    for model_id, row in piv.iterrows():
        exc = sorted({int(c) for c in model_id})
        for z in exc:
            others = [o for o in exc if o != z] or [z]
            for o in others:
                cur = heat[z - 1, o - 1]
                heat[z - 1, o - 1] = row[z] if np.isnan(cur) else np.nanmean([cur, row[z]])
    im = ax.imshow(np.log10(heat), cmap='viridis', origin='lower')
    ax.set_xticks(range(9)); ax.set_xticklabels(range(1, 10)); ax.set_yticks(range(9)); ax.set_yticklabels(range(1, 10))
    ax.set_xlabel('Other excluded charge (diagonal = excluded alone)'); ax.set_ylabel('Evaluated charge')
    cb = fig.colorbar(im, ax=ax); cb.set_label('log10(median relative error in σ [%])')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'heatmap_charges.png'), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in ['interpolation', 'extrapolation']:
        sub = points[points['charge_class'] == c]
        if sub.empty:
            continue
        b = sub.groupby(pd.cut(sub['log_E'], 15), observed=True)['rel_sigma'].median() * 100
        ax.plot([iv.mid for iv in b.index], b.values, marker='o', label=c)
    ax.set_yscale('log'); ax.set_xlabel('log10 E [keV/u]'); ax.set_ylabel('Median relative error in σ [%]')
    ax.set_title('Error vs energy by charge class'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'error_vs_energy.png'), dpi=130); plt.close(fig)


def single_charge_analysis(df, models, points):
    """Models trained with exactly one excluded charge (111, 222, ..., 999)."""
    os.makedirs(SINGLE_DIR, exist_ok=True)
    s_models = models[models['n_excluded'] == 1].sort_values('excluded').reset_index(drop=True)
    s_points = points[points['n_excluded'] == 1]
    s_valid = s_points[is_valid(s_points)]

    s_models.to_csv(os.path.join(SINGLE_DIR, 'metrics_per_model.csv'), index=False)
    tables = {
        'by_model': summarize(s_points, ['Z_ion', 'charge_class']),
        'by_model_without_low_E_CDW': summarize(s_valid, ['Z_ion', 'charge_class']),
        'by_model_and_theory': summarize(s_valid, ['Z_ion', 'charge_class', 'Theory_ID']),
        'by_class': summarize(s_points, ['charge_class']),
        'by_class_without_low_E_CDW': summarize(s_valid, ['charge_class']),
    }
    for name, t in tables.items():
        t.to_csv(os.path.join(SINGLE_DIR, f'summary_{name}.csv'), index=False)
    stats = s_models.groupby('class')[['test_MAE', 'test_RMSE', 'test_MBE', 'test_MaxAE',
                                       'test_sigma_MedRE_%', 'valid_sigma_MRE_%', 'train_MAE']].describe().T
    stats.to_csv(os.path.join(SINGLE_DIR, 'model_statistics_by_class.csv'))

    zs = s_models['excluded'].astype(int).tolist()
    colors = {'interpolation': 'tab:blue', 'extrapolation': 'tab:red'}
    cls = dict(zip(zs, s_models['class']))

    # 1) Predicted vs reference σ(E) for each excluded charge
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    for ax, z in zip(axes.ravel(), zs):
        m = load_model(model_path(f'{z}{z}{z}'), df)
        for theory, mk, col in [('CDW-EIS', 'o', 'tab:green'), ('CTMC', 's', 'tab:purple')]:
            ref = df[(df['Z_ion'] == z) & (df['Theory_ID'] == theory)].sort_values('log_E')
            grid = pd.DataFrame({'Z_ion': z, 'Theory_ID': theory,
                                 'log_E': np.linspace(ref['log_E'].min(), ref['log_E'].max(), 200)})
            ax.plot(10**grid['log_E'], 10**m['predict'](grid), color=col, label=f'NN {theory}')
            ax.plot(ref['E_proj[keV/u]'], ref['Cross_section[cm2]'], mk, color=col, mfc='none', ms=4,
                    label=f'{theory} reference')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-19, 1e-14)
        ax.set_title(f'$Z_P$ = {z} excluded ({cls[z]})', color=colors[cls[z]])
        ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.6)
    for ax in axes[-1]:
        ax.set_xlabel('E [keV/u]')
    for ax in axes[:, 0]:
        ax.set_ylabel('σ [cm²]')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Single-charge exclusion: NN prediction on the unseen charge (target H)')
    fig.tight_layout(); fig.savefig(os.path.join(SINGLE_DIR, 'curves_per_excluded_charge.png'), dpi=130); plt.close(fig)

    # 2) Error metrics per excluded charge
    t = tables['by_model_without_low_E_CDW'].set_index('Z_ion')
    t_all = tables['by_model'].set_index('Z_ion')
    bar_colors = [colors[cls[z]] for z in t.index]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].bar(t.index, t_all['MAE_log'], color=bar_colors, alpha=0.35, label='all points')
    ax[0].bar(t.index, t['MAE_log'], color=bar_colors, label=f'without CDW-EIS E<{CDW_EMIN:g} keV/u')
    ax[0].set_ylabel('MAE [log10 σ]'); ax[0].set_title('Mean absolute error (log10)'); ax[0].legend(fontsize=8)
    ax[1].bar(t.index, t['MBE_log'], color=bar_colors); ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set_ylabel('MBE [log10 σ]'); ax[1].set_title('Mean bias error (log10, valid points)')
    ax[2].bar(t.index, t['sigma_MedRE_%'], color=bar_colors, label='median')
    ax[2].scatter(t.index, t['sigma_P90_%'], color='k', marker='_', s=200, label='P90')
    ax[2].set_yscale('log'); ax[2].set_ylabel('Relative error in σ [%]')
    ax[2].set_title('Relative error in σ (valid points)'); ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xticks(t.index); a.set_xlabel('Excluded Z_ion')
    fig.tight_layout(); fig.savefig(os.path.join(SINGLE_DIR, 'metrics_per_excluded_charge.png'), dpi=130); plt.close(fig)

    # 3) Signed relative error vs energy
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap('viridis', 9)
    for z in zs:
        sub = s_points[s_points['Z_ion'] == z]
        b = sub.groupby(pd.cut(sub['log_E'], 14), observed=True).apply(
            lambda g: 100 * np.median((10**g['pred'] - 10**g['log_sigma']) / 10**g['log_sigma']), include_groups=False)
        ax.plot([iv.mid for iv in b.index], b.values, marker='o', ms=3, color=cmap(z - 1),
                ls='--' if cls[z] == 'extrapolation' else '-', label=f'Z={z} ({cls[z][:6]}.)')
    ax.axhline(0, color='k', lw=0.8); ax.set_yscale('symlog', linthresh=10)
    ax.set_xlabel('log10 E [keV/u]'); ax.set_ylabel('Median signed relative error (σ_NN − σ_ref)/σ_ref [%]')
    ax.set_title('Single-charge exclusion: signed error vs energy'); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(SINGLE_DIR, 'signed_error_vs_energy.png'), dpi=130); plt.close(fig)

    # 4) Parity plot
    fig, ax = plt.subplots(figsize=(6, 6))
    for z in zs:
        sub = s_points[s_points['Z_ion'] == z]
        ax.scatter(sub['log_sigma'], sub['pred'], s=8, color=cmap(z - 1), label=f'Z={z}',
                   marker='x' if cls[z] == 'extrapolation' else 'o', alpha=0.7)
    lim = [-20, -14.5]
    ax.plot(lim, lim, 'k-', lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('log10 σ reference'); ax.set_ylabel('log10 σ NN')
    ax.set_title('Parity plot, single-charge exclusion\n(o interpolation, x extrapolation)'); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(SINGLE_DIR, 'parity.png'), dpi=130); plt.close(fig)

    return s_models, tables


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    df = read_all_databases(NAMELIST)
    df = df[df['Z_target'] == 1].reset_index(drop=True)
    df['log_E'] = np.log10(df['E_proj[keV/u]'].values)
    df['log_sigma'] = np.log10(df['Cross_section[cm2]'].values)

    rows, all_points = [], []
    for path in map(model_path, model_names()):
        row, pts = evaluate_model(path, df)
        rows.append(row); all_points.append(pts)
        print(f"{row['model']}  {row['class']:<14} test MAE={row['test_MAE']:.4f}  σ-MRE={row['test_sigma_MRE_%']:.2f}%")

    models = pd.DataFrame(rows)
    points = pd.concat(all_points, ignore_index=True)
    points['err'] = points['pred'] - points['log_sigma']
    points['abs_err'] = points['err'].abs()
    points['sq_err'] = points['err'] ** 2
    points['rel_sigma'] = np.abs(10**points['pred'] - 10**points['log_sigma']) / 10**points['log_sigma']
    points.to_csv(os.path.join(OUT_DIR, 'test_predictions.csv'), index=False)

    tables = {
        'by_model_class': summarize(points, ['model_class']),
        'by_charge_class': summarize(points, ['charge_class']),
        'by_charge': summarize(points, ['Z_ion', 'charge_class']),
        'by_theory': summarize(points, ['Theory_ID', 'charge_class']),
        'by_n_excluded': summarize(points, ['n_excluded', 'model_class']),
    }
    valid = points[is_valid(points)]
    for name, keys in [('by_model_class', ['model_class']), ('by_charge_class', ['charge_class']),
                       ('by_charge', ['Z_ion', 'charge_class'])]:
        tables[f'{name}_without_low_E_CDW'] = summarize(valid, keys)
    per_model_valid = summarize(valid, ['model']).rename(columns=lambda c: c if c == 'model' else f'valid_{c}')
    models = models.merge(per_model_valid[['model', 'valid_MAE_log', 'valid_sigma_MRE_%', 'valid_sigma_MaxRE_%']],
                          on='model', how='left')
    models.to_csv(os.path.join(OUT_DIR, 'metrics_per_model.csv'), index=False)
    for name, t in tables.items():
        t.to_csv(os.path.join(OUT_DIR, f'summary_{name}.csv'), index=False)

    metric_cols = ['test_MAE', 'test_RMSE', 'test_MBE', 'test_MRE', 'test_MaxAE', 'test_MaxRE',
                   'test_sigma_MRE_%', 'test_sigma_MedRE_%', 'test_sigma_MaxRE_%', 'valid_sigma_MRE_%',
                   'train_MAE', 'train_sigma_MRE_%']
    models.groupby('class')[metric_cols].describe().T.to_csv(os.path.join(OUT_DIR, 'model_statistics_by_class.csv'))

    make_plots(models, points)
    s_models, s_tables = single_charge_analysis(df, models, points)

    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
    print('\n=== Models per class ===\n', models['class'].value_counts())
    for name, t in tables.items():
        print(f'\n=== {name} ===\n', t.round(4).to_string(index=False))
    print('\n=== Best 5 / worst 5 models (test σ-MRE %) ===')
    cols = ['model', 'excluded', 'class', 'test_MAE', 'test_sigma_MRE_%', 'test_sigma_MaxRE_%',
            'train_sigma_MRE_%', 'valid_sigma_MRE_%']
    print(models.sort_values('test_sigma_MRE_%')[cols].head().round(3).to_string(index=False))
    print(models.sort_values('test_sigma_MRE_%')[cols].tail().round(3).to_string(index=False))

    print('\n\n##### SINGLE-CHARGE EXCLUSION #####')
    cols = ['model', 'excluded', 'class', 'n_test', 'n_test_skipped', 'test_MAE', 'test_RMSE', 'test_MBE',
            'test_MaxAE', 'test_MRE', 'test_sigma_MedRE_%', 'valid_MAE_log', 'valid_sigma_MRE_%',
            'valid_sigma_MaxRE_%', 'train_MAE', 'train_sigma_MRE_%']
    print(s_models[cols].round(4).to_string(index=False))
    for name, t in s_tables.items():
        print(f'\n=== single charge: {name} ===\n', t.round(4).to_string(index=False))
