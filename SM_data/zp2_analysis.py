"""Z_P^2 (first-Born) scaling analysis of the CDW-EIS cross sections and of the NN models.

In first Born approximation the ionization cross section of H by a bare ion of charge Z_P at fixed
velocity scales as sigma = Z_P^2 sigma_p(E), with sigma_p the proton cross section. This script measures
how the CDW-EIS reference data, the ACSNN interpolator (trained on the full data set) and the hole models
(trained without some charges) follow or depart from that scaling:

  - scaled cross section sigma / Z^2,
  - scaling ratio R(Z,E) = sigma(Z,E) / (Z^2 sigma_ref(1,E)), R = 1 in first Born,
  - effective exponent n_eff(Z,E) = d ln sigma / d ln Z (n_eff = 2 in first Born); for the networks it is
    computed exactly by differentiating the network with respect to its Z input,
  - for the hole models, the prediction on the excluded charges against three physics baselines built from
    the CDW-EIS data of the training charges only: first-Born scaling from the proton (sigma = z^2 sigma_p),
    Z^2 scaling from the nearest training charge(s), and a power law in Z through two training charges.

The exponent of the networks is compared with the reference on the same finite-difference stencil
(centred on interior charges, one-sided at Z = 1 and 9); the exact derivative is shown in the figures.

Only CDW-EIS is analysed. Metrics use the valid points (CDW-EIS E >= 50 keV/u).
Run: MODEL_SET=b100k python3 zp2_analysis.py   (hole models best_statedict_gauss_128_hole_XYZ_b100k.pth)
"""
import os, logging
logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from train_totals_NN_holes import FlexibleNN
from evaluate_holes import (NAMELIST, CDW_EMIN, MODEL_SET, read_all_databases, model_path, model_names)

# ACSNN interpolator: the model of the main article (all charges, CDW-EIS + CTMC + Rudd). Set ACSNN_WEIGHTS to a .npz
# file to use calculator-format weights instead.
ACSNN_WEIGHTS = os.environ.get('ACSNN_WEIGHTS', 'best_statedict_gauss_128.pth')
ACSNN_THEORIES = ['CDW-EIS', 'CTMC', 'Semiempiric_1985Rudd']     # one-hot order of the ACSNN interpolator
OUT = 'eval_zp2' + (f'_{MODEL_SET}' if MODEL_SET else '')
THEORY = 'CDW-EIS'
ZS = np.arange(1, 10)
LN10 = np.log(10.0)
CASE_ORDER = ['interp. gap 1', 'interp. gap 2', 'interp. gap 3', 'extrap. d=1', 'extrap. d=2', 'extrap. d=3']
CASE_COLORS = dict(zip(CASE_ORDER, ['tab:blue', 'tab:cyan', 'tab:green', 'tab:orange', 'tab:red', 'darkred']))
ZCOLORS = plt.get_cmap('viridis', 9)


# ---------------------------------------------------------------- networks with a differentiable Z input
class Net:
    """A FlexibleNN with the one-hot position of CDW-EIS; predicts log10 sigma and d log10 sigma / dZ."""

    def __init__(self, model, n_theories, cdw_index):
        self.model, self.n_theories, self.cdw_index = model.eval(), n_theories, cdw_index

    def _x(self, z, log_e):
        z = torch.as_tensor(np.asarray(z, dtype=np.float32))
        log_e = torch.as_tensor(np.asarray(log_e, dtype=np.float32))
        z, log_e = torch.broadcast_tensors(z, log_e)
        onehot = torch.zeros(z.shape + (self.n_theories,))
        onehot[..., self.cdw_index] = 1.0
        return z, torch.cat([z[..., None], log_e[..., None], onehot], dim=-1)

    def log_sigma(self, z, log_e):
        with torch.no_grad():
            return self.model(self._x(z, log_e)[1]).squeeze(-1).numpy()

    def n_eff(self, z, log_e):
        """d ln sigma / d ln Z = Z ln(10) d log10 sigma / dZ, exact (autograd)."""
        z_t, x = self._x(z, log_e)
        x = x.clone().requires_grad_(True)
        y = self.model(x).sum()
        dy_dz = torch.autograd.grad(y, x)[0][..., 0]
        return (z_t * LN10 * dy_dz).detach().numpy()


def load_acsnn():
    if ACSNN_WEIGHTS.endswith('.npz'):
        w = np.load(ACSNN_WEIGHTS)
        state = {k.replace('_weight', '.weight').replace('_bias', '.bias'): torch.tensor(v) for k, v in w.items()}
    else:
        state = torch.load(ACSNN_WEIGHTS, map_location='cpu')
    model = FlexibleNN(state['lin1.weight'].shape[1], state['lin1.weight'].shape[0], 'gauss', 0.0)
    model.load_state_dict(state)
    return Net(model, len(ACSNN_THEORIES), ACSNN_THEORIES.index(THEORY))


def load_hole(name, theories_all):
    excluded = sorted({int(c) for c in name})
    state = torch.load(model_path(name), map_location='cpu')
    # Same one-hot as in training: categories of the training data, sorted alphabetically
    theories = sorted(t for t in theories_all if t != 'Semiempiric_1985Rudd' or 1 not in excluded)
    model = FlexibleNN(state['lin1.weight'].shape[1], state['lin1.weight'].shape[0], 'gauss', 0.0)
    model.load_state_dict(state)
    assert state['lin1.weight'].shape[1] == 2 + len(theories)
    return excluded, Net(model, len(theories), theories.index(THEORY))


# ---------------------------------------------------------------- reference quantities
def reference_exponent(ref):
    """Finite-difference d ln sigma / d ln Z on the integer charge grid (centred; one-sided at Z = 1, 9)."""
    ln = np.log(ref.values)                                        # ref: index Z, columns E
    n = np.empty_like(ln)
    for i, z in enumerate(ref.index):
        lo, hi = (i - 1 if i > 0 else i), (i + 1 if i < len(ref.index) - 1 else i)
        n[i] = (ln[hi] - ln[lo]) / (np.log(ref.index[hi]) - np.log(ref.index[lo]))
    return pd.DataFrame(n, index=ref.index, columns=ref.columns)


def classify(z, train):
    lower, upper = [t for t in train if t < z], [t for t in train if t > z]
    if lower and upper:
        return f'interp. gap {min(upper) - max(lower) - 1}'
    return f'extrap. d={min(abs(z - t) for t in train)}'


def baselines(z, train, ref):
    """log10 sigma predicted for charge z from the reference data of the training charges only.

    born:  first-Born scaling from the proton, z^2 sigma(1); NaN when Z = 1 is not a training charge.
    z2:    Z^2 scaling from the nearest training charge t, (z/t)^2 sigma(t); with two equally close
           charges the two estimates are averaged in log10.
    power: sigma proportional to Z^n through the nearest training charges below and above z
           (interpolation) or the two nearest training charges (extrapolation).
    """
    lsig = np.log10(ref)
    born = lsig.loc[1].values + 2 * np.log10(z) if 1 in train else np.full(lsig.shape[1], np.nan)
    dist = sorted(train, key=lambda t: (abs(t - z), t))
    near = [t for t in train if abs(t - z) == abs(dist[0] - z)]
    # Z^2 scaling from the nearest training charge (average in log if two are equally close)
    z2 = np.mean([lsig.loc[t].values + 2 * np.log10(z / t) for t in near], axis=0)
    lower, upper = [t for t in train if t < z], [t for t in train if t > z]
    if lower and upper:
        a, b = max(lower), min(upper)
    else:                                                         # extrapolation: two nearest on one side
        a, b = sorted(dist[:2])
    slope = (lsig.loc[b].values - lsig.loc[a].values) / (np.log10(b) - np.log10(a))
    power = lsig.loc[a].values + slope * (np.log10(z) - np.log10(a))
    return born, z2, power, (a, b)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# ---------------------------------------------------------------- figures
def fig_reference_and_acsnn(ref, acs, e_grid):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    sp = ref.loc[1]
    for z in ZS:
        col = ZCOLORS(z - 1)
        lg = np.log10(e_grid)
        s_nn = 10**acs.log_sigma(np.full_like(lg, z), lg)
        ax[0].plot(ref.columns, ref.loc[z] / z**2, 'o', color=col, ms=3.5, mfc='none')
        ax[0].plot(e_grid, s_nn / z**2, '-', color=col, lw=1.2, label=rf'$Z_P$ = {z}')
        sp_grid = np.interp(lg, np.log10(ref.columns), np.log10(sp.values))
        ax[1].plot(ref.columns, ref.loc[z] / (z**2 * sp), 'o', color=col, ms=3.5, mfc='none')
        ax[1].plot(e_grid, s_nn / (z**2 * 10**sp_grid), '-', color=col, lw=1.2)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel('E [keV/u]'); a.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)
        a.axvspan(ref.columns.min(), CDW_EMIN, color='0.9', zorder=0)
    ax[0].set_yscale('log'); ax[0].set_ylim(1e-21, 1e-15); ax[0].set_ylabel(r'$\sigma\,/\,Z_P^2$ [cm$^2$]')
    ax[0].set_title('Scaled cross section (markers: CDW-EIS, lines: ACSNN)'); ax[0].legend(fontsize=8, ncol=3)
    ax[1].set_yscale('log'); ax[1].set_ylim(0.03, 2); ax[1].axhline(1, color='k', lw=0.8)
    ax[1].set_ylabel(r'$R = \sigma(Z_P)\,/\,[Z_P^2\,\sigma_p]$')
    ax[1].set_title(r'Scaling ratio ($\sigma_p$: CDW-EIS proton cross section)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'acsnn_scaled_cross_section.png'), dpi=130); plt.close(fig)


def fig_exponent_acsnn(ref_n, acs, e_grid):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    lg = np.log10(e_grid)
    for z in ZS:
        col = ZCOLORS(z - 1)
        ax[0].plot(ref_n.columns, ref_n.loc[z], 'o', color=col, ms=3.5, mfc='none')
        ax[0].plot(e_grid, acs.n_eff(np.full_like(lg, z), lg), '-', color=col, lw=1.2, label=f'Z = {z}')
    ax[0].axhline(2, color='k', lw=0.8); ax[0].axvspan(ref_n.columns.min(), CDW_EMIN, color='0.9', zorder=0)
    ax[0].set_xscale('log'); ax[0].set_ylim(-2, 4); ax[0].set_xlabel('E [keV/u]')
    ax[0].set_ylabel('n_eff = ∂ln σ / ∂ln Z'); ax[0].legend(fontsize=8, ncol=3)
    ax[0].set_title('Effective charge exponent (markers: CDW-EIS finite differences, lines: ACSNN exact)')
    ax[0].grid(True, which='both', ls='--', lw=0.4, alpha=0.5)
    zc = np.linspace(1, 9, 161); lgc = np.linspace(1, 5, 161)
    ZZ, LL = np.meshgrid(zc, lgc)
    N = acs.n_eff(ZZ, LL)
    im = ax[1].pcolormesh(zc, lgc, N, cmap='RdBu_r', vmin=-2, vmax=6, shading='auto')
    cs = ax[1].contour(zc, lgc, N, levels=[0, 1, 1.5, 1.9, 2.1], colors='k', linewidths=0.8)
    ax[1].clabel(cs, fontsize=7)
    ax[1].axhline(np.log10(CDW_EMIN), color='w', ls='--', lw=0.8)
    fig.colorbar(im, ax=ax[1], label='n_eff (ACSNN)')
    ax[1].set_xlabel('Z (continuous network input)'); ax[1].set_ylabel('log10 E [keV/u]')
    ax[1].set_title('ACSNN exponent map (contours n_eff = 0, 1, 1.5, 1.9, 2.1)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'acsnn_effective_exponent.png'), dpi=130); plt.close(fig)


def fig_holes_ratio(ref, hole_curves):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    sp = ref.loc[1]
    for ax, z in zip(axes.ravel(), ZS):
        for c in CASE_ORDER:
            for name, ls in hole_curves.get((z, c), []):
                ax.plot(ref.columns, 10**ls / (z**2 * sp.values), '-', color=CASE_COLORS[c], lw=0.8, alpha=0.6)
        ax.plot(ref.columns, ref.loc[z] / (z**2 * sp), 'ko', ms=3.5, mfc='w', zorder=5)
        ax.axhline(1, color='k', lw=0.6); ax.axvspan(ref.columns.min(), CDW_EMIN, color='0.9', zorder=0)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.03, 3)
        ax.set_title(f'Excluded charge z = {z}'); ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)
    for c in CASE_ORDER:
        axes[0, 0].plot([], [], color=CASE_COLORS[c], label=c)
    axes[0, 0].plot([], [], 'ko', mfc='w', label='CDW-EIS reference')
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('E [keV/u]')
    for ax in axes[:, 0]:
        ax.set_ylabel('R = σ / [z² σ_CDW-EIS(1)]')
    fig.suptitle('Hole models: Z² scaling ratio of the excluded charge (one line per model, colored by case class)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'holes_scaling_ratio.png'), dpi=120); plt.close(fig)


def fig_holes_exponent(ref_n, hole_n):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    for ax, z in zip(axes.ravel(), ZS):
        for c in CASE_ORDER:
            for n in hole_n.get((z, c), []):
                ax.plot(ref_n.columns, n, '-', color=CASE_COLORS[c], lw=0.8, alpha=0.6)
        ax.plot(ref_n.columns, ref_n.loc[z], 'ko', ms=3.5, mfc='w', zorder=5)
        ax.axhline(2, color='k', lw=0.6); ax.axvspan(ref_n.columns.min(), CDW_EMIN, color='0.9', zorder=0)
        ax.set_xscale('log'); ax.set_ylim(-2, 5); ax.set_title(f'Excluded charge z = {z}')
        ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)
    for c in CASE_ORDER:
        axes[0, 0].plot([], [], color=CASE_COLORS[c], label=c)
    axes[0, 0].plot([], [], 'ko', mfc='w', label='CDW-EIS finite differences')
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('E [keV/u]')
    for ax in axes[:, 0]:
        ax.set_ylabel('n_eff = ∂ln σ / ∂ln Z')
    fig.suptitle('Hole models: effective charge exponent at the excluded charge (exact derivative of each network)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'holes_effective_exponent.png'), dpi=120); plt.close(fig)


def fig_baselines(cases):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={'width_ratios': [1, 1.25]})
    for c in CASE_ORDER:
        s = cases[cases['case_class'] == c]
        ax[0].scatter(s['MAE_power'], s['MAE_NN'], s=14, color=CASE_COLORS[c], label=c, alpha=0.8)
    lim = [cases[['MAE_NN', 'MAE_power']].min().min() * 0.7, cases[['MAE_NN', 'MAE_power']].max().max() * 1.4]
    ax[0].plot(lim, lim, 'k-', lw=0.8)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel('MAE of the power-law baseline [log10 σ]'); ax[0].set_ylabel('MAE of the hole network [log10 σ]')
    ax[0].set_title('Below the diagonal: network better than the power law', fontsize=10)
    ax[0].legend(fontsize=7.5)
    present = [c for c in CASE_ORDER if c in cases['case_class'].values]
    x = np.arange(len(present)); w = 0.26
    for j, (col, lab, colr) in enumerate([('MAE_NN', 'hole network (NN)', 'tab:purple'),
                                          ('MAE_Born', 'Born: z² σ_p (Z = 1 in training)', 'tab:brown'),
                                          ('MAE_power', 'power law in Z', 'tab:olive')]):
        med = [cases.loc[cases['case_class'] == c, col].median() for c in present]
        ax[1].bar(x + (j - 1) * w, med, w, color=colr, label=lab)
    ax[1].set_xticks(x); ax[1].set_xticklabels(present, rotation=30, ha='right'); ax[1].set_yscale('log')
    ax[1].set_ylabel('Median per-case MAE [log10 σ]'); ax[1].legend(fontsize=8)
    ax[1].set_title('Median per-case error by case class', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'holes_vs_physics_baselines.png'), dpi=130); plt.close(fig)


def fig_exponent_error(cases):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    present = [c for c in CASE_ORDER if c in cases['case_class'].values]
    data = [cases.loc[cases['case_class'] == c, 'n_eff_MAE'] for c in present]
    bp = ax[0].boxplot(data, tick_labels=present, patch_artist=True)
    for p, c in zip(bp['boxes'], present):
        p.set_facecolor(CASE_COLORS[c]); p.set_alpha(0.5)
    ax[0].set_yscale('log'); ax[0].set_ylabel('mean |n_eff(NN) − n_eff(CDW-EIS)|, same stencil, valid E')
    ax[0].tick_params(axis='x', rotation=30); ax[0].set_title('Error of the charge exponent at the excluded charge')
    for c in present:
        s = cases[cases['case_class'] == c]
        ax[1].scatter(s['n_eff_MAE'], s['MAE_NN'], s=14, color=CASE_COLORS[c], label=c, alpha=0.8)
    rho, p = stats.spearmanr(cases['n_eff_MAE'], cases['MAE_NN'])
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].legend(fontsize=7.5)
    ax[1].set_xlabel('mean |Δ n_eff| at the excluded charge'); ax[1].set_ylabel('MAE of the hole model [log10 σ]')
    ax[1].set_title(f'Cross-section error vs exponent error (Spearman ρ = {rho:.2f})')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'holes_exponent_error.png'), dpi=130); plt.close(fig)


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    df = read_all_databases(NAMELIST)
    df = df[df['Z_target'] == 1]
    theories_all = sorted(df['Theory_ID'].unique())
    cdw = df[df['Theory_ID'] == THEORY]
    ref = cdw.pivot_table(index='Z_ion', columns='E_proj[keV/u]', values='Cross_section[cm2]')
    energies = ref.columns.values
    log_e = np.log10(energies)
    valid = energies >= CDW_EMIN
    ref_n = reference_exponent(ref)
    e_grid = np.logspace(1, 5, 300)

    # ---------------- reference: how well does CDW-EIS follow Z^2 scaling?
    R_ref = ref.div(ref.loc[1], axis=1).div(ref.index.values**2, axis=0)
    ref_rows = []
    for z in ZS:
        for lo, hi, band in [(CDW_EMIN, 500, '50-500'), (500, 5000, '500-5000'), (5000, 1.01e5, '5000-1e5')]:
            m = (energies >= lo) & (energies < hi)
            ref_rows.append({'Z': z, 'E_band_keV/u': band, 'R_ref_median': R_ref.loc[z, m].median(),
                             'n_eff_ref_median': ref_n.loc[z, m].median()})
    ref_tab = pd.DataFrame(ref_rows)
    ref_tab.to_csv(os.path.join(OUT, 'reference_scaling_by_band.csv'), index=False)
    R_ref.to_csv(os.path.join(OUT, 'reference_scaling_ratio.csv'))
    ref_n.to_csv(os.path.join(OUT, 'reference_effective_exponent.csv'))

    # ---------------- ACSNN interpolator (full training set)
    acs = load_acsnn()
    acs_rows = []
    for z in ZS:
        ls = acs.log_sigma(np.full_like(log_e, z), log_e)
        n_nn = acs.n_eff(np.full_like(log_e, z), log_e)
        lr = np.log10(ref.loc[z].values)
        # finite-difference exponent of the network on the same stencil as the reference
        zl, zh = max(z - 1, 1), min(z + 1, 9)
        n_fd = (acs.log_sigma(np.full_like(log_e, zh), log_e) - acs.log_sigma(np.full_like(log_e, zl), log_e)) \
            * LN10 / (np.log(zh) - np.log(zl))
        R_nn = 10**ls / (z**2 * ref.loc[1].values)
        acs_rows.append({'Z': z, 'MAE_log_sigma': mae(ls[valid], lr[valid]), 'MBE_log_sigma': float(np.mean(ls[valid] - lr[valid])),
                         'MAE_log_R': mae(np.log10(R_nn[valid]), np.log10(R_ref.loc[z].values[valid])),
                         'n_eff_FD_MAE_vs_ref': mae(n_fd[valid], ref_n.loc[z].values[valid]),
                         'n_eff_exact_MAE_vs_ref_FD': mae(n_nn[valid], ref_n.loc[z].values[valid]),
                         'n_eff_ref_E>=1e4': float(np.median(ref_n.loc[z].values[energies >= 1e4])),
                         'n_eff_NN_E>=1e4': float(np.median(n_nn[energies >= 1e4]))})
    acs_tab = pd.DataFrame(acs_rows)
    acs_tab.to_csv(os.path.join(OUT, 'acsnn_scaling_metrics.csv'), index=False)
    fig_reference_and_acsnn(ref, acs, e_grid)
    fig_exponent_acsnn(ref_n, acs, e_grid)

    # ---------------- hole models
    case_rows, hole_curves, hole_n = [], {}, {}
    for name in model_names():
        excluded, net = load_hole(name, theories_all)
        train = [t for t in ZS if t not in excluded]
        for z in excluded:
            c = classify(z, train)
            c = 'interp. gap 3' if c.startswith('interp. gap 3') else c
            ls = net.log_sigma(np.full_like(log_e, z), log_e)
            n_nn = net.n_eff(np.full_like(log_e, z), log_e)
            zl, zh = max(z - 1, 1), min(z + 1, 9)
            n_fd = (net.log_sigma(np.full_like(log_e, zh), log_e) - net.log_sigma(np.full_like(log_e, zl), log_e)) \
                * LN10 / (np.log(zh) - np.log(zl))
            born, z2, power, (a, b) = baselines(z, train, ref)
            lr = np.log10(ref.loc[z].values)
            v = valid
            case_rows.append({'model': name, 'n_excluded': len(excluded), 'Z': z, 'case_class': c,
                              'power_law_charges': f'{a},{b}',
                              'MAE_NN': mae(ls[v], lr[v]), 'MBE_NN': float(np.mean(ls[v] - lr[v])),
                              'MAE_Born': mae(born[v], lr[v]) if 1 in train else np.nan,
                              'MBE_Born': float(np.mean(born[v] - lr[v])) if 1 in train else np.nan,
                              'MAE_Z2': mae(z2[v], lr[v]), 'MBE_Z2': float(np.mean(z2[v] - lr[v])),
                              'MAE_power': mae(power[v], lr[v]), 'MBE_power': float(np.mean(power[v] - lr[v])),
                              'n_eff_MAE': mae(n_fd[v], ref_n.loc[z].values[v]),
                              'n_eff_MBE': float(np.mean(n_fd[v] - ref_n.loc[z].values[v])),
                              'n_eff_exact_minus_ref_FD_MBE': float(np.mean(n_nn[v] - ref_n.loc[z].values[v]))})
            hole_curves.setdefault((z, c), []).append((name, ls))
            hole_n.setdefault((z, c), []).append(n_nn)
    cases = pd.DataFrame(case_rows)
    cases['NN_better_than_Z2'] = cases['MAE_NN'] < cases['MAE_Z2']
    cases['NN_better_than_power'] = cases['MAE_NN'] < cases['MAE_power']
    cases.to_csv(os.path.join(OUT, 'holes_cases.csv'), index=False)

    def summary(g):
        out = {'cases': len(g)}
        for k in ['MAE_NN', 'MAE_Born', 'MAE_Z2', 'MAE_power', 'n_eff_MAE']:
            out[f'median_{k}'] = g[k].median()
        for k in ['MBE_NN', 'MBE_Born', 'MBE_Z2', 'MBE_power', 'n_eff_MBE']:
            out[f'median_{k}'] = g[k].median()
        gb = g.dropna(subset=['MAE_Born'])
        out['cases_with_Born'] = len(gb)
        out['pct_NN_better_than_Born'] = 100 * (gb['MAE_NN'] < gb['MAE_Born']).mean() if len(gb) else np.nan
        out['pct_NN_better_than_Z2'] = 100 * g['NN_better_than_Z2'].mean()
        out['pct_NN_better_than_power'] = 100 * g['NN_better_than_power'].mean()
        for base in ['Z2', 'power']:
            d = g['MAE_NN'] - g[f'MAE_{base}']
            out[f'wilcoxon_p_vs_{base}'] = stats.wilcoxon(g['MAE_NN'], g[f'MAE_{base}']).pvalue if len(g) >= 6 and (d != 0).any() else np.nan
        return pd.Series(out)

    by_class = cases.groupby('case_class').apply(summary, include_groups=False).reindex(
        [c for c in CASE_ORDER if c in cases['case_class'].values]).reset_index()
    by_n_class = cases.groupby(['n_excluded', 'case_class']).apply(summary, include_groups=False).reset_index()
    by_z = cases.groupby('Z').apply(summary, include_groups=False).reset_index()
    for nm, t in [('holes_by_case_class', by_class), ('holes_by_n_excluded_and_case_class', by_n_class), ('holes_by_charge', by_z)]:
        t.to_csv(os.path.join(OUT, f'{nm}.csv'), index=False)

    fig_holes_ratio(ref, hole_curves)
    fig_holes_exponent(ref_n, {k: v for k, v in hole_n.items()})
    fig_baselines(cases)
    fig_exponent_error(cases)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    with open(os.path.join(OUT, 'console_output.txt'), 'w') as fp:
        for title, t in [('reference scaling by band', ref_tab), ('ACSNN', acs_tab), ('holes by case class', by_class),
                         ('holes by n excluded and class', by_n_class), ('holes by charge', by_z)]:
            txt = f'\n=== {title} ===\n' + t.round(4).to_string(index=False)
            print(txt); fp.write(txt + '\n')
        rho, p = stats.spearmanr(cases['n_eff_MAE'], cases['MAE_NN'])
        txt = f'\nSpearman(n_eff error, MAE_NN) rho={rho:.3f} p={p:.2e}'
        print(txt); fp.write(txt + '\n')
