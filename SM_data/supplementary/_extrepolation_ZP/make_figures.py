"""Figures and numbers for the Z_P = 1 vs Z_P = 9 extrapolation analysis (companion to Fig. S2).

Usage:  python3 make_figures.py
Reads the CDW-EIS and CTMC H-target databases, writes
  fig_S2a_charge_dependence.{pdf,png}
  fig_S2b_edge_extrapolation.{pdf,png}
  numbers.txt   (values quoted in extrapolation_ZP.tex)
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"  # copies of z_Paper_v1/NN-ion-papiro-main/dbs/{CDWEIS,ctmc}/*.csv
F_CDW = DATA / "cdw-eis_TCS_database_single_ionization_atoms_except_Kr.csv"
F_CTMC = DATA / "ctmc_TCS_database_single_ionization_hydrogen.csv"

# NN bias on the excluded charge, Sec. S6 / Table S4 (single-charge exclusion)
NN_Z1_MBE, NN_Z1_LO, NN_Z1_HI = 0.412, np.log10(2.0), np.log10(3.6)  # +100% .. +260%
NN_Z9_MBE = 0.010

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "axes.titlesize": 9, "savefig.bbox": "tight"})


def load(path):
    d = pd.read_csv(path)
    d = d[d.Z_target == 1]
    p = d.pivot(index="E_proj[keV/u]", columns="Z_ion", values="Cross_section[cm2]")
    return np.log10(p.sort_index())  # rows: E, cols: Z = 1..9


L_cdw, L_ctmc = load(F_CDW), load(F_CTMC)
Z = np.arange(1, 10)
lnZ = np.log(Z)


def edge_bias(L):
    """Bias (prediction - reference) in log10 sigma of three neighbour-based extrapolations."""
    Y = L.values

    def lin_ln(i, j, k):  # line through charges i, j (0-based) in ln Z, evaluated at k
        return Y[:, i] + (Y[:, i] - Y[:, j]) * (lnZ[k] - lnZ[i]) / (lnZ[i] - lnZ[j])

    return {
        "flat": (Y[:, 1] - Y[:, 0], Y[:, 7] - Y[:, 8]),
        "linZ": (2 * Y[:, 1] - Y[:, 2] - Y[:, 0], 2 * Y[:, 7] - Y[:, 6] - Y[:, 8]),
        "linlnZ": (lin_ln(1, 2, 0) - Y[:, 0], lin_ln(7, 6, 8) - Y[:, 8]),
    }


def n_eff(L, z):
    """Local charge exponent between z and z+1: d ln sigma / d ln Z_P (finite difference)."""
    return (L[z + 1] - L[z]) / np.log10((z + 1) / z)


# ---------------------------------------------------------------- Fig. S2a
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
Es = [50, 100, 500, 5000, 100000]
cols = plt.cm.viridis(np.linspace(0, 0.9, len(Es)))
for E, c in zip(Es, cols):
    y = L_cdw.loc[E].values - L_cdw.loc[E].values[0]
    lab = f"{E/1000:g} MeV/u" if E >= 1000 else f"{E:g} keV/u"
    ax[0].plot(Z, y, "o-", c=c, ms=3, lw=1.1, label=lab)
    ax[1].plot(Z, y, "o-", c=c, ms=3, lw=1.1)
    ax[1].plot(Z, 2 * np.log10(Z), ":", c="0.5", lw=0.8) if E == Es[-1] else None
ax[0].set_xlabel(r"$Z_P$ (linear axis, as seen by the NN)")
ax[1].set_xlabel(r"$Z_P$ (logarithmic axis)")
ax[1].set_xscale("log")
ax[1].minorticks_off()
for a in ax[:2]:
    a.set_ylabel(r"$\log_{10}[\sigma(Z_P)/\sigma(1)]$")
ax[0].set_xticks(Z)
ax[1].set_xticks([1, 2, 3, 4, 5, 6, 9]); ax[1].set_xticklabels([1, 2, 3, 4, 5, 6, 9])
ax[1].text(5.2, 1.05, r"$Z_P^2$", color="0.4", fontsize=8)
fig.legend(*ax[0].get_legend_handles_labels(), loc="outside upper center", ncol=5, frameon=False)
pairs = [(1, "C3"), (2, "C1"), (4, "C2"), (8, "C0")]
for z, c in pairs:
    ax[2].plot(L_cdw.index, n_eff(L_cdw, z), "-", c=c, lw=1.2, label=f"$Z_P={z}\\to{z+1}$")
    ax[2].plot(L_ctmc.index, n_eff(L_ctmc, z), "s", c=c, ms=2.5, mfc="none", mew=0.7)
ax[2].axhline(2, c="k", lw=0.6, ls="--")
ax[2].set_xscale("log"); ax[2].set_ylim(-1, 2.6)
ax[2].set_xlabel(r"$E$ (keV/u)"); ax[2].set_ylabel(r"$n_{\rm eff}$")
ax[2].legend(loc="lower right", frameon=False, title="lines: CDW-EIS\nsquares: CTMC", title_fontsize=7)
for a, t in zip(ax, "abc"):
    a.grid(alpha=0.3, lw=0.4)
    a.text(0.03, 0.96, f"({t})", transform=a.transAxes, va="top", fontweight="bold")
fig.savefig(HERE / "fig_S2a_charge_dependence.pdf"); fig.savefig(HERE / "fig_S2a_charge_dependence.png", dpi=220)

# ---------------------------------------------------------------- Fig. S2b
B_cdw, B_ctmc = edge_bias(L_cdw), edge_bias(L_ctmc)
lab = {"flat": "constant (copy neighbour)", "linZ": r"linear in $Z_P$", "linlnZ": r"linear in $\ln Z_P$"}
col = {"flat": "C3", "linZ": "C0", "linlnZ": "C2"}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True, constrained_layout=True)
for k in ("flat", "linZ", "linlnZ"):
    for i in (0, 1):
        ax[i].plot(L_cdw.index, B_cdw[k][i], "-", c=col[k], lw=1.3, label=lab[k] if i == 0 else None)
        ax[i].plot(L_ctmc.index, B_ctmc[k][i], "s", c=col[k], ms=2.5, mfc="none", mew=0.7)
# exact NN predictions (provided by the authors) take precedence over the curves digitized from Fig. S2
F_NN = next((f for f in (HERE / "nn_edge_predictions.csv", HERE / "nn_edge_digitized.csv") if f.exists()), HERE / "none")
if F_NN.exists():
    nn = pd.read_csv(F_NN)
    nn["bias"] = nn.log10_sigma_nn - nn.log10_sigma_ref
    for i, h in enumerate((1, 9)):
        for th, ls, mk in (("CDW-EIS", "-", "o"), ("CTMC", "none", "^")):
            g = nn[(nn.hole == h) & (nn.theory == th)].sort_values("E")
            ax[i].plot(g.E, g.bias, ls=ls, marker=mk, c="k", lw=1.0, ms=2.5, mfc="k" if th == "CDW-EIS" else "none",
                       mew=0.8, label=f"NN, Fig. S2 ({th})" if i == 0 else None)
else:
    ax[0].axhspan(NN_Z1_LO, NN_Z1_HI, color="k", alpha=0.10, lw=0)
    ax[0].axhline(NN_Z1_MBE, c="k", lw=1.0, ls="--", label="NN (Sec. S6)")
    ax[1].axhline(NN_Z9_MBE, c="k", lw=1.0, ls="--")
ax[0].set_title(r"$Z_P = 1$ extrapolated from $Z_P = 2, 3$")
ax[1].set_title(r"$Z_P = 9$ extrapolated from $Z_P = 8, 7$")
for a, t in zip(ax, "ab"):
    a.axhline(0, c="k", lw=0.5); a.set_xscale("log"); a.set_ylim(-0.3, 0.8)
    a.axvspan(10, 50, color="0.85", alpha=0.5, lw=0)
    a.set_xlabel(r"$E$ (keV/u)"); a.grid(alpha=0.3, lw=0.4)
    a.text(0.03, 0.96, f"({t})", transform=a.transAxes, va="top", fontweight="bold")
ax[0].set_ylabel(r"bias in $\log_{10}\sigma$ (prediction $-$ reference)")
ax[0].legend(loc="lower right", frameon=False)
fig.savefig(HERE / "fig_S2b_edge_extrapolation.pdf"); fig.savefig(HERE / "fig_S2b_edge_extrapolation.png", dpi=220)

# ---------------------------------------------------------------- numbers for the text
with open(HERE / "numbers.txt", "w") as f:
    for name, L, B in (("CDW-EIS", L_cdw, B_cdw), ("CTMC", L_ctmc, B_ctmc)):
        m = L.index >= 50
        f.write(f"== {name}: mean bias over E >= 50 keV/u ({m.sum()} energies)\n")
        for k in B:
            f.write(f"  {k:7s}  Z=1: {B[k][0][m].mean():+.3f}   Z=9: {B[k][1][m].mean():+.3f}\n")
        f.write(f"  Delta log10 sigma (2-1): {(L[2]-L[1])[m].mean():.3f}  (8->9): {(L[9]-L[8])[m].mean():.3f}\n")
        for E in (100, 1000, 100000):
            if E in L.index:
                f.write(f"  E={E:g}: n_eff(1->2)={n_eff(L,1)[E]:.2f} n_eff(8->9)={n_eff(L,8)[E]:.2f} "
                        f"flat Z1={B['flat'][0][L.index.get_loc(E)]:+.3f}\n")
print((HERE / "numbers.txt").read_text())
