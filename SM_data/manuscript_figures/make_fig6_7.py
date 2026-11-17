#!/usr/bin/env python3
"""Manuscript Figs. 6 and 7 regenerated with the SM single-hole networks (series b100k).

Same layout as the submitted performance_gausshole_{3,9}.pdf: (a) CDW-EIS, (b) CTMC;
network curve, calculated TCS (stars) and percentage relative error as a filled curve
on the right axis (0-20 %). Networks: best_statedict_gauss_128_hole_{333,999}_b100k.pth,
loaded with evaluate_holes.load_model from the package root, i.e. exactly
the networks of SM Sec. S7, Table S3 and Fig. S2.

Usage: python3 make_fig6_7.py   -> performance_gausshole_{3,9}.pdf/.png + error summary
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent                      # package root (models, dbs, evaluate_holes.py)
os.environ["MODEL_SET"] = "b100k"
os.chdir(CODE)
sys.path.insert(0, str(CODE))

import logging
logging.disable(logging.WARNING)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate_holes import NAMELIST, read_all_databases, load_model, model_path

plt.rcParams.update({"font.size": 16, "font.family": "serif",
                     "font.serif": ["TeX Gyre Pagella", "P052", "Palatino"],
                     "mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Pagella",
                     "mathtext.it": "TeX Gyre Pagella:italic", "pdf.fonttype": 42})
TEAL, CYAN, AQUA, GOLD = "#005f73", "#0a9396", "#94d2bd", "#ee9b00"
E_RANGE = {"CDW-EIS": (10.0, 1e5), "CTMC": (14.85, 2498.0)}

df = read_all_databases(NAMELIST)
df = df[df["Z_target"] == 1].reset_index(drop=True)
df["log_E"] = np.log10(df["E_proj[keV/u]"].values)


def style(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=True, width=1.5, length=4)
    ax.tick_params(axis="both", which="minor", length=2, width=1)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


for z in (3, 9):
    m = load_model(model_path(f"{z}{z}{z}"), df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    print(f"--- Z_P = {z} (hole {z}{z}{z}_b100k)")
    for ax, th, letter in zip(axes, ("CDW-EIS", "CTMC"), "ab"):
        d = df[(df.Z_ion == z) & (df.Theory_ID == th)].sort_values("log_E")
        E, sref = d["E_proj[keV/u]"].values, d["Cross_section[cm2]"].values
        snn = 10 ** m["predict"](d)
        lo, hi = E_RANGE[th]
        g = pd.DataFrame({"Z_ion": z, "log_E": np.linspace(np.log10(lo), np.log10(hi), 300),
                          "Theory_ID": th})
        ax.loglog(10 ** g.log_E, 10 ** m["predict"](g), lw=3, color=TEAL, zorder=3)
        ax.loglog(E, sref, "*", ms=11, color=GOLD, mec="w", mew=0.5, zorder=4)
        err = 100 * np.abs(snn / sref - 1)
        axr = ax.twinx()
        axr.fill_between(E, np.minimum(err, 30), color=AQUA, alpha=0.5, lw=0)
        axr.plot(E, np.minimum(err, 30), color=CYAN, lw=1.5)
        axr.set_ylim(0, 20)
        axr.tick_params(axis="y", direction="in", labelcolor=CYAN, width=1.5, length=4)
        if letter == "b":
            axr.set_ylabel("relative error (%)", color=CYAN)
        ax.set_zorder(axr.get_zorder() + 1); ax.patch.set_visible(False)
        style(ax)
        ax.set_xlabel("Projectile Energy (keV/u)")
        if letter == "a":
            ax.set_ylabel(r"Total cross section (cm$^2$)")
        ax.text(0.04, 0.95, f"({letter})", transform=ax.transAxes, fontsize=20, va="top")
        ax.text(0.40, 0.55, rf"$Z_P$={z}", transform=ax.transAxes, fontsize=20)
        # summary for the text
        hi25 = E >= 25
        print(f"  {th:8s} median {np.median(err):5.1f}%  max(E>=25) {err[hi25].max():5.1f}% at "
              f"{E[hi25][np.argmax(err[hi25])]:.0f} keV/u;  max overall {err.max():.1f}% at {E[np.argmax(err)]:.0f} keV/u;"
              f"  E>=100: max {err[E >= 100].max():.1f}%")
    fig.savefig(HERE / f"performance_gausshole_{z}.pdf")
    fig.savefig(HERE / f"performance_gausshole_{z}.png", dpi=110)
print("saved")
