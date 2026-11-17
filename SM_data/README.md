# ACSNN — analysis code and data of the Supplementary Material

Code, data and trained networks used for the Supplementary Material (SM) of

> M. A. Quinto, J. M. Monti and S. D. López, *ACSNN: Neural Network Surrogates for Ionization Cross Sections in
> Heavy Ion–Hydrogen Collisions*, J. Phys. B: At. Mol. Opt. Phys. (submitted).

The SM tests how well networks of the ACSNN architecture predict the total single-ionization cross section of
H(1s) by bare ions for projectile charges **absent from training**: 129 networks, one for every set of one, two
and three excluded charges ($\binom91+\binom92+\binom93=9+36+84$), each evaluated on its excluded charges, plus
the charge-scaling ($Z_P^2$) analysis, the perturbative regime of CDW-EIS ($\eta=Z_P/v<1$) and the extrapolation
to $Z_P=1$ and $Z_P=9$.

The ACSNN calculator itself (the model of the main article, for end users) is in the root of this repository:
<https://github.com/juanmontiz/ACSNN_Calculator>.

## Contents

| Path | Content |
|---|---|
| `dbs/` | Reference total cross sections: CDW-EIS (`CDWEIS/`), CTMC (`ctmc/`), Rudd semi-empirical (`Semiempiric_db/`) and experimental data (`Experimental_data_db/`). |
| `train_totals_NN.py` | Training of the final ACSNN model (CDW-EIS + CTMC + Rudd, all charges) → `best_statedict_gauss_128.pth`. |
| `train_totals_NN_holes.py` | Training of the 129 hole networks (CDW-EIS + CTMC, no Rudd; 10⁵ epochs, seed 17) → `best_statedict_gauss_128_hole_XYZ_b100k.pth`, `history_hole_XYZ_b100k.npz`. |
| `best_statedict_gauss_128.pth` | Final ACSNN model (main article). |
| `best_statedict_gauss_128_hole_XYZ_b100k.pth` | The 129 hole networks. `XYZ` lists the excluded charges, padded with the lowest one: `333` = only $Z_P=3$ excluded, `335` = 3 and 5, `357` = 3, 5 and 7. |
| `histories/history_hole_XYZ_b100k_reduced.npz` | Reduced training/validation loss histories of the 129 hole networks (Fig. S1; 1.1 MB in total): losses at the ~330 log-spaced epochs plotted in Fig. S1, running minimum of the validation loss, and epoch and value of the minimum (see `supplementary/history_io.py`). The full histories (`history_hole_XYZ_b100k.npz`, 10⁵ epochs, 99 MB in total) are not distributed: retraining (step 0) writes them and `supplementary/reduce_histories.py` reduces them. Fig. S1 is identical with either. |
| `evaluate_holes.py`, `evaluate_two_charges.py`, `evaluate_three_charges.py` | Evaluation of the networks with one, two and three excluded charges → `eval_holes_b100k/`. |
| `zp2_analysis.py` | $Z_P^2$ scaling of CDW-EIS, ACSNN interpolator and scaling baselines → `eval_zp2_b100k/`. |
| `sommerfeld_analysis.py` | Perturbative regime $\eta<1$ of CDW-EIS → `eval_sommerfeld_b100k/`. |
| `eval_*_b100k/` | Pre-computed outputs of the three steps above (tables as CSV, diagnostic plots). |
| `supplementary/` | Scripts that produce the SM figures and tables from the outputs above. |
| `supplementary/_extrepolation_ZP/` | Edge-extrapolation analysis ($Z_P=1$ vs $Z_P=9$, SM Sec. S12). |
| `manuscript_figures/` | Main-article Fig. 5 (`make_combined_fit_figure.py`) and Figs. 6–7 (`make_fig6_7.py`). |

## Requirements

Python ≥ 3.10 and the packages in `requirements.txt` (`pip install -r requirements.txt`).
The analysis was run with Python 3.14, PyTorch 2.12, scikit-learn 1.8, pandas 3.0, NumPy 2.4, Matplotlib 3.10
and SciPy 1.17. Training uses a GPU if available (CUDA); evaluation runs on CPU in a few minutes.

`supplementary/make_sm_figures.py` and `manuscript_figures/make_combined_fit_figure.py` render text with LaTeX
(`text.usetex`): a TeX installation with the `type1cm` and `type1ec` packages is required
(e.g. Fedora: `dnf install texlive-type1cm texlive-cm-super`; Debian/Ubuntu: `apt install cm-super texlive-latex-extra`).

## Reproducing the results

All commands are run **from the root of this folder**. The networks are selected with the environment
variable `MODEL_SET=b100k`.

```bash
# 0) (optional, long) retrain the networks; existing .pth files with the same name are skipped
python3 train_totals_NN.py                       # final ACSNN model
EPOCHS=100000 python3 train_totals_NN_holes.py   # 129 hole networks
python3 supplementary/reduce_histories.py        # reduced histories for Fig. S1 (histories/)

# 1) evaluation of the hole networks (SM Secs. S7-S9, Tables S3-S9)
MODEL_SET=b100k python3 evaluate_holes.py
MODEL_SET=b100k python3 evaluate_two_charges.py
MODEL_SET=b100k python3 evaluate_three_charges.py

# 2) charge scaling and perturbative regime (SM Secs. S10-S11, Tables S10-S13)
MODEL_SET=b100k python3 zp2_analysis.py
MODEL_SET=b100k python3 sommerfeld_analysis.py
MODEL_SET=b100k python3 supplementary/make_hole_ratio_table.py

# 3) SM figures (S1-S9)
MODEL_SET=b100k python3 supplementary/make_sm_figures.py          # or a subset: ... S1 S2

# 4) edge extrapolation, Z_P = 1 vs 9 (SM Sec. S12, Table S14)
MODEL_SET=b100k python3 supplementary/_extrepolation_ZP/export_nn_edge_predictions.py
python3 supplementary/_extrepolation_ZP/make_figures.py

# 5) main-article figures
python3 manuscript_figures/make_combined_fit_figure.py   # Fig. 5
python3 manuscript_figures/make_fig6_7.py                # Figs. 6-7
```

Steps 1–5 use the trained networks included here, so step 0 is not needed to reproduce the SM. Training is not
bit-reproducible across hardware: the training/validation split is not seeded (SM Sec. S4), so retrained
networks differ slightly from the ones provided, within the training-to-training variability discussed in
SM Sec. S9.

## Data

The CDW-EIS and CTMC cross sections were calculated by the authors (main article, Secs. 2.1–2.2); the Rudd
cross sections follow Rudd *et al.*, Rev. Mod. Phys. **57**, 965 (1985). The experimental data in
`dbs/Experimental_data_db/` are compiled from the publications cited in the main article and are included only
for convenience.

## License and citation

Code under the MIT license (`LICENSE`). If you use this material, please cite the article above
(see `CITATION.cff`).
