"""Regenerate the baseline figures of the supplementary material from the saved per-case tables.

Run from the project root: MODEL_SET=b100k python3 supplementary/make_baseline_figures.py
"""
import os, shutil, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import zp2_analysis
import sommerfeld_analysis

zp2_analysis.fig_baselines(pd.read_csv(os.path.join(zp2_analysis.OUT, 'holes_cases.csv')))
sommerfeld_analysis.fig_baselines(pd.read_csv(os.path.join(sommerfeld_analysis.OUT, 'baselines_eta_lt1_cases.csv')))
shutil.copy(os.path.join(zp2_analysis.OUT, 'holes_vs_physics_baselines.png'), 'supplementary/figures/zp2_holes_vs_physics_baselines.png')
shutil.copy(os.path.join(sommerfeld_analysis.OUT, 'baselines_eta_lt1.png'), 'supplementary/figures/eta_baselines_eta_lt1.png')
