# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT

"""Configuration paths and constants for ACSNN Calculator."""

import os
from pathlib import Path

_BASE = Path(__file__).parent.parent

MODEL_PATH = os.environ.get('MODEL_PATH', str(_BASE / 'models_folder'))
DATA_PATH = os.environ.get('DATA_PATH', str(_BASE / 'dbs'))
EXPERIMENTAL_DATA_PATH = os.path.join(DATA_PATH, 'Experimental_data_db')

DEFAULT_DEVICE = 'cpu'
DEFAULT_X_MIN = 100
DEFAULT_X_MAX = 1000
DEFAULT_N_POINTS = 100

# NN training energy range in keV/u (per theory)
TRAIN_E_MIN = {"CDW-EIS": 10.0,     "CTMC": 15.0,   "Semiempiric_1985Rudd": 1.0}
TRAIN_E_MAX = {"CDW-EIS": 100000.0, "CTMC": 2500.0, "Semiempiric_1985Rudd": 10000.0}

# Safe Z_proj limit (max trained per theory)
# CDW-EIS and CTMC safe up to Z_proj=9; Semiempiric only for Z_proj=1
ZPROJ_SAFE = {"CDW-EIS": 9, "CTMC": 9, "Semiempiric_1985Rudd": 1}

# Physical validity domain of each theory (soft limit: the NN value is returned,
# with a notice, because the theory itself is unreliable there; see the paper).
# E [keV/u] of a projectile with velocity v = 1 a.u.
E_PER_AU_VELOCITY2 = 24.98
# CDW-EIS: perturbative regime, Sommerfeld parameter eta = Z_proj/v < 1,
#          i.e. E > E_PER_AU_VELOCITY2 * Z_proj**2 keV/u.
# CTMC: intermediate energies (approximate window, boundaries are blurred).
# Semiempiric_1985Rudd: no restriction inside its training range.
CTMC_VALID_E = (75.0, 500.0)
