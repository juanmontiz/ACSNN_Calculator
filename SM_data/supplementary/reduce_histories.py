"""Write the reduced training histories used for SM Fig. S1 (see history_io.py).

Reads the full histories history_hole_XYZ_b100k.npz written by train_totals_NN_holes.py and writes
histories/history_hole_XYZ_b100k_reduced.npz. Run from the root folder: python3 supplementary/reduce_histories.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_io import FULL, REDUCED_DIR, reduce

files = sorted(glob.glob(FULL))
if not files:
    sys.exit(f'no full histories ({FULL}) in {os.getcwd()}')
os.makedirs(REDUCED_DIR, exist_ok=True)
for f in files:
    h = np.load(f)
    out = os.path.join(REDUCED_DIR, os.path.basename(f).replace('.npz', '_reduced.npz'))
    np.savez_compressed(out, **reduce(h['train'], h['val']))
print(f'{len(files)} reduced histories written to {REDUCED_DIR}/')
