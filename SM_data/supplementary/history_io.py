"""Training histories of the hole networks (SM Fig. S1).

The full histories, history_hole_XYZ_b100k.npz in the root folder (10^5 epochs, 0.8 MB each, 99 MB in total), are
written by train_totals_NN_holes.py and are not distributed. The repository ships reduced versions,
histories/history_hole_XYZ_b100k_reduced.npz (a few kB each), written by supplementary/reduce_histories.py. They keep
exactly what Fig. S1 plots:

    epoch         epochs plotted in Fig. S1 (SAMPLE + 1, log-spaced between 1 and 10^5)
    train, val    training and validation loss at those epochs
    best          lowest validation loss up to each of those epochs (running minimum over all epochs)
    argmin_epoch  epoch of the lowest validation loss
    min_val       lowest validation loss

load_histories() reads the full files when they are present (e.g. after retraining) and the reduced ones otherwise;
both give the same Fig. S1.
"""
import glob
import os

import numpy as np

N_EPOCHS = 100000
SAMPLE = np.unique(np.logspace(0, 5, 400).astype(int)) - 1     # 0-based indices of the epochs plotted in Fig. S1
FULL = 'history_hole_*_b100k.npz'
REDUCED_DIR = 'histories'
REDUCED = os.path.join(REDUCED_DIR, 'history_hole_*_b100k_reduced.npz')


def n_excluded(path):
    """Number of excluded charges of a network, from the tag XYZ of its file name (333 -> 1, 335 -> 2, 357 -> 3)."""
    return len(set(os.path.basename(path)[13:16]))


def reduce(train, val):
    """Reduced history of one network from its full training and validation loss per epoch."""
    assert len(train) == len(val) == N_EPOCHS
    best = np.minimum.accumulate(val)
    return dict(epoch=(SAMPLE + 1).astype(np.int32), train=train[SAMPLE], val=val[SAMPLE], best=best[SAMPLE],
                argmin_epoch=np.int32(val.argmin() + 1), min_val=val.min())


def load_histories():
    """Reduced histories grouped by the number of excluded charges: {1: [dict, ...], 2: [...], 3: [...]}."""
    files = sorted(glob.glob(FULL))
    full = bool(files)
    if not full:
        files = sorted(glob.glob(REDUCED))
    if not files:
        raise FileNotFoundError(f'no training histories found ({FULL} or {REDUCED}); run from the root folder')
    hist = {1: [], 2: [], 3: []}
    for f in files:
        h = np.load(f)
        hist[n_excluded(f)].append(reduce(h['train'], h['val']) if full else {k: h[k] for k in h.files})
    return hist
