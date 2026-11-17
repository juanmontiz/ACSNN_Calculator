"""Training-convergence figure for the supplementary material (models *_b100k)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join('supplementary', 'figures', 'training_convergence.png')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_io import load_histories
hist = load_histories()   # full histories if present, otherwise the reduced ones in histories/

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for k, col in zip((1, 2, 3), ('tab:blue', 'tab:green', 'tab:red')):
    H = hist[k]
    ep = H[0]['epoch']
    tr = np.array([h['train'] for h in H]); va = np.array([h['val'] for h in H])
    best = np.array([h['best'] for h in H])
    q = np.percentile(best, [25, 50, 75], axis=0)
    ax[0].plot(ep, q[1], color=col, label=f'{k} excluded ({len(H)} models)')
    ax[0].fill_between(ep, q[0], q[2], color=col, alpha=0.2)
    ax[1].plot(ep, np.median(tr, axis=0), color=col, lw=1, label=f'{k} excluded, training (dropout on)')
    ax[1].plot(ep, np.median(va, axis=0), color=col, lw=1, ls='--', label=f'{k} excluded, validation (dropout off)')
    ax[0].plot(np.median([h['argmin_epoch'] for h in H]), np.median([h['min_val'] for h in H]), 'v', color=col, ms=9, mec='k')
for a in ax:
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('Epoch'); a.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)
ax[0].set_ylabel('Lowest validation loss so far'); ax[0].set_title('Best validation loss (median, IQR band; ▼ = median epoch of the minimum)')
ax[1].set_ylabel('Loss per epoch (median over models)'); ax[1].set_title('Training vs validation loss')
ax[0].legend(fontsize=8); ax[1].legend(fontsize=7, ncol=1)
ax[0].title.set_fontsize(10); ax[1].title.set_fontsize(10)
fig.tight_layout(); fig.savefig(OUT, dpi=130); plt.close(fig)
print('saved', OUT)
