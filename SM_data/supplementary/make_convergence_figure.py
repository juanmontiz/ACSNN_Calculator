"""Training-convergence figure for the supplementary material (models *_b100k)."""
import glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join('supplementary', 'figures', 'training_convergence.png')
hist = {1: [], 2: [], 3: []}
for f in sorted(glob.glob('history_hole_*_b100k.npz')):
    h = np.load(f)
    hist[len(set(os.path.basename(f)[13:16]))].append((h['train'], h['val']))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ep = np.arange(1, 100001)
for k, col in zip((1, 2, 3), ('tab:blue', 'tab:green', 'tab:red')):
    tr = np.array([t for t, _ in hist[k]]); va = np.array([v for _, v in hist[k]])
    best = np.minimum.accumulate(va, axis=1)
    q = np.percentile(best, [25, 50, 75], axis=0)
    ax[0].plot(ep, q[1], color=col, label=f'{k} excluded ({len(hist[k])} models)')
    ax[0].fill_between(ep, q[0], q[2], color=col, alpha=0.2)
    s = np.unique(np.logspace(0, 5, 400).astype(int)) - 1
    ax[1].plot(ep[s], np.median(tr, axis=0)[s], color=col, lw=1, label=f'{k} excluded, training (dropout on)')
    ax[1].plot(ep[s], np.median(va, axis=0)[s], color=col, lw=1, ls='--', label=f'{k} excluded, validation (dropout off)')
    epochs_best = va.argmin(axis=1) + 1
    ax[0].plot(np.median(epochs_best), np.median(best[:, -1]), 'v', color=col, ms=9, mec='k')
for a in ax:
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('Epoch'); a.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)
ax[0].set_ylabel('Lowest validation loss so far'); ax[0].set_title('Best validation loss (median, IQR band; ▼ = median epoch of the minimum)')
ax[1].set_ylabel('Loss per epoch (median over models)'); ax[1].set_title('Training vs validation loss')
ax[0].legend(fontsize=8); ax[1].legend(fontsize=7, ncol=1)
ax[0].title.set_fontsize(10); ax[1].title.set_fontsize(10)
fig.tight_layout(); fig.savefig(OUT, dpi=130); plt.close(fig)
print('saved', OUT)
