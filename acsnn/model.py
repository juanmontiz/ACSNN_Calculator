# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT

"""Pure NumPy neural network for atomic ionization cross-section prediction."""

import logging
import sys
import numpy as np
from acsnn.config import TRAIN_E_MIN, TRAIN_E_MAX, ZPROJ_SAFE

log = logging.getLogger(__name__)
if not log.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    if sys.stderr.isatty():
        import platform
        _YELLOW = "\033[33m" if platform.system() != "Windows" else ""
        _RED = "\033[31m" if platform.system() != "Windows" else ""
        _BOLD = "\033[1m" if platform.system() != "Windows" else ""
        _RESET = "\033[0m" if platform.system() != "Windows" else ""
        class _Fmt(logging.Formatter):
            _MAP = {logging.WARNING: _YELLOW, logging.ERROR: _RED}
            def format(self, r):
                c = self._MAP.get(r.levelno)
                return (c + _BOLD + "[ACSNN] " + _RESET + c +
                        "%(levelname)s: %(message)s" % {"levelname": r.levelname, "message": r.getMessage()} + _RESET) if c else logging.Formatter.format(self, r)
        _handler.setFormatter(_Fmt())
    else:
        _handler.setFormatter(logging.Formatter("[ACSNN] %(levelname)s: %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.WARNING)

_THEORIES = ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"]
_ONE_HOT = {t: np.eye(3, dtype=np.float32)[i] for i, t in enumerate(_THEORIES)}


def _gaussian(x):
    """Gaussian activation function: exp(-x^2)."""
    return np.exp(-x**2)


class FlexibleNN:
    """
    Pure NumPy implementation of the FlexibleNN model.
    
    Architecture: 3 layers (lin1, lin2, lin5) with Gaussian activation.
    Input: 5 dimensions [Z_proj, log10(energy), theory_one_hot(3)]
    Output: 1 dimension (log10 of cross section)
    """
    
    def __init__(self, weights_path):
        """
        Load model weights from NumPy file.
        
        Parameters:
        -----------
        weights_path : str
            Path to .npz file containing model weights
        """
        data = np.load(weights_path)
        
        # Load 3-layer architecture
        self._layers = [
            (data['lin1_weight'], data['lin1_bias']),
            (data['lin2_weight'], data['lin2_bias']),
            (data['lin5_weight'], data['lin5_bias']),
        ]
        
        # Store input dimension for validation
        self.input_dim = self._layers[0][0].shape[1]

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : np.ndarray
            Input features, shape (n_samples, input_dim)
            
        Returns:
        --------
        output : np.ndarray
            Model predictions, shape (n_samples, 1)
        """
        # Layer 1: Linear + Gaussian activation
        x = x @ self._layers[0][0].T + self._layers[0][1]
        x = _gaussian(x)
        
        # Layer 2: Linear + Gaussian activation
        x = x @ self._layers[1][0].T + self._layers[1][1]
        x = _gaussian(x)
        
        # Layer 5 (output): Linear, no activation
        x = x @ self._layers[2][0].T + self._layers[2][1]
        
        return x


def predict_conditional(model, theory: str, Z_proj, Z_target, values):
    """
    Make predictions for a specific theory and projectile combination.
    
    Parameters:
    -----------
    model : FlexibleNN
        The loaded model
    theory : str
        Theory name (must be in _THEORIES)
    Z_proj : float
        Projectile atomic number. Safe limit (max trained): 9 (CDW-EIS, CTMC),
        1 (Semiempiric). Values above this limit trigger a warning.
    Z_target : float
        Target atomic number. Model was trained only for Z_target=1; other values
        are silently ignored (kept for API compatibility) and may mislead.
    values : np.ndarray
        Energy values (log10 of keV/u, since model was trained on log10 energy)
        
    Returns:
    --------
    predictions : np.ndarray
        Cross-section predictions (10^model_output)
    """
    enc = _ONE_HOT[theory]
    n = len(values)

    if Z_target != 1:
        log.warning(
            "Model trained only for Z_target=1 (hydrogen target). "
            "Z_target=%s is not a model input — predictions "
            "are for hydrogen regardless. Set Z_target=1 for clarity.",
            Z_target,
        )

    safe_zp = ZPROJ_SAFE[theory]
    zp_oob = Z_proj > safe_zp
    if zp_oob:
        log.warning(
            "Z_proj=%s exceeds safe limit (%s) for %s. "
            "Returning NaN.",
            Z_proj, safe_zp, theory,
        )

    # Warn if energy outside training range for this theory
    e_min = TRAIN_E_MIN[theory]
    e_max = TRAIN_E_MAX[theory]
    log10_min = np.log10(e_min)
    log10_max = np.log10(e_max)
    energy_oob = (values < log10_min) | (values > log10_max)
    if energy_oob.any():
        e_in = 10.0 ** values
        lo = e_in < e_min
        hi = e_in > e_max
        parts = []
        if lo.any():
            parts.append(f"({e_in[lo].min():.3g} keV/u) below")
        if hi.any():
            parts.append(f"({e_in[hi].max():.3g} keV/u) above")
        log.warning(
            "Energy %s training range [%g, %g] keV/u for %s. "
            "Returning NaN for out-of-range values.",
            " and ".join(parts), e_min, e_max, theory,
        )

    x = np.column_stack([
        np.full(n, Z_proj, dtype=np.float32),
        values.astype(np.float32),
        np.tile(enc, (n, 1)),
    ])
    predictions = 10.0 ** model.forward(x)

    if zp_oob:
        predictions[:] = np.nan
    elif energy_oob.any():
        predictions[energy_oob] = np.nan

    return predictions
