# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT

"""Pure NumPy neural network for atomic ionization cross-section prediction."""

import logging
import sys
import numpy as np
from acsnn.config import (TRAIN_E_MIN, TRAIN_E_MAX, ZPROJ_SAFE,
                          E_PER_AU_VELOCITY2, CTMC_VALID_E)

# Soft notice level (between INFO and WARNING): theory outside its physical
# validity domain, value still returned. Hard problems (NaN) use WARNING.
NOTICE = 25
logging.addLevelName(NOTICE, "NOTICE")

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
            _MAP = {NOTICE: _YELLOW, logging.WARNING: _RED, logging.ERROR: _RED}
            def format(self, r):
                c = self._MAP.get(r.levelno)
                return (c + _BOLD + "[ACSNN] " + _RESET + c +
                        "%(levelname)s: %(message)s" % {"levelname": r.levelname, "message": r.getMessage()} + _RESET) if c else logging.Formatter.format(self, r)
        _handler.setFormatter(_Fmt())
    else:
        _handler.setFormatter(logging.Formatter("[ACSNN] %(levelname)s: %(message)s"))
    log.addHandler(_handler)
    log.setLevel(NOTICE)

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


VALID, OUTSIDE_VALIDITY, OUTSIDE_TRAINING = "valid", "outside_validity", "outside_training"


def validity_range(theory: str, Z_proj):
    """
    Physical validity domain (E_min, E_max) in keV/u of a theory for a given
    projectile charge (soft limit, see acsnn/config.py):
    CDW-EIS: eta = Z_proj/v < 1, i.e. E > 24.98 * Z_proj**2 keV/u;
    CTMC: 75-500 keV/u; Semiempiric_1985Rudd: unrestricted.
    """
    if theory == "CDW-EIS":
        return (E_PER_AU_VELOCITY2 * float(Z_proj) ** 2, np.inf)
    if theory == "CTMC":
        return CTMC_VALID_E
    if theory == "Semiempiric_1985Rudd":
        return (0.0, np.inf)
    raise ValueError(f"Unknown theory: {theory}")


def validity_flags(theory: str, Z_proj, energies):
    """
    Per-energy validity label for energies in keV/u (not log10):
    "outside_training" (hard: prediction is NaN), "outside_validity" (soft:
    prediction returned but the theory is outside its physical validity
    domain) or "valid".
    """
    e = np.atleast_1d(np.asarray(energies, dtype=float))
    v_min, v_max = validity_range(theory, Z_proj)
    flags = np.full(e.shape, VALID, dtype=object)
    flags[(e <= v_min) | (e > v_max)] = OUTSIDE_VALIDITY
    if Z_proj > ZPROJ_SAFE[theory]:
        flags[:] = OUTSIDE_TRAINING
    else:
        flags[(e < TRAIN_E_MIN[theory]) | (e > TRAIN_E_MAX[theory])] = OUTSIDE_TRAINING
    return flags


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
        Cross-section predictions (10^model_output).
        Hard limit: outside the training domain (energy or Z_proj) a WARNING is
        logged and NaN is returned. Soft limit: inside the training range but
        outside the physical validity domain of the theory (see validity_range)
        a NOTICE is logged and the value is returned. Use validity_flags() for
        per-point labels.
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
            parts.append(f"({e_in[lo].min():.4g} keV/u) below")
        if hi.any():
            parts.append(f"({e_in[hi].max():.4g} keV/u) above")
        log.warning(
            "Energy %s training range [%g, %g] keV/u for %s. "
            "Returning NaN for out-of-range values.",
            " and ".join(parts), e_min, e_max, theory,
        )

    # Soft notice: inside the training range but outside the theory's validity domain
    if not zp_oob:
        v_min, v_max = validity_range(theory, Z_proj)
        log10_vmin = np.log10(v_min) if v_min > 0 else -np.inf
        log10_vmax = np.log10(v_max)
        soft_lo = ~energy_oob & (values <= log10_vmin)
        soft_hi = ~energy_oob & (values > log10_vmax)
        if soft_lo.any() or soft_hi.any():
            e_in = 10.0 ** values
            parts = []
            if soft_lo.any():
                parts.append(f"({e_in[soft_lo].min():.4g} keV/u) below")
            if soft_hi.any():
                parts.append(f"({e_in[soft_hi].max():.4g} keV/u) above")
            if theory == "CDW-EIS":
                domain = (f"perturbative regime eta = Z_proj/v < 1, "
                          f"i.e. E > {v_min:.4g} keV/u for Z_proj={Z_proj}")
            else:
                domain = f"approx. [{v_min:g}, {v_max:g}] keV/u"
            log.log(
                NOTICE,
                "Energy %s validity domain of %s (%s). Values are returned, "
                "but the underlying theory may be inaccurate there.",
                " and ".join(parts), theory, domain,
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
