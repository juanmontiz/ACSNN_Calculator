# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT

"""ACSNN Calculator — Atomic Collision Surrogate Neural Network."""

from .model import (FlexibleNN, predict_conditional, validity_range, validity_flags,
                    _THEORIES, VALID, OUTSIDE_VALIDITY, OUTSIDE_TRAINING)

THEORIES = _THEORIES
__all__ = ["FlexibleNN", "predict_conditional", "validity_range", "validity_flags", "THEORIES",
           "VALID", "OUTSIDE_VALIDITY", "OUTSIDE_TRAINING"]
