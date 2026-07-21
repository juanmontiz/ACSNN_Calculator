# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT

"""ACSNN Calculator — Atomic Cross Section Neural Network."""

from .model import FlexibleNN, predict_conditional, _THEORIES

THEORIES = _THEORIES
__all__ = ["FlexibleNN", "predict_conditional", "THEORIES"]
