# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT

"""
Tests for the prediction pipeline (model.py) and CLI calculation functions (cli.py).
Run with: python -m pytest test_model.py -v
"""

import numpy as np
import pytest

from acsnn_cli import load_model, run_calculation, run_single_calculation
from acsnn import FlexibleNN, predict_conditional

THEORIES = ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_model():
    return load_model()


# ---------------------------------------------------------------------------
# FlexibleNN architecture
# ---------------------------------------------------------------------------

def test_flexiblenn_output_shape(loaded_model):
    x = np.random.randn(10, 5).astype(np.float32)
    out = loaded_model.forward(x)
    assert out.shape == (10, 1)


def test_flexiblenn_eval_mode_is_deterministic(loaded_model):
    x = np.random.randn(5, 5).astype(np.float32)
    out1 = loaded_model.forward(x)
    out2 = loaded_model.forward(x)
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# predict_conditional
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theory", THEORIES)
def test_predict_conditional_output_shape(loaded_model, theory):
    values = np.log10(np.array([10.0, 100.0, 1000.0]))
    result = predict_conditional(loaded_model, theory, Z_proj=1, Z_target=1, values=values)
    assert result.shape == (3, 1)


@pytest.mark.parametrize("theory,emin,emax", [
    ("CDW-EIS", 10, 100000),
    ("CTMC", 15, 2500),
    ("Semiempiric_1985Rudd", 0.1, 5000),
])
def test_predict_conditional_positive_output(loaded_model, theory, emin, emax):
    """Cross sections must be positive (output is 10^predicted)."""
    values = np.log10(np.linspace(emin, emax, 10))
    result = predict_conditional(loaded_model, theory, Z_proj=1, Z_target=1, values=values)
    assert np.all(result > 0)


def test_predict_conditional_unknown_theory(loaded_model):
    with pytest.raises(Exception):
        predict_conditional(loaded_model, "NonexistentTheory", 1, 1, np.array([2.0]))


# ---------------------------------------------------------------------------
# predict_conditional: out-of-range returns NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theory", ["CDW-EIS", "CTMC"])
def test_predict_conditional_zproj_oob_returns_nan(loaded_model, theory):
    """Z_proj above safe limit must return NaN for all values."""
    values = np.log10(np.array([100.0, 500.0]))
    result = predict_conditional(loaded_model, theory, Z_proj=99, Z_target=1, values=values)
    assert result.shape == (2, 1)
    assert np.all(np.isnan(result))


def test_predict_conditional_zproj_oob_returns_nan_semiempiric(loaded_model):
    """Semiempiric Z_proj=2 exceeds limit of 1, must return NaN."""
    values = np.log10(np.array([100.0]))
    result = predict_conditional(loaded_model, "Semiempiric_1985Rudd", Z_proj=2, Z_target=1, values=values)
    assert np.isnan(result.flatten()[0])


@pytest.mark.parametrize("theory", ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"])
def test_predict_conditional_energy_below_range_returns_nan(loaded_model, theory):
    """Energy below training min must return NaN."""
    values = np.log10(np.array([1e-6]))  # far below any training range
    result = predict_conditional(loaded_model, theory, Z_proj=1, Z_target=1, values=values)
    assert np.isnan(result.flatten()[0])


@pytest.mark.parametrize("theory", ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"])
def test_predict_conditional_energy_above_range_returns_nan(loaded_model, theory):
    """Energy above training max must return NaN."""
    values = np.log10(np.array([1e9]))  # far above any training range
    result = predict_conditional(loaded_model, theory, Z_proj=1, Z_target=1, values=values)
    assert np.isnan(result.flatten()[0])


def test_predict_conditional_mixed_energy_range(loaded_model):
    """Mixed in-range and out-of-range energies: only out-of-range entries are NaN."""
    e_in = np.log10(np.array([100.0, 200.0]))       # both in range for CDW-EIS
    result = predict_conditional(loaded_model, "CDW-EIS", Z_proj=1, Z_target=1, values=e_in)
    assert result.shape == (2, 1)
    assert not np.any(np.isnan(result))

    e_mixed = np.log10(np.array([100.0, 1e-6, 500.0, 1e9]))
    result = predict_conditional(loaded_model, "CDW-EIS", Z_proj=1, Z_target=1, values=e_mixed)
    assert result.shape == (4, 1)
    assert not np.isnan(result[0, 0])
    assert np.isnan(result[1, 0])
    assert not np.isnan(result[2, 0])
    assert np.isnan(result[3, 0])


def test_predict_conditional_reproducible(loaded_model):
    """Same inputs must always yield the same output."""
    values = np.log10(np.array([100.0]))
    r1 = predict_conditional(loaded_model, "CDW-EIS", 1, 1, values)
    r2 = predict_conditional(loaded_model, "CDW-EIS", 1, 1, values)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Regression: known input → known output
# Baseline captured from a passing run; update if the model weights change.
# ---------------------------------------------------------------------------

def test_predict_conditional_regression(loaded_model):
    result = predict_conditional(loaded_model, "CDW-EIS", 1, 1, np.array([np.log10(100.0)]))
    expected = 1.192564e-16  # Updated for new model (3-layer Gaussian)
    assert abs(float(result.flatten()[0]) - expected) / expected < 1e-4


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def test_load_model_returns_flexiblenn():
    model = load_model()
    assert isinstance(model, FlexibleNN)


@pytest.mark.parametrize("theory", THEORIES)
def test_run_calculation_dataframe_shape(theory):
    df = run_calculation(theory, Zp=1, Zt=1, npoints=10)
    assert len(df) == 10
    assert "E[keV/u]" in df.columns
    assert f"TCS_{theory}" in df.columns


def test_run_calculation_energy_range():
    df = run_calculation("CDW-EIS", Zp=1, Zt=1, x_min=50, x_max=500, npoints=5)
    assert df["E[keV/u]"].min() == pytest.approx(50, rel=1e-6)
    assert df["E[keV/u]"].max() == pytest.approx(500, rel=1e-6)


def test_run_calculation_invalid_theory():
    with pytest.raises(ValueError, match="Unknown theory"):
        run_calculation("BadTheory", Zp=1, Zt=1)


def test_run_single_calculation_returns_float():
    result = run_single_calculation("CDW-EIS", Zp=1, Zt=1, energy_value=100.0)
    assert isinstance(result, float)
    assert result > 0


def test_run_single_calculation_matches_range():
    """Single-point result must match the equivalent point from a range calculation."""
    energy = 100.0
    single = run_single_calculation("CDW-EIS", Zp=1, Zt=1, energy_value=energy)
    df = run_calculation("CDW-EIS", Zp=1, Zt=1, x_min=energy, x_max=energy * 1.001, npoints=2)
    range_val = float(df["TCS_CDW-EIS"].iloc[0])
    assert abs(single - range_val) / single < 1e-4


def test_run_calculation_csv_export(tmp_path):
    out = tmp_path / "results.csv"
    run_calculation("CTMC", Zp=2, Zt=6, npoints=5, output_file=str(out))
    assert out.exists()
    import pandas as pd
    df = pd.read_csv(out)
    assert len(df) == 5
