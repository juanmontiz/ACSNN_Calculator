# ACSNN Model Information

## Pure NumPy Implementation

This implementation uses **100% pure NumPy** - no PyTorch required for inference!

## Model Architecture

```
Input (5) → Linear(128) + Gaussian → Linear(128) + Gaussian → Linear(1)
```

### Details
- **Layers**: 3 (lin1, lin2, lin5)  
- **Hidden units**: 128 per layer  
- **Activation**: Gaussian (exp(-x²))  
- **Input dimension**: 5  
- **Output dimension**: 1  
- **Parameters**: 17,409  

### Input Features
```
[Z_projectile, log10(energy), theory_one_hot_bit_0, theory_one_hot_bit_1, theory_one_hot_bit_2]
```

Where:  
- `Z_projectile`: Atomic number of the projectile  
- `log10(energy)`: Log10 of energy in keV/u  
- `theory_one_hot`: One-hot encoding of theory (CDW-EIS, CTMC, or Semiempiric_1985Rudd)

### Output
- `log10(cross_section)`: Model outputs log10 of the cross section  
- Actual cross section is obtained by: `cross_section = 10^output`

## Usage

### Load Model

```python
from acsnn.model import FlexibleNN, predict_conditional
import numpy as np

# Load model
model = FlexibleNN('models_folder/model_weights.npz')
```

### Make Predictions

```python
# Energy values in log10 scale
energies = np.log10(np.array([100.0, 500.0, 1000.0]))

# Get cross-section predictions
predictions = predict_conditional(
    model,
    theory="CDW-EIS",
    Z_proj=1,  # Hydrogen projectile
    Z_target=1,  # Model trained only for Z_target=1; ignored by NN
    values=energies
)

print(predictions)  # Cross sections in cm²
```

### Direct Forward Pass

```python
# Manually construct input
Z_proj = 1.0
energy_log10 = np.log10(100.0)
theory_encoding = np.array([1.0, 0.0, 0.0])  # CDW-EIS

x = np.array([[Z_proj, energy_log10, *theory_encoding]], dtype=np.float32)

# Forward pass
log_cs = model.forward(x)
cross_section = 10.0 ** log_cs
```

## Supported Theories

The model supports three theories for cross-section predictions:

1. **CDW-EIS** - Continuum Distorted Wave - Eikonal Initial State
2. **CTMC** - Classical Trajectory Monte Carlo
3. **Semiempiric_1985Rudd** - Semiempirical model from Rudd (1985)

Each theory is encoded as a one-hot vector:
- CDW-EIS: [1, 0, 0]
- CTMC: [0, 1, 0]
- Semiempiric_1985Rudd: [0, 0, 1]

## Activation Function

The Gaussian activation function is defined as:

```python
def _gaussian(x):
    return np.exp(-x**2)
```

This is applied after each hidden layer (lin1 and lin2), but not after the output layer (lin5).

## Model Weights

The model weights are stored in `models_folder/model_weights.npz` with the following structure:

| Key | Shape | Description |
|-----|-------|-------------|
| `lin1_weight` | (128, 5) | First layer weights |
| `lin1_bias` | (128,) | First layer bias |
| `lin2_weight` | (128, 128) | Second layer weights |
| `lin2_bias` | (128,) | Second layer bias |
| `lin5_weight` | (1, 128) | Output layer weights |
| `lin5_bias` | (1,) | Output layer bias |

## Performance

- **Speed**: 6000+ predictions/second for batch processing
- **Accuracy**: Numerically verified against PyTorch implementation
- **Memory**: Low memory footprint (no PyTorch dependency)

## Testing

Run the test suite:

```bash
python -m pytest test_model.py -v
```

Expected: All 20 tests pass ✅

## Dependencies

### Required
```
numpy >= 1.19.0
```

### Optional (for CLI/GUI)
```
pandas >= 1.0.0
matplotlib >= 3.0.0
```

### NOT Required
```
❌ PyTorch (not needed for inference!)
```

## Example Results

For H + H collision at 100 keV/u using CDW-EIS theory:
- Cross section: ~1.19 × 10⁻¹⁶ cm²

## Technical Notes

- The model does not use `Z_target` as an input (only `Z_proj`)
- All predictions are deterministic (no dropout at inference)
- The model expects energy in log10 scale
- Output is also in log10 scale and needs to be converted

## Training Constraints

### Energy Range

Training energy ranges vary by theory:

| Theory | Energy range (keV/u) |
|--------|---------------------|
| CDW-EIS | 10 – 100,000 |
| CTMC | 15 – 2,500 |
| Semiempiric_1985Rudd | 1 – 10,000 |

### Target Atomic Number (Z_target)

The model was **trained exclusively for Z_target = 1** (hydrogen target). The `Z_target` parameter is not part of the NN input features — predictions are identical regardless of its value. Passing Z_target > 1 emits a warning.

### Projectile Atomic Number (Z_proj)

The maximum trained Z_proj varies by theory:

| Theory | Max Z_proj |
|--------|-----------|
| CDW-EIS | 9 |
| CTMC | 9 |
| Semiempiric_1985Rudd | 1 |

Passing Z_proj above this limit emits a warning. These limits are defined in `ZPROJ_SAFE` in `acsnn/config.py`.

## Behavior Outside Training Constraints

`predict_conditional()` checks two kinds of limits at inference time using Python's `logging` module
(logger `acsnn.model`, messages to stderr with the prefix `[ACSNN]`):

- **Hard limit — training domain** (`WARNING`, red in a terminal): the input lies outside the data the
  NN was trained on. The prediction is replaced by **NaN**.
- **Soft limit — physical validity domain of the theory** (`NOTICE`, yellow in a terminal): the input
  is inside the training range, so the NN reproduces the theory faithfully, but the theory itself is
  outside its regime of validity. The value **is returned**.

### Hard limit (NaN)

- **Energy outside training range (below / above / both):**

  ```
  [ACSNN] WARNING: Energy (1 keV/u) below training range [10, 100000] keV/u for CDW-EIS. Returning NaN for out-of-range values.
  [ACSNN] WARNING: Energy (2e+05 keV/u) above training range [10, 100000] keV/u for CDW-EIS. Returning NaN for out-of-range values.
  [ACSNN] WARNING: Energy (1 keV/u) below and (2e+05 keV/u) above training range [10, 100000] keV/u for CDW-EIS. Returning NaN for out-of-range values.
  ```

- **Z_proj exceeds trained maximum:**

  ```
  [ACSNN] WARNING: Z_proj=10 exceeds safe limit (9) for CDW-EIS. Returning NaN.
  ```

- **Z_target ≠ 1** (warning only; predictions are for hydrogen):

  ```
  [ACSNN] WARNING: Model trained only for Z_target=1 (hydrogen target). Z_target=6 is not a model input — predictions are for hydrogen regardless. Set Z_target=1 for clarity.
  ```

### Soft limit (value returned)

| Theory | Validity domain | Origin |
|--------|-----------------|--------|
| CDW-EIS | E > 24.98 · Z_proj² keV/u | perturbative regime, Sommerfeld parameter η = Z_proj/v < 1 |
| CTMC | ≈ 75 – 500 keV/u | intermediate energies (approximate, the boundaries are blurred) |
| Semiempiric_1985Rudd | whole training range | — |

```
[ACSNN] NOTICE: Energy (100 keV/u) below validity domain of CDW-EIS (perturbative regime eta = Z_proj/v < 1, i.e. E > 899.3 keV/u for Z_proj=6). Values are returned, but the underlying theory may be inaccurate there.
[ACSNN] NOTICE: Energy (1000 keV/u) above validity domain of CTMC (approx. [75, 500] keV/u). Values are returned, but the underlying theory may be inaccurate there.
```

Per-point labels are available programmatically:

```python
from acsnn import validity_flags, validity_range
validity_flags("CTMC", 2, [10, 50, 100, 1000])
# -> ['outside_training', 'outside_validity', 'valid', 'outside_validity']
validity_range("CDW-EIS", 6)   # -> (899.28, inf)
```

The CSV files written by the CLI (`--output`), the GUI and the web app contain a **Validity** column with
these labels (`valid`, `outside_validity`, `outside_training`). To silence the soft notices only:
`logging.getLogger("acsnn.model").setLevel(logging.WARNING)`.

### Why Extrapolation Matters

Neural networks perform **interpolation** well (predicting inside the training range) but **extrapolation** poorly (predicting outside it). A Gaussian-activated network like this one will saturate to near-constant output for extreme values, giving physically meaningless results.

### Adjusting the Limits

Training constraints defined in `acsnn/config.py`:

```python
TRAIN_E_MIN = {"CDW-EIS": 10.0,     "CTMC": 15.0,   "Semiempiric_1985Rudd": 1.0}
TRAIN_E_MAX = {"CDW-EIS": 100000.0, "CTMC": 2500.0, "Semiempiric_1985Rudd": 10000.0}
ZPROJ_SAFE  = {"CDW-EIS": 9, "CTMC": 9, "Semiempiric_1985Rudd": 1}
```

Theory validity domains (soft limits) are defined in the same file:

```python
E_PER_AU_VELOCITY2 = 24.98          # E [keV/u] for v = 1 a.u.; CDW-EIS valid for E > 24.98 * Z_proj**2
CTMC_VALID_E = (75.0, 500.0)        # CTMC validity window in keV/u
```

If you retrain the model on a different range, update these to match.

### Disabling Warnings

Use the Python `logging` API to suppress or increase severity:

```python
import logging

# Silence all ACSNN warnings
logging.getLogger("acsnn.model").setLevel(logging.ERROR)
```

**Note**: Only suppress warnings if you fully understand the extrapolation risks.
