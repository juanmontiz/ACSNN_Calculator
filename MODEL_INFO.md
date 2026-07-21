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
- **Parameters**: ~33,000  

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
| CTMC | 25 – 2,500 |
| Semiempiric_1985Rudd | 0.1 – 5,000 |

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

`predict_conditional()` validates all three constraints at inference time using Python's `logging` module. Warnings are emitted to stderr with the prefix `[ACSNN] WARNING:`:

- **Energy outside training range (below):**

  ```
  [ACSNN] WARNING: Energy (1 keV/u) below training range [10, 100000] keV/u for CDW-EIS.
  Extrapolation may be unreliable.
  ```

- **Energy outside training range (above):**

  ```
  [ACSNN] WARNING: Energy (2e+05 keV/u) above training range [10, 100000] keV/u for CDW-EIS.
  Extrapolation may be unreliable.
  ```

- **Energy outside both bounds:**

  ```
  [ACSNN] WARNING: Energy (1 keV/u) below and (2e+05 keV/u) above training range
  [10, 100000] keV/u for CDW-EIS. Extrapolation may be unreliable.
  ```

- **Z_target ≠ 1:**

  ```
  [ACSNN] WARNING: Model trained only for Z_target=1 (hydrogen target).
  Z_target=6 is not a model input — predictions are for hydrogen regardless.
  Set Z_target=1 for clarity.
  ```

- **Z_proj exceeds trained maximum:**

  ```
  [ACSNN] WARNING: Z_proj=10 exceeds max trained (9) for CDW-EIS.
  Extrapolation may be unreliable.
  ```

### Why Extrapolation Matters

Neural networks perform **interpolation** well (predicting inside the training range) but **extrapolation** poorly (predicting outside it). A Gaussian-activated network like this one will saturate to near-constant output for extreme values, giving physically meaningless results.

### Adjusting the Limits

Training constraints defined in `acsnn/config.py`:

```python
TRAIN_E_MIN = {"CDW-EIS": 10.0, "CTMC": 10.0, "Semiempiric_1985Rudd": 0.1}
TRAIN_E_MAX = {"CDW-EIS": 100000.0, "CTMC": 100000.0, "Semiempiric_1985Rudd": 5000.0}
ZPROJ_SAFE  = {"CDW-EIS": 9, "CTMC": 9, "Semiempiric_1985Rudd": 1}
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
