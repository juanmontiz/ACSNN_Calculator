# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT

"""
Command-line interface for Atomic Cross Section Calculator
Allows running calculations without the GUI
"""

import argparse
import numpy as np
import os
import pandas as pd
from acsnn.config import MODEL_PATH, EXPERIMENTAL_DATA_PATH, DEFAULT_X_MIN, DEFAULT_X_MAX, DEFAULT_N_POINTS
from acsnn import FlexibleNN, predict_conditional


def load_model():
    return FlexibleNN(os.path.join(MODEL_PATH, 'model_weights.npz'))


def run_calculation(theory, Zp, Zt, x_min=DEFAULT_X_MIN, x_max=DEFAULT_X_MAX, npoints=DEFAULT_N_POINTS, output_file=None):
    """Run a calculation over a range and optionally save to file"""
    model = load_model()

    x = np.logspace(np.log10(x_min), np.log10(x_max), npoints)

    if theory in ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"]:
        y = predict_conditional(model, theory, Zp, Zt, np.log10(x))
    else:
        raise ValueError(f"Unknown theory: {theory}")

    results_df = pd.DataFrame({
        'E[keV/u]': x,
        f'TCS_{theory}': y.flatten() if y.ndim > 1 else y
    })

    if output_file:
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Results saved to {output_file}")

    return results_df


def run_single_calculation(theory, Zp, Zt, energy_value):
    """Run a calculation for a single energy value"""
    model = load_model()

    if theory in ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"]:
        y = predict_conditional(model, theory, Zp, Zt, np.array([np.log10(energy_value)]))
    else:
        raise ValueError(f"Unknown theory: {theory}")

    return float(y.flatten()[0]) if y.ndim > 1 else float(y[0])


def main():
    parser = argparse.ArgumentParser(description='Atomic Cross Section Calculator CLI')
    parser.add_argument('theory', choices=["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"],
                        help='Theory to use for calculation')
    parser.add_argument('Zp', type=int, help='Projectile charge')
    parser.add_argument('Zt', type=int, help='Target atomic number')
    parser.add_argument('energy_value', type=float, nargs='?',
                        help='Single energy value for calculation (keV/u). If provided, calculates only this value instead of a range.')
    parser.add_argument('--x-min', type=float, default=DEFAULT_X_MIN,
                        help='Minimum x value (default: 10)')
    parser.add_argument('--x-max', type=float, default=DEFAULT_X_MAX,
                        help='Maximum x value (default: 10000)')
    parser.add_argument('--npoints', type=int, default=DEFAULT_N_POINTS,
                        help='Number of points (default: 300)')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--value-only', action='store_true',
                        help='Output only the numerical value without descriptive text')

    args = parser.parse_args()

    if args.Zp <= 0 or args.Zt <= 0:
        print("Error: Zp and Zt must be positive integers")
        return 1

    if args.energy_value is not None:
        try:
            result = run_single_calculation(
                theory=args.theory,
                Zp=args.Zp,
                Zt=args.Zt,
                energy_value=args.energy_value
            )
            if args.value_only:
                print(f"{result}")
            else:
                print(f"TCS for {args.theory} at E={args.energy_value} keV/u, Zp={args.Zp}, Zt=1 (hydrogen): {result}")
            return 0
        except Exception as e:
            print(f"Error during single value calculation: {str(e)}")
            return 1
    else:
        if args.x_min <= 0 or args.x_max <= 0 or args.x_min >= args.x_max:
            print("Error: x_min and x_max must be positive with x_min < x_max")
            return 1
        if args.npoints <= 0:
            print("Error: npoints must be a positive integer")
            return 1
        try:
            results = run_calculation(
                theory=args.theory,
                Zp=args.Zp,
                Zt=args.Zt,
                x_min=args.x_min,
                x_max=args.x_max,
                npoints=args.npoints,
                output_file=args.output
            )
            if not args.output:
                if not args.value_only:
                    print(f"Calculation completed for {args.theory}, Zp={args.Zp}, Zt=1 (hydrogen)")
                    print("\nFirst 10 results:")
                print(results.head(10) if not args.value_only else results.to_string(index=False))
            return 0
        except Exception as e:
            print(f"Error during calculation: {str(e)}")
            return 1


if __name__ == "__main__":
    exit(main())
