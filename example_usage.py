#!/usr/bin/env python3
# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT
# NOTE (Windows users): run this as `python example_usage.py` from cmd.exe or PowerShell.
# The shebang line above is ignored on Windows.
"""
Example usage of the ACSNN pure NumPy model.
No PyTorch required!
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from acsnn.model import FlexibleNN, predict_conditional


def main():
    print("="*60)
    print("ACSNN Pure NumPy Model - Example Usage")
    print("="*60)
    
    # Load the model
    print("\n1. Loading model...")
    model = FlexibleNN('models_folder/model_weights.npz')
    print(f"   Model loaded successfully!")
    print(f"   Input dimension: {model.input_dim}")
    print(f"   Number of layers: {len(model._layers)}")
    
    # Example 1: Single energy prediction
    print("\n2. Single energy prediction:")
    energy_kev = 100.0
    energy_log10 = np.log10(np.array([energy_kev]))
    
    cross_section = predict_conditional(
        model,
        theory="CDW-EIS",
        Z_proj=1,  # Hydrogen projectile
        Z_target=1,  # Model trained only for Z_target=1 (ignored by NN)
        values=energy_log10
    )
    
    print(f"   Theory: CDW-EIS")
    print(f"   Projectile: H (Z=1)")
    print(f"   Energy: {energy_kev} keV/u")
    print(f"   Cross section: {cross_section[0,0]:.6e} cm²")
    
    # Example 2: Energy scan
    print("\n3. Energy scan (multiple energies):")
    energies_kev = np.array([10, 50, 100, 500, 1000, 5000])
    energies_log10 = np.log10(energies_kev)
    
    cross_sections = predict_conditional(
        model,
        theory="CTMC",
        Z_proj=2,  # Helium
        Z_target=1,
        values=energies_log10
    )
    
    print(f"\n   Theory: CTMC, He projectile (Z=2)")
    print(f"   {'Energy [keV/u]':<20} {'Cross Section [cm²]':<20}")
    print("   " + "-"*40)
    for E, cs in zip(energies_kev, cross_sections):
        print(f"   {E:<20.1f} {cs[0]:<20.6e}")
    
    # Example 3: Compare theories
    print("\n4. Comparing theories at 500 keV/u:")
    energy_log10 = np.log10(np.array([500.0]))
    theories = ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"]
    
    print(f"\n   {'Theory':<30} {'Cross Section [cm²]':<20}")
    print("   " + "-"*50)
    for theory in theories:
        cs = predict_conditional(model, theory, Z_proj=1, Z_target=1, 
                                values=energy_log10)
        print(f"   {theory:<30} {cs[0,0]:<20.6e}")
    
    # Example 4: Batch processing
    print("\n5. Batch processing (100 energies):")
    import time
    
    energies_kev = np.logspace(1, 4, 100)
    energies_log10 = np.log10(energies_kev)
    
    start = time.time()
    cross_sections = predict_conditional(
        model,
        theory="CDW-EIS",
        Z_proj=1,
        Z_target=1,
        values=energies_log10
    )
    elapsed = time.time() - start
    
    print(f"   Processed {len(energies_kev)} points in {elapsed*1000:.2f} ms")
    print(f"   Rate: {len(energies_kev)/elapsed:.0f} predictions/second")
    
    print("\n" + "="*60)
    print("✓ Examples completed successfully!")
    print("="*60)
    print("\nKey features:")
    print("  ✓ 100% pure NumPy - no PyTorch needed")
    print("  ✓ Fast vectorized predictions")
    print("  ✓ Easy to use and deploy")


if __name__ == '__main__':
    main()
