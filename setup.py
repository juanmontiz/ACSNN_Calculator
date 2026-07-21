# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT

"""
Setup configuration for ACSNN Calculator
Atomic Cross Section Neural Network - Pure NumPy Implementation

To uninstall after installing via pip:
    pip uninstall acsnn-calculator
"""
from setuptools import setup, find_packages

setup(
    name="acsnn-calculator",
    version="1.0.0",
    description="Atomic Cross Section Neural Network Calculator - Pure NumPy Implementation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Juan Manuel Monti",
    author_email="juan.monti@gmail.com",
    url="https://github.com/juanmonti/acsnn-calculator",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "matplotlib>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'acsnn-cli=acsnn_cli:main',
            'acsnn-gui=acsnn_gui:main',
        ],
    },
    include_package_data=True,
    package_data={
        'acsnn': ['../models_folder/*.npz'],
        '': ['dbs/Experimental_data_db/*'],
    },
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="atomic cross-section neural-network physics ionization numpy",
)