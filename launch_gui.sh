#!/bin/bash

# Launcher for ACSNN Calculator GUI
# Pure NumPy implementation - works with Python 3.8+
# - Verifies Python availability and version
# - Optionally creates a virtual environment

PYTHON_CMD=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_CMD="$candidate"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PY_VERSION=$("$PYTHON_CMD" -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
VERSION_STATUS=$("$PYTHON_CMD" -c "import sys; print('ok' if sys.version_info >= (3, 8) else 'old')")

USE_VENV=false
if [ "$VERSION_STATUS" = "old" ]; then
    echo "ERROR: Detected Python $PY_VERSION (below 3.8)."
    echo "Please install Python 3.8 or higher."
    exit 1
else
    echo "Detected Python $PY_VERSION ✓"
    read -r -p "Create/use a virtual environment? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy] ]]; then
        USE_VENV=true
    fi
fi

if [ "$USE_VENV" = true ]; then
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment with $PYTHON_CMD..."
        "$PYTHON_CMD" -m venv venv
    else
        echo "Using existing virtual environment in venv"
    fi
    # shellcheck source=/dev/null
    source venv/bin/activate
    RUN_PYTHON="python"
else
    RUN_PYTHON="$PYTHON_CMD"
fi

echo "Installing dependencies..."
$RUN_PYTHON -m pip install --no-cache-dir -r requirements.txt

echo "Starting Atomic Cross Section Calculator..."
$RUN_PYTHON acsnn_gui.py
