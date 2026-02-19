#!/bin/bash

# ==============================================================================
# Setup Script for Sprott Chaos 2026 (using uv)
# ==============================================================================

# 1. Install uv (if not present)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    # Ensure uv is in path for this session
    source $HOME/.cargo/env
else
    echo "uv is already installed."
fi

# 2. Create Virtual Environment
echo "Creating virtual environment..."
uv venv .venv

# 3. Activate Virtual Environment
# Note: In a script, this only affects the script's scope. 
# Instructions are printed for the user at the end.
source .venv/bin/activate

# 4. Install Dependencies
echo "Installing dependencies from pyproject.toml..."
# Compile requirements to a lock file for reproducibility
uv pip compile pyproject.toml -o requirements.txt

# Sync the environment with the requirements
uv pip sync requirements.txt

# Install dev dependencies
echo "Installing dev dependencies..."
uv pip install -e .[dev]

echo "=============================================================================="
echo "Setup Complete!"
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo "To run the solver:"
echo "  python sprott_solver.py"
echo "=============================================================================="