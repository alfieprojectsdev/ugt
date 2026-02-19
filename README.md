# Investigating the Dynamics of a Chaotic Sprott Circuit (2026 Re-verification)

> **Original Thesis**: *Numerical Investigation of Chaos in a Modified Electronic System*  
> **Author**: Alfie Revilla Pelicano  
> **Institution**: National Institute of Physics, University of the Philippines Diliman (2004)

## Overview

This repository contains a modern re-investigation of the undergraduate physics thesis authored by Alfie Revilla Pelicano in 2004. The original research focused on a numerical investigation of chaos within a modified electronic system—specifically, a Sprott circuit with sinusoidally varying resistance.

The 2026 iteration leverages **JAX** for accelerated numerical integration and Lyapunov exponent calculation, replacing the original Fortran implementation to utilize modern hardware capabilities (GPU/TPU acceleration) and automatic differentiation.

## Key Research Objectives

1.  **Model Verification**: Re-implement the modified Sprott circuit equations:
    $$ \dddot{x} = -A\ddot{x} - \dot{x} + |x| - 1 $$
    Where $A = (R_c + R_{amp} \sin(\omega t))^{-1}$.
2.  **Stability Analysis**: Investigate how the sinusoidally varying resistance ($R_{amp} \sin(\omega t)$) influences the stability and chaotic nature of the system.
3.  **Quantification of Chaos**: Compute the Maximal Lyapunov Exponent (LLE) to identify chaotic regions in the parameter space.

## Technical Implementation

The project is modernized using Python 3.11+ and the JAX ecosystem:

-   **`sprott_solver.py`**: Solves the system of ODEs using a custom RK4 integrator. Supports both "Real-Time" (RT) animation and "Post-Processing" (PP) batch simulation.
-   **`lle_solver.py`**: Computes the Largest Lyapunov Exponent using Benettin's algorithm. It utilizes `jax.jacfwd` for automatic Jacobian computation, replacing manual linearization.
-   **`setup_env.sh`**: A helper script to set up the environment using `uv`.

## Installation & Usage

### Prerequisites
-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv) (automatically installed by setup script if missing)

### Setup
Initialize the environment:
```bash
./setup_env.sh
source .venv/bin/activate
```

### Running Simulations

**Trajectory Visualization:**
```bash
python sprott_solver.py
```
*Edit `MODE` in `sprott_solver.py` to switch between Real-Time animation and Batch processing.*

**Lyapunov Exponent Sweep:**
```bash
python lle_solver.py
```
*This will perform a parameter sweep over $\omega$ and plot the LLE spectrum.*

## Original Acknowledgements (2004)

This work reflects the academic environment of the National Institute of Physics at UP Diliman. The original thesis included administrative certifications of successful defense and personal acknowledgements to the thesis adviser and colleagues.

## License

MIT
