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
    Where $A = (R_c + R_{amp} \sin(\omega t))^{-1}$ and $R_c = 1/0.6$ (see the note on $R_c$ below).
2.  **Stability Analysis**: Investigate how the sinusoidally varying resistance ($R_{amp} \sin(\omega t)$) influences the stability and chaotic nature of the system.
3.  **Quantification of Chaos**: Compute the Maximal Lyapunov Exponent (LLE) to identify chaotic regions in the parameter space.

## Technical Implementation

Python 3.12 and JAX, CPU or GPU.

- `sprott_solver.py`: RK4 integrator for the ODE, `jax.lax.scan` inside `jit`, `vmap` over a parameter batch.
- `lle_solver.py`: largest Lyapunov exponent by Benettin's method. The tangent vector is evolved with the Jacobian from `jax.jacfwd`, renormalised every 10 steps.
- `lyapunov_spectrum.py`: all three exponents by Gram-Schmidt on a 3x3 tangent frame, with per-parameter block standard errors, the Kaplan-Yorke dimension, and the trace check sum(lambda) = -<A>. Use this rather than `lle_solver.py` for anything that gets published.
- `wolf_fet.py`: Wolf's fixed-evolution-time estimator from a scalar series, reimplemented from the 2004 thesis's Appendix B with its parameters, for comparing against the 2004 numbers.
- `bifurcation.py`: Poincare-section bifurcation diagrams (see below).
- `run_all.py`: headless driver that runs every stage and writes PNGs, NPZ data and `timings.json` to `results/`.
- `verify.py`: the checks behind the numbers (Wolf vs Benettin, dt convergence, spectrum, error bars, and which R_c the 2004 code used). Writes to `results/verify/`.

### A note on R_c

The thesis text gives R_c = 1.67 ohm. The 2004 code used A = 0.6 exactly, i.e. R_c = 1/0.6 = 1.6667:
rerunning its protocol with 1/0.6 reproduces the seven plateau values in its Fig 5.5 to 0.0035 bits
rms, while 1.67 misses by 0.043 bits because 1.67 sits inside a period-5 window of the unforced
system. All scripts default to 1/0.6; `run_all.py --rc 1.67` gives the rounded value.

## Installation

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python jax jaxlib numpy matplotlib scipy
```

`setup_env.sh` does the same and also installs the dev extras.

### Windows note (Smart App Control)

On a Windows 11 machine with Smart App Control turned on, `import jax` fails with
`DLL load failed ... An Application Control policy has blocked this file` (event ID 3077 in
`Microsoft-Windows-CodeIntegrity/Operational`). The jaxlib and matplotlib wheels ship unsigned
native extensions that the policy refuses. Smart App Control has no per-application exception list;
the only alternatives are to turn it off (irreversible without reinstalling Windows) or to run in
WSL2. The test run recorded below used WSL2 Ubuntu with the venv on the Linux filesystem:

```bash
wsl -d Ubuntu -- bash -lc 'uv venv ~/.venvs/sprott --python 3.12 && uv pip install --python ~/.venvs/sprott/bin/python jax jaxlib numpy matplotlib scipy'
wsl -d Ubuntu -- bash -c 'cd /mnt/c/path/to/repo && ~/.venvs/sprott/bin/python run_all.py'
```

## Running

```bash
python run_all.py            # everything, results/
python run_all.py --quick    # ~4x smaller sweeps, smoke test
python run_all.py --stages lle_omega,bif_omega
python run_all.py --x64      # float64
python verify.py             # numerical checks, ~12 min
```

Stages:

| stage | what | output |
|---|---|---|
| `trajectories` | phase portraits at six omegas | `01_trajectories.png` |
| `lle_omega` | LLE vs omega on [0, 20] and a dense pass on [0, 3], R = 0.15 and 0.05 | `02_lle_vs_omega.png` |
| `bif_rc` | bifurcation vs R_c, unforced (R = 0) | `03_bif_rc_unforced.png` |
| `bif_omega` | bifurcation vs omega, peak map and stroboscopic | `04_bif_omega.png` |
| `bif_ramp` | bifurcation vs R at omega = 1 and 2 | `05_bif_ramp.png` |
| `lle_map` | LLE over the (R, omega) plane | `06_lle_map.png` |
| `thesis_grid` | the 2004 protocol verbatim: 7 R values, omega on [0, 20] step 2^-8, 2^15 steps | `07_thesis_fig5[145].png` |

The original entry points `python sprott_solver.py` and `python lle_solver.py` still work (set
`MPLBACKEND=Agg` on a headless box).

## Bifurcation diagrams

A bifurcation diagram plots a Poincare section of the attractor against a swept parameter. The
system has two regimes that call for different sections, and `bifurcation.py` implements both.

**Peak map** (`peak_map_batch`). Section: local maxima of x, i.e. crossings of x' = 0 with x'' < 0.
This is the section Sprott uses for jerk systems and it works whether or not the forcing is on, so it
is the one used for the sweep in R_c (unforced) and in R. A period-n orbit shows as n points per
column, chaos as a band.

**Stroboscopic map** (`stroboscopic_batch`). Section: the state sampled once per forcing period
T = 2 pi / omega. This is the standard section for periodically forced oscillators and is the one
that makes forcing-locked period-doubling clean. It needs omega > 0. To keep the sample stride a
static integer under `vmap`, the time step is set per parameter to dt = T / 512, so every row of
the batch takes the same number of steps per period while dt varies (0.049 at omega = 0.25, 0.0045
at omega = 2.75).

Both kernels return a fixed-length array plus a boolean mask (JAX needs static shapes) and only the
x component is kept, so 1200 parameters x 80 000 steps fits in a few hundred MB. Rendering uses a
per-column-normalised 2-D histogram with gamma 0.35 rather than a scatter: with millions of points a
scatter is a uniform grey block, and per-column normalisation keeps a periodic orbit (few bins, high
count) and a chaotic band (many bins, low count) both visible.

Results from the ThinkPad test run are in `results/` and summarised in `docs/thinkpad-test-run.md`.

## Building the paper

`main.tex` compiles with [Tectonic](https://tectonic-typesetting.github.io/) (XeTeX, single binary,
fetches packages on demand; about 60 MB all told) or with LuaLaTeX from a full TeX Live. The
preamble is engine-conditional: under LuaTeX it keeps `bidi=basic` and `onchar=ids fonts`, under
XeTeX it drops them (English-only document, no effect on output). Noto Sans must be installed as a
system font.

Install in WSL without root:

```bash
curl -sSL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz | tar xz -C ~/.local/bin tectonic
apt-get download fonts-noto-core && dpkg -x fonts-noto-core_*.deb /tmp/noto && mkdir -p ~/.local/share/fonts && cp /tmp/noto/usr/share/fonts/truetype/noto/NotoSans-{Regular,Bold,Italic,BoldItalic}.ttf ~/.local/share/fonts/ && fc-cache -f
```

Build:

```bash
tectonic --outdir build main.tex
```

## Original Acknowledgements (2004)

This work reflects the academic environment of the National Institute of Physics at UP Diliman. The original thesis included administrative certifications of successful defense and personal acknowledgements to the thesis adviser and colleagues.

## License

MIT
