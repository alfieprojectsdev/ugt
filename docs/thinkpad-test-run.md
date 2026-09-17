# ThinkPad T14 test run, 2026-09-17

Full run of the pipeline on the laptop before the Dell R740 run. Everything here was produced by
`python run_all.py` and `python verify.py` on branch `claude/thesis-thinkpad-bifurcation-5c739e`;
figures are in `results/` and `results/verify/`, numbers in `results/timings.json` and
`results/verify/verify.json`.

## Environment

- ThinkPad T14, Intel i5-10310U, 4 cores / 8 threads, 16 GB, no discrete GPU.
- Windows 11 Pro 26200 with Smart App Control on. `import jax` and `import matplotlib` fail there
  (`DLL load failed ... An Application Control policy has blocked this file`, CodeIntegrity event
  3077). numpy and scipy import fine. Smart App Control has no exception list, so the run used WSL2.
- WSL2 Ubuntu 26.04, 8 vCPU, 7 GB visible, Python 3.12.14, jax 0.11.1 CPU, float32 unless stated.
  Venv on the Linux filesystem (`~/.venvs/sprott`), repo read from `/mnt/c`.
- PyPI download rate on this connection was about 250 KB/s; the two installs took about 45 minutes
  in total. Budget for that on the R740 if it is on the same network, or pre-download wheels.

## Bugs found in the original scripts

Neither original entry point ran as committed.

1. `sprott_solver.py` and `lle_solver.py`: the step counts (`time_steps`, `total_steps`,
   `norm_interval`) are used as `jax.lax.scan(length=...)` inside `@jit`, where they are tracers.
   Fixed with `static_argnames`.
2. `lle_solver.py` `__main__`: `batch_lle(...)` returns a tuple; `.block_until_ready()` was called
   on the tuple.
3. `lle_solver.calculate_lle` reported a finite, positive exponent (about 0.1 to 0.16 nats) for
   orbits that escape to infinity, because the tangent vector tracks the unstable linear mode of the
   x < 0 half-space before the state overflows. Added an escape guard that returns NaN when the
   final state is non-finite or `|state| > 1e3`. Without it the (R, omega) map shows the escape
   region as the most chaotic part of the plane.
4. `setup_env.sh`: the `curl` line had Markdown link syntax inside it and the uv installer no
   longer puts anything in `~/.cargo/env`.

## Timings (float32, CPU)

| stage | work | wall |
|---|---|---|
| trajectories | 6 x 32 768 RK4 steps | 1.4 s (0.2 s compile), 6.7 M steps/s |
| lle_omega | 2 800 LLEs x 60 000 augmented steps | 20 s |
| bif_rc | 1 200 params x 80 000 steps, peak map | 3.3 s |
| bif_omega | 1 200 params: peak map 80 000 steps + stroboscopic 512 x 500 steps | 21 s |
| bif_ramp | 2 x 1 200 params x 80 000 steps | 11 s |
| lle_map | 4 800 LLEs x 30 000 augmented steps | 18 s |
| thesis_grid | 35 840 LLEs x 32 768 augmented steps (the 2004 protocol) | 134 s, 8.8 M aug. steps/s |
| total | | 208 s |

"Augmented step" is one RK4 step of the 6-D state + tangent system with a `jacfwd` Jacobian. Peak
throughput on this CPU is about 9 M augmented steps/s or 7 M plain steps/s. float64 (`--x64`) ran the
spot checks at the same speed as float32 and agreed with it to about 0.003 bits; float32 is adequate
for sweeps, float64 is cheap insurance for anything that gets published.

The 2004 thesis reports 2^15 steps per parameter and 5 120 omega values per R. The whole seven-R
grid now runs in 2.2 minutes on the laptop. For the R740 the interesting question is not this grid
but what to spend the headroom on: longer integrations (the LLE is still drifting by about 0.005
nats over the last 10 % of a 60 000-step run), the (R_c, R, omega) cube, or basin-of-attraction
maps near the escape boundary.

## Results against the thesis

Thesis values are in bits/s from Wolf's algorithm on the x time series; ours are Benettin/Jacobian
exponents converted with 1/ln 2. All figures in `results/` are from the corrected R_c = 1/0.6 run;
the first-pass R_c = 1.67 figures are kept in `results_rc1.67/` (untracked).

### The R_c correction, and a retracted claim

The first pass used R_c = 1.67 as printed in the thesis text and in this repo's scripts, and found
the plateau for omega > 2.75 collapsing to zero for R <= 0.0125 while the thesis reports about
0.055 bits/s for every R. I first attributed that to a positive floor in Wolf's algorithm. That was
wrong, and `verify.py` shows why on two counts:

1. Wolf's FET with the thesis's own parameters (tau = 15, dim = 4, evolve = 9, dismin 2 %, dismax
   15 %, thmax 30 deg, 2^15 samples) returns -0.0005 and +0.0002 bits on the period-5 orbit at
   R_c = 1.67, with and without the transient. It has no floor. On chaotic series it runs 0.01 to
   0.02 bits above Benettin, so it is biased high but not by 0.05. (`v1_wolf_vs_benettin.png`)
2. Rerunning the plateau with R_c = 1/0.6 = 1.6667, which is Sprott's A = 0.6 exactly, reproduces
   all seven thesis values with an rms deviation of 0.0035 bits, inside their quoted std. With
   R_c = 1.67 the rms deviation is 0.043 bits. (`verify.json`, check `rc`)

So the 2004 Fortran used A_0 = 0.6 and the text rounded it to 1.67 ohm. The rounded value happens
to fall inside a period-5 window of the unforced system (about 1.67 to 1.72 in
`03_bif_rc_unforced.png`), which is why the first pass diverged from the thesis at small R. The
repo now defaults to R_c = 1/0.6 everywhere; `run_all.py --rc 1.67` reproduces the first pass.

### Plateau, omega > 2.75 (thesis Fig 5.5)

Mean +/- std over 3 420 omega values per R, 2^15 steps, R_c = 1/0.6:

| R | thesis (bits/s) | this run (bits) | full spectrum at omega = 10, 200k steps: lambda_1 +/- SE, lambda_3 (nats), D_KY |
|---|---|---|---|
| 0.15 | 0.0666 +/- 0.0072 | 0.0707 +/- 0.0070 | 0.0517 +/- 0.0018, -0.654, 2.079 |
| 0.10 | 0.0625 +/- 0.0075 | 0.0659 +/- 0.0083 | 0.0447 +/- 0.0023, -0.646, 2.069 |
| 0.05 | 0.0552 +/- 0.0078 | 0.0563 +/- 0.0089 | 0.0416 +/- 0.0025, -0.643, 2.067 |
| 0.025 | 0.0547 +/- 0.0074 | 0.0576 +/- 0.0090 | 0.0407 +/- 0.0024, -0.641, 2.063 |
| 0.0125 | 0.0559 +/- 0.0079 | 0.0527 +/- 0.0089 | 0.0379 +/- 0.0027, -0.639, 2.060 |
| 0.00625 | 0.0551 +/- 0.0079 | 0.0513 +/- 0.0088 | 0.0372 +/- 0.0022, -0.638, 2.059 |
| 0.003125 | 0.0544 +/- 0.0078 | 0.0500 +/- 0.0087 | 0.0351 +/- 0.0026, -0.636, 2.056 |

The plateau is reproduced. Three things the 2004 study could not say about it:

The "averaging behaviour" has a quantitative form. With the first-pass R_c the plateau matched the
unforced exponent at R_eff = sqrt(R_c^2 - R^2) within 1 sigma for all seven R (the multiple-timescale
argument the thesis proposed in its conclusions). With R_c = 1/0.6 the effective point moves from
1.6667 at R = 0 to 1.660 at R = 0.15, which is why the plateau rises with R: <A> = 1/sqrt(R_c^2 - R^2)
increases with R and pushes the operating point deeper into the chaotic band. The trace identity
sum(lambda) = -<A> holds to a median of 1e-5 nats across 300 omega values (`v3_spectrum.png`).

The scatter in the thesis's Fig 5.1 and 5.5 is finite-time noise, not omega dependence. The
per-parameter block standard error of a 2^15-step estimate is 0.006 to 0.007 bits; the across-omega
std the thesis quotes as its uncertainty is 0.006 to 0.010 bits, a ratio of 0.9 to 1.5. To resolve
real omega structure in the plateau, integrate 10 to 100 times longer. That is an R740 job.

The Kaplan-Yorke dimension on the plateau is 2.06 to 2.08 and increases with R, a second measure of
chaos strength that agrees with the exponent ordering.

### Fluctuating band, 0 <= omega <= 2.75 (thesis Figs 5.1, 5.4)

Reproduced: all seven curves sit in the 0.05 to 0.08 bits band, small-R curves are flat near 0.055
with dips, and the R = 0.15 curve peaks near omega = 1.9 (0.10 bits here, 0.12 in the thesis). What
the thesis's Bezier smoothing hid is that the dips are resonances where the modulation locks the
system into a periodic window: omega = 0.19, 0.38, 0.58, 0.95, 1.15, 1.33, 1.52, 1.92, 2.10, 2.50,
2.68 at R = 0.15, and the deepest of them (1.52, 1.92, 2.10) suppress chaos even at R = 0.025. The
two "minima" the thesis reports near omega = 1 and 2.75 are the two widest windows.

### Boundedness

R = 0.15 is the largest amplitude for which every omega in [0, 2.75] stays bounded (`lle_map`
reports 0.153 at 30 000 steps; not yet checked at longer times). Above that, orbits escape for
omega in roughly [0, 0.9] and [1.3, 2.2], with a tongue of bounded chaos around omega = 1.1 that
survives to R = 0.6. The thesis chose 0.15 as its largest R without stating why; this is presumably
why.

### Numerical checks (`verify.py`)

dt: at fixed total time T = 3000, float64, the exponent at dt = 0.05 agrees with dt = 0.0125 within
1.8 block-SE at 11 of 12 test points. The exception is omega = 0.38 (a window edge), where the
negative exponent goes from -0.02 to -0.08 bits. dt = 0.05 is adequate for chaotic values at the
+/- 0.007 bits level and not for the magnitude of negative exponents inside windows.
(`v2_dt_convergence.png`)

Spectrum: lambda_2 is zero to within 0.002 nats on chaotic orbits (it is the negative exponent
inside windows, where lambda_1 becomes the zero one), and the trace check passes.

## Bifurcation diagrams: what was tried and what to use

The task was to add bifurcation diagrams as a second chaos characterisation next to the LLE. Two
Poincare sections were implemented in `bifurcation.py`; both are `vmap`-batched and chunked.

Peak map (local maxima of x, the section Sprott uses for jerk systems) is the one to use for this
system. It works with the forcing on or off, it reproduces the textbook Linz-Sprott cascade for the
unforced sweep in R_c (period doubling from R_c = 1.36, chaos from 1.55, period-5 window at
1.67-1.72 just above the operating point 1/0.6, attractor destroyed at 1.83, i.e. A = 0.55), and in the forced sweep in omega every
periodic window is a sharp gap with visible period-doubling structure inside.

Stroboscopic map (state once per forcing period) is the textbook section for periodically forced
oscillators, and it is implemented with a per-parameter dt = T/512 so the stride is static under
`vmap`. For this system it is much less legible: the forcing is weak (R/R_c = 9 %), the autonomous
attractor dominates, and one sample per period lands anywhere on it, so chaotic columns are a
uniform smear and windows are faint. It would come into its own at larger R inside the bounded
tongue near omega = 1.1, where the forcing is a large fraction of R_c. Keep it, but it is not the
headline figure.

Rendering matters as much as the section. A scatter of a few million points is a grey rectangle.
`scatter_bifurcation` now draws a per-column-normalised 2-D histogram with gamma 0.35, which keeps
a period-n orbit (n bins, high count) and a chaotic band (many bins, low count) both visible.

Numbers behind the diagrams: 40 000 transient steps discarded, 40 000 kept, dt = 0.05, 1 200 columns
per sweep. For the paper, 2 000 to 4 000 columns and a longer transient near the window edges would
be worth the minutes it costs.

## For the R740 run

- Build the venv on a Linux filesystem, not `/mnt/c` or NFS; JAX import time and scan compile time
  both suffer on slow filesystems.
- `run_all.py --quick` is a 15-second smoke test; run it first.
- Set `XLA_FLAGS=--xla_force_host_platform_device_count=N` only if you want to shard across
  sockets with `pmap`; plain `vmap` already saturated all 8 threads here.
- If the R740 has a CUDA GPU, install `jax[cuda12]` instead of `jax jaxlib`; nothing in the code
  changes.
- Escape detection is inside `calculate_lle`; any new sweep that goes beyond R = 0.15 or outside
  R_c in [1.55, 1.82] depends on it.
- Use `lyapunov_spectrum.py` rather than `lle_solver.py` for anything that will be published: it
  costs about 3x, gives all three exponents, D_KY, per-parameter error bars, and the trace check.
- The 2^15-step estimate has a 0.007-bit standard error. Resolving the plateau's omega dependence
  needs 2^19 to 2^20 steps per parameter; that is the first thing the R740 should be spent on.
- `verify.py` (about 12 minutes, dominated by the brute-force Wolf search) should be rerun there
  as the acceptance test.
