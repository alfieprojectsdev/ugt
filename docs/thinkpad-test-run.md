# ThinkPad T14 test run, 2026-09-17

Full run of the pipeline on the laptop before the Dell R740 run. Everything here was produced by
`python run_all.py` from commit on branch `claude/thesis-thinkpad-bifurcation-5c739e`; figures are in
`results/`, raw per-stage numbers in `results/timings.json` and `results_full.log`.

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
exponents converted with 1/ln 2.

**Plateau, omega > 2.75 (thesis Fig 5.5).** Mean +/- std over 3 420 omega values per R:

| R | thesis (bits/s) | this run (bits) | unforced LLE at R_eff = sqrt(R_c^2 - R^2), same step count |
|---|---|---|---|
| 0.15 | 0.0666 +/- 0.0072 | 0.0654 +/- 0.0082 | 0.075 |
| 0.10 | 0.0625 +/- 0.0075 | 0.0539 +/- 0.0096 | 0.052 |
| 0.05 | 0.0552 +/- 0.0078 | 0.0318 +/- 0.0104 | 0.041 |
| 0.025 | 0.0547 +/- 0.0074 | 0.0067 +/- 0.0049 | 0.007 |
| 0.0125 | 0.0559 +/- 0.0079 | -0.0025 +/- 0.0034 | -0.003 |
| 0.00625 | 0.0551 +/- 0.0079 | -0.0045 +/- 0.0027 | -0.004 |
| 0.003125 | 0.0544 +/- 0.0078 | -0.0027 +/- 0.0019 | -0.003 |

Two things follow.

The "averaging behaviour" the thesis observed and proposed to explain with a multiple-timescale
argument checks out quantitatively: the plateau equals the unforced exponent at the period-averaged
conductance <A> = 1/sqrt(R_c^2 - R^2), within one standard deviation for all seven R. The plateau
rises with R because <A> rises with R, moving the effective operating point deeper into the chaotic
band.

The thesis's finding that "for R < 0.05 the average DLE values for higher omega no longer decrease"
(plateau stuck near 0.055 bits/s) is not reproduced. The reason is at the unforced limit: R_c = 1.67
is Sprott's A = 0.6 rounded, and A = 0.5988 lies inside a period-5 window of the unforced system
(see `03_bif_rc_unforced.png`; the window runs from about 1.67 to 1.72). Spot check, float64,
2^19 steps: LLE(R_c = 1.6667) = +0.054 bits (chaotic, matches the literature value of about
0.036 nats), LLE(R_c = 1.67) = -0.013 bits (periodic). So for R small enough that the modulation
stays inside the window, the system is periodic and the exponent is zero or negative. Wolf's
algorithm on a finite noisy series has a positive floor and cannot return a negative exponent, which
is the likely origin of the 0.055 plateau. This is worth a paragraph in the re-evaluation paper: the
2004 "chaos baseline" was, in the R -> 0 limit, a period-5 orbit.

**Fluctuating band, 0 <= omega <= 2.75 (thesis Figs 5.1, 5.4).** Reproduced in outline: the R = 0.15
curve has its highest values near omega = 1.9 (0.10 bits here, 0.12 in the thesis) and the R-ordering
of the curves is preserved. What the thesis's Bezier smoothing hid is that the band contains
periodic windows where the exponent drops to zero or below: omega = 0.38, 0.49, 0.63, 0.77, 1.15,
1.47, 1.63, 1.75, 1.93, 2.68 at R = 0.15. The two "minima" the thesis reports near omega = 1 and 2.75
are the two widest of these windows.

**Boundedness.** R = 0.15 is the largest amplitude for which every omega in [0, 2.75] stays bounded
(`lle_map` reports 0.153). Above that, orbits escape for omega in roughly [0, 0.9] and [1.3, 2.2],
with a tongue of bounded chaos around omega = 1.1 that survives to R = 0.6. The thesis chose 0.15 as
its largest R without stating why; this is presumably why.

## Bifurcation diagrams: what was tried and what to use

The task was to add bifurcation diagrams as a second chaos characterisation next to the LLE. Two
Poincare sections were implemented in `bifurcation.py`; both are `vmap`-batched and chunked.

Peak map (local maxima of x, the section Sprott uses for jerk systems) is the one to use for this
system. It works with the forcing on or off, it reproduces the textbook Linz-Sprott cascade for the
unforced sweep in R_c (period doubling from R_c = 1.36, chaos from 1.55, period-5 window at
1.67-1.72, attractor destroyed at 1.83, i.e. A = 0.55), and in the forced sweep in omega every
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
