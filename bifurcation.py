"""
Bifurcation diagrams for the modified Sprott circuit.

    x''' = -A x'' - x' + |x| - 1,   A = 1 / (R_c + R_amp sin(omega t))

Two Poincare-section strategies are implemented, because the system has
two regimes with different natural sections:

1. Peak map (local maxima of x).  Section: y = x' = 0 crossing with z < 0.
   Works for the autonomous case (R_amp = 0, or omega = 0) and for the forced
   case.  This is the section Sprott and Linz use for jerk systems.

2. Stroboscopic map.  Sample the state once per forcing period T = 2*pi/omega.
   Only defined for omega > 0.  This is the standard section for periodically
   forced oscillators (Duffing, forced van der Pol) and is the one that makes
   period-doubling cascades in the forcing look clean: a period-n orbit shows
   up as exactly n points.  To keep the sample stride a static integer under
   vmap, dt is chosen per-parameter as T / STEPS_PER_PERIOD, so the stride is
   the same for every parameter in the batch while dt differs.

Both return a variable number of section points per parameter.  JAX needs
static shapes, so the kernels return a fixed-length array of candidates plus a
boolean mask, and the host side filters.

Memory: only the x-component of the trajectory is kept, and the parameter
batch is processed in chunks so a 2000-parameter sweep fits on a 16 GB laptop.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from sprott_solver import rk4_step

# --- kernels ---------------------------------------------------------------


@partial(jit, static_argnames=("n_transient", "n_keep"))
def _x_series(initial_state, params, dt, n_transient, n_keep):
    """Integrate, discard n_transient steps, return x for the next n_keep steps."""

    def step(carry, _):
        s, t = carry
        return (rk4_step(s, t, dt, params), t + dt), None

    def step_keep(carry, _):
        s, t = carry
        ns = rk4_step(s, t, dt, params)
        return (ns, t + dt), ns[0]

    carry, _ = jax.lax.scan(step, (initial_state, 0.0), None, length=n_transient)
    _, xs = jax.lax.scan(step_keep, carry, None, length=n_keep)
    return xs


@jit
def _peaks(xs):
    """Local maxima of a 1-D series. Returns (values, mask), same length as xs."""
    left = jnp.concatenate([jnp.array([jnp.inf]), xs[:-1]])
    right = jnp.concatenate([xs[1:], jnp.array([jnp.inf])])
    mask = (xs > left) & (xs >= right)
    return xs, mask


def peak_map_batch(initial_state, batch_params, dt, n_transient, n_keep, chunk=200):
    """
    Peak-map bifurcation data.

    Returns a list of 1-D numpy arrays, one per parameter row: the local
    maxima of x after the transient.
    """
    f = vmap(_x_series, in_axes=(None, 0, None, None, None))
    out = []
    for i in range(0, len(batch_params), chunk):
        xs = f(initial_state, batch_params[i : i + chunk], dt, n_transient, n_keep)
        vals, mask = vmap(_peaks)(xs)
        vals = np.asarray(vals)
        mask = np.asarray(mask)
        out.extend(vals[j][mask[j]] for j in range(len(vals)))
    return out


@partial(jit, static_argnames=("stride", "n_transient_periods", "n_periods"))
def _strobe_series(initial_state, params, dt, stride, n_transient_periods, n_periods):
    """
    Integrate with per-parameter dt so that `stride` steps == one forcing
    period.  Returns the full state at the end of each retained period,
    shape (n_periods, 3).
    """

    def period(carry, _):
        def step(c, _):
            s, t = c
            return (rk4_step(s, t, dt, params), t + dt), None

        carry, _ = jax.lax.scan(step, carry, None, length=stride)
        return carry, carry[0]

    carry, _ = jax.lax.scan(period, (initial_state, 0.0), None, length=n_transient_periods)
    _, samples = jax.lax.scan(period, carry, None, length=n_periods)
    return samples


def stroboscopic_batch(
    initial_state, batch_params, steps_per_period, n_transient_periods, n_periods, chunk=200
):
    """
    Stroboscopic bifurcation data.  batch_params[:, 2] (omega) must be > 0.

    Returns array of shape (n_params, n_periods, 3).  dt for row i is
    2*pi / (omega_i * steps_per_period).
    """
    omegas = batch_params[:, 2]
    if bool(jnp.any(omegas <= 0)):
        raise ValueError("stroboscopic map needs omega > 0 for every row")
    dts = 2.0 * jnp.pi / (omegas * steps_per_period)
    f = vmap(_strobe_series, in_axes=(None, 0, 0, None, None, None))
    out = []
    for i in range(0, len(batch_params), chunk):
        out.append(
            np.asarray(
                f(
                    initial_state,
                    batch_params[i : i + chunk],
                    dts[i : i + chunk],
                    steps_per_period,
                    n_transient_periods,
                    n_periods,
                )
            )
        )
    return np.concatenate(out, axis=0)


# --- plotting helper ---------------------------------------------------------


def scatter_bifurcation(ax, sweep_values, samples, ybins=700, gamma=0.35, cmap="Greys"):
    """
    Density rendering of a bifurcation diagram.

    samples: list (len = len(sweep_values)) of 1-D arrays of section values.
    Each parameter value gets one column; counts per column are normalised so
    a periodic orbit (few distinct values, high count each) and a chaotic
    band (many values, low count each) both stay visible, then a power-law
    gamma < 1 lifts the faint chaotic smear.  Scatter plots with millions of
    points render as a uniform grey block; this does not.
    """
    sweep_values = np.asarray(sweep_values)
    n = len(sweep_values)
    xi = np.repeat(np.arange(n), [len(v) for v in samples])
    ys = np.concatenate(samples)
    lo, hi = np.nanmin(ys), np.nanmax(ys)
    pad = 0.02 * (hi - lo)
    yedges = np.linspace(lo - pad, hi + pad, ybins + 1)
    H, _, _ = np.histogram2d(xi, ys, bins=[np.arange(n + 1) - 0.5, yedges])
    colmax = H.max(axis=1, keepdims=True)
    colmax[colmax == 0] = 1
    img = (H / colmax) ** gamma
    # x extent uses the real parameter values; assumes a uniform sweep grid
    dx = (sweep_values[-1] - sweep_values[0]) / max(n - 1, 1)
    ax.imshow(
        img.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[sweep_values[0] - dx / 2, sweep_values[-1] + dx / 2, yedges[0], yedges[-1]],
        interpolation="nearest",
    )
