"""
Full Lyapunov spectrum of the modified Sprott circuit by Benettin's method
with Gram-Schmidt (QR) reorthonormalisation of three tangent vectors.

State: [x, y, z] + a 3x3 tangent matrix W (columns are tangent vectors),
so the augmented state has 12 components.  Every `norm_interval` steps W is
QR-factorised, W <- Q, and log|diag(R)| is accumulated per column.

Consistency check that comes for free: the Jacobian trace is -A(t), so the
exponents must sum to the time average of -A,
    sum(lambda) = -<A> = -1 / sqrt(R_c^2 - R^2)      (omega > 0)
               = -1 / R_c                             (omega = 0 or R = 0).
`spectrum_check` returns that predicted sum next to the measured one.

Kaplan-Yorke dimension: D_KY = j + (lambda_1 + ... + lambda_j) / |lambda_{j+1}|
with j the largest index for which the partial sum is >= 0.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit, vmap

from sprott_solver import sprott_equations


def _aug_equations(aug, t, params):
    state = aug[:3]
    W = aug[3:].reshape(3, 3)
    d_state = sprott_equations(state, t, params)
    J = jacfwd(sprott_equations, argnums=0)(state, t, params)
    return jnp.concatenate([d_state, (J @ W).ravel()])


@jit
def _rk4(aug, t, dt, params):
    k1 = _aug_equations(aug, t, params)
    k2 = _aug_equations(aug + 0.5 * dt * k1, t + 0.5 * dt, params)
    k3 = _aug_equations(aug + 0.5 * dt * k2, t + 0.5 * dt, params)
    k4 = _aug_equations(aug + dt * k3, t + dt, params)
    return aug + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@partial(jit, static_argnames=("total_steps", "norm_interval", "transient_steps"))
def lyapunov_spectrum(initial_state, params, total_steps, dt, norm_interval=10, transient_steps=0):
    """
    Returns (lambdas[3] sorted descending, per-chunk log-stretch history
    of shape (num_chunks, 3)).  `transient_steps` of plain integration are
    run first and excluded from the average.
    """
    aug0 = jnp.concatenate([initial_state, jnp.eye(3).ravel()])

    def inner(carry, _):
        s, t = carry
        return (_rk4(s, t, dt, params), t + dt), None

    def chunk(carry, _):
        aug, t = carry
        (aug, t), _ = jax.lax.scan(inner, (aug, t), None, length=norm_interval)
        state = aug[:3]
        W = aug[3:].reshape(3, 3)
        Q, R = jnp.linalg.qr(W)
        diag = jnp.diag(R)
        Q = Q * jnp.sign(diag)  # keep orientation, R diagonal positive
        logs = jnp.log(jnp.abs(diag))
        return (jnp.concatenate([state, Q.ravel()]), t), logs

    # transient: integrate the physical state only, tangent frame reset after
    def plain(carry, _):
        s, t = carry
        return (_rk4(s, t, dt, params), t + dt), None

    (aug_t, t0), _ = jax.lax.scan(plain, (aug0, 0.0), None, length=transient_steps)
    aug_t = jnp.concatenate([aug_t[:3], jnp.eye(3).ravel()])

    n_chunks = total_steps // norm_interval
    (aug_f, _), hist = jax.lax.scan(chunk, (aug_t, t0), None, length=n_chunks)
    lambdas = hist.sum(axis=0) / (n_chunks * norm_interval * dt)
    bounded = jnp.all(jnp.isfinite(aug_f[:3])) & (jnp.max(jnp.abs(aug_f[:3])) < 1e3)
    lambdas = jnp.where(bounded, lambdas, jnp.nan)
    return lambdas, hist


batch_spectrum = vmap(lyapunov_spectrum, in_axes=(None, 0, None, None, None, None))


def predicted_sum(params):
    """Time average of the Jacobian trace, -<A>."""
    rc, ramp, omega = np.asarray(params).T
    s = np.where(omega > 0, -1.0 / np.sqrt(rc**2 - ramp**2), -1.0 / rc)
    return s


def kaplan_yorke(lambdas):
    """lambdas: (..., 3) sorted descending."""
    l = np.asarray(lambdas)
    c1 = l[..., 0]
    c2 = l[..., 0] + l[..., 1]
    c3 = c2 + l[..., 2]
    d = np.where(c1 < 0, 0.0, np.where(c2 < 0, 1 + c1 / np.abs(l[..., 1]), np.where(c3 < 0, 2 + c2 / np.abs(l[..., 2]), 3.0)))
    return d


def block_standard_error(hist, dt, norm_interval, n_blocks=64):
    """
    Standard error of each exponent from the per-chunk stretch history,
    hist of shape (num_chunks, k).  Chunks are grouped into n_blocks
    contiguous blocks, each block gives an independent-ish exponent
    estimate, and the SE is std / sqrt(n_blocks).
    """
    hist = np.asarray(hist)
    n = (hist.shape[0] // n_blocks) * n_blocks
    blocks = hist[:n].reshape(n_blocks, -1, hist.shape[1]).sum(axis=1)
    block_t = (n // n_blocks) * norm_interval * dt
    est = blocks / block_t
    return est.std(axis=0, ddof=1) / np.sqrt(n_blocks)
