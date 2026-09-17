"""
Wolf's fixed-evolution-time (FET) estimator of the largest Lyapunov exponent
from a scalar time series, following the Fortran listing in Appendix B of the
2004 thesis (Wolf, Swift, Swinney, Vastano 1985).

Default parameters are the ones the thesis used:
    tau = 15 samples, ndim = 4, dt = 0.05, evolve = 9,
    dismin = 0.02 * range, dismax = 0.15 * range, thmax = 30 degrees.

The box-grid neighbour search of the original is replaced by a brute-force
search over all embedded points; the acceptance rules are the same:

  search(0): nearest point with dismin <= d < dismax, temporally separated
             from the fiducial point by at least `evolve` samples and not within
             2*evolve of the end.  If none, dismax is doubled and retried.
  search(1): as above, but among points with d < dismax the one with the
             smallest orientation change theta < thmax wins (in the Fortran the
             boxes are scanned in layers so the outcome is "nearest point with
             theta < thmax"; that is what is implemented here).

Returns the exponent in bits per time unit, as the thesis reported it.
"""

import numpy as np


def embed(x, ndim, tau):
    n = len(x) - (ndim - 1) * tau
    return np.stack([x[i * tau : i * tau + n] for i in range(ndim)], axis=1)


def wolf_fet(
    x,
    dt=0.05,
    tau=15,
    ndim=4,
    evolve=9,
    dismin_frac=0.02,
    dismax_frac=0.15,
    thmax=30.0,
    return_trace=False,
):
    x = np.asarray(x, dtype=np.float64)
    rng = x.max() - x.min()
    dismin = dismin_frac * rng
    dismax0 = dismax_frac * rng
    Y = embed(x, ndim, tau)
    npts = len(Y)
    datuse = npts - evolve  # Fortran: datcnt - (ndim-1)*tau - evolve
    idx = np.arange(npts)

    def search(oldpnt, newpnt, dismax, orient):
        d = np.linalg.norm(Y - Y[oldpnt], axis=1)
        ok = (d >= dismin) & (d < dismax) & (np.abs(idx - oldpnt) >= evolve)
        ok &= np.abs(idx - datuse) >= 2 * evolve
        ok &= d > 0
        if orient:
            v_old = Y[oldpnt] - Y[newpnt]
            oldist = np.linalg.norm(v_old)
            v_new = Y[oldpnt] - Y  # (npts, ndim)
            ctheta = np.abs(v_new @ v_old) / np.maximum(d * oldist, 1e-300)
            theta = np.degrees(np.arccos(np.minimum(ctheta, 1.0)))
            ok &= theta < thmax
        if not ok.any():
            return None, None
        cand = np.where(ok)[0]
        best = cand[np.argmin(d[cand])]
        return best, d[best]

    oldpnt = 0
    dismax = dismax0
    while True:
        newpnt, disold = search(oldpnt, oldpnt, dismax, orient=False)
        if newpnt is not None:
            break
        dismax *= 2.0
    dismax = dismax0

    its = 0
    total = 0.0
    trace = []
    while True:
        oldpnt += evolve
        newpnt += evolve
        if oldpnt >= datuse:
            break
        if newpnt >= datuse:
            # fell off the end: back up and take the nearest neighbour again
            oldpnt -= evolve
            dm = dismax
            while True:
                newpnt, disold = search(oldpnt, oldpnt, dm, orient=False)
                if newpnt is not None:
                    break
                dm *= 2.0
            continue
        disnew = np.linalg.norm(Y[oldpnt] - Y[newpnt])
        if disnew < 0.1 * dismin:
            disnew = 0.1 * dismin
        its += 1
        total += np.log(disnew / disold)
        if return_trace:
            trace.append((its * evolve * dt, total / (its * evolve * dt * np.log(2.0))))
        if disnew <= dismax:
            disold = disnew
            continue
        rep, rdis = search(oldpnt, newpnt, dismax, orient=True)
        if rep is None:
            dm = dismax
            while True:
                rep, rdis = search(oldpnt, oldpnt, dm, orient=False)
                if rep is not None:
                    break
                dm *= 2.0
        newpnt, disold = rep, rdis

    lle_bits = total / (its * evolve * dt * np.log(2.0))
    if return_trace:
        return lle_bits, np.array(trace)
    return lle_bits
