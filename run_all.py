"""
Full headless run of the thesis pipeline plus bifurcation analysis.

    python run_all.py [--quick] [--x64] [--out results]

Stages (each writes PNG + NPZ to the output dir and a line to timings.json):

  1. trajectories   phase portraits for a handful of omegas (sprott_solver)
  2. lle_omega      LLE vs omega at R_amp = 0.15 (thesis Fig 5.1, lle_solver)
  3. bif_rc         peak-map bifurcation vs R_c, unforced (Linz-Sprott check)
  4. bif_omega      peak-map + stroboscopic bifurcation vs omega, R_amp = 0.15
  5. bif_ramp       peak-map bifurcation vs R_amp at fixed omega
  6. lle_map        2-D LLE map over (R_amp, omega)
  7. thesis_grid    the 2004 protocol verbatim: 7 R x 5120 omega x 2^15 steps

--quick shrinks every sweep ~4x for a smoke test.
--x64 enables float64 in JAX (default float32).
"""

import argparse
import json
import os
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--x64", action="store_true")
ap.add_argument("--out", default="results")
ap.add_argument("--stages", default="all", help="comma list, e.g. lle_omega,bif_omega")
ap.add_argument("--rc", type=float, default=1.0 / 0.6, help="R_c; 1/0.6 is Sprott A=0.6 and what the 2004 code used, 1.67 is the rounded value in the thesis text")
args = ap.parse_args()

import jax  # noqa: E402

if args.x64:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from bifurcation import (  # noqa: E402
    peak_map_batch,
    scatter_bifurcation,
    stroboscopic_batch,
)
from lle_solver import batch_lle  # noqa: E402
from sprott_solver import batch_simulate  # noqa: E402

OUT = args.out
os.makedirs(OUT, exist_ok=True)
Q = 0.25 if args.quick else 1.0
DT = 0.05
R_C = args.rc
R_AMP = 0.15
X0 = jnp.array([0.05, 0.0, 0.0])
timings = {
    "host": platform.node(),
    "cpu": platform.processor(),
    "python": sys.version.split()[0],
    "jax": jax.__version__,
    "devices": [str(d) for d in jax.devices()],
    "x64": args.x64,
    "quick": args.quick,
    "R_c": R_C,
    "stages": {},
}
# a partial --stages run keeps the other stages' entries from the previous timings.json
try:
    with open(os.path.join(OUT, "timings.json")) as f:
        timings["stages"] = json.load(f).get("stages", {})
except (OSError, ValueError):
    pass


def want(stage):
    return args.stages == "all" or stage in args.stages.split(",")


def timed(stage, fn):
    print(f"\n=== {stage} ===", flush=True)
    t0 = time.perf_counter()
    info = fn() or {}
    dt = time.perf_counter() - t0
    info["seconds"] = round(dt, 2)
    timings["stages"][stage] = info
    print(f"    {stage}: {dt:.1f}s {info}", flush=True)
    with open(os.path.join(OUT, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)


def params_for(rc, ramp, omega):
    rc, ramp, omega = np.broadcast_arrays(rc, ramp, omega)
    return jnp.asarray(np.stack([rc, ramp, omega], axis=1))


# --- 1. trajectories ---------------------------------------------------------


def stage_trajectories():
    steps = 32768
    omegas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.75])
    P = params_for(R_C, R_AMP, omegas)
    t0 = time.perf_counter()
    batch_simulate(X0, P, steps, DT).block_until_ready()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    traj = np.asarray(batch_simulate(X0, P, steps, DT).block_until_ready())
    run_s = time.perf_counter() - t0
    np.savez_compressed(os.path.join(OUT, "trajectories.npz"), omegas=omegas, traj=traj, dt=DT)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, w, tr in zip(axes.flat, omegas, traj):
        keep = tr[steps // 4 :]
        ax.plot(keep[:, 0], keep[:, 1], lw=0.3, color="black")
        ax.set_title(f"omega = {w:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("dx/dt")
    fig.suptitle(f"Phase portraits, R_c={R_C}, R_amp={R_AMP}, dt={DT}, {steps} steps (last 3/4 shown)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_trajectories.png"), dpi=130)
    plt.close(fig)
    return {
        "n_params": len(omegas),
        "steps": steps,
        "compile_s": round(compile_s, 2),
        "run_s": round(run_s, 3),
        "nan": bool(np.isnan(traj).any()),
        "steps_per_s": round(len(omegas) * steps / run_s),
    }


# --- 2. LLE vs omega ---------------------------------------------------------


def stage_lle_omega():
    """
    Thesis protocol: omega on [0, 20], LLE per (R, omega).  Thesis used
    d_omega = 2^-8 (5120 points); here the full range is coarser and the
    fluctuating band [0, 2.75] gets a dense second pass.
    """
    steps = int(60000 * Q)
    ramps = [0.15, 0.05]
    colors = {0.15: "black", 0.05: "tab:blue"}
    out = {"steps": steps}
    LN2 = np.log(2)

    def sweep(omegas, tag):
        res = {}
        for ramp in ramps:
            P = params_for(R_C, ramp, omegas)
            t0 = time.perf_counter()
            lles, hist = batch_lle(X0, P, steps, DT, 10)
            lles = np.asarray(lles.block_until_ready())
            hist = np.asarray(hist)
            secs = time.perf_counter() - t0
            t_axis = np.arange(1, hist.shape[1] + 1) * 10 * DT
            running = hist / t_axis
            drift = np.abs(running[:, -1] - running[:, int(0.9 * hist.shape[1])]).max()
            res[ramp] = lles
            out[f"{tag}_R{ramp}"] = {
                "seconds": round(secs, 1),
                "n": len(omegas),
                "max_drift_last10pct": float(drift),
                "mean_bits_per_s": float(lles.mean() / LN2),
            }
            np.savez_compressed(os.path.join(OUT, f"lle_{tag}_R{ramp}.npz"), omegas=omegas, lle=lles)
        return res

    # (a) full thesis range, Fig 5.1 / 5.2
    om_full = np.linspace(0.0, 20.0, int(800 * Q))
    full = sweep(om_full, "full")
    # (b) dense pass on the fluctuating band, Fig 5.4
    om_zoom = np.linspace(0.0, 3.0, int(600 * Q))
    zoom = sweep(om_zoom, "zoom")

    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    for ax, om, data, title in [
        (axes[0], om_full, full, "omega on [0, 20]  (thesis Fig 5.1 / 5.2)"),
        (axes[1], om_zoom, zoom, "omega on [0, 3]  (thesis Fig 5.4, unsmoothed)"),
    ]:
        for ramp in ramps:
            ax.plot(om, data[ramp] / LN2, lw=0.7, color=colors[ramp], label=f"R_amp = {ramp}")
        ax.axhline(0, color="red", ls="--", lw=0.6)
        ax.set_ylabel("LLE (bits / time unit)")
        ax.set_title(f"{title}, R_c={R_C}, {steps} steps, dt={DT}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("omega (rad / time unit)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_lle_vs_omega.png"), dpi=130)
    plt.close(fig)
    plateau = full[0.15][om_full > 5].mean() / LN2
    out["plateau_bits_R0.15_omega>5"] = float(plateau)
    return out


# --- 3. bifurcation vs R_c, unforced ----------------------------------------


def stage_bif_rc():
    n = int(1200 * Q)
    rcs = np.linspace(1.3, 1.9, n)  # A = 1/R_c from 0.77 down to 0.53; orbits escape past ~1.83
    P = params_for(rcs, 0.0, 0.0)
    n_tr, n_keep = int(40000 * Q), int(40000 * Q)
    peaks = peak_map_batch(X0, P, DT, n_tr, n_keep)
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter_bifurcation(ax, rcs, peaks)
    ax.axvline(1.0 / 0.6, color="red", lw=0.6, ls="--", label="R_c = 1/0.6 (Sprott A = 0.6, 2004 code)")
    ax.axvline(1.67, color="tab:blue", lw=0.6, ls=":", label="R_c = 1.67 (thesis text, rounded)")
    ax.set_xlabel("R_c   (A = 1/R_c, unforced: R_amp = 0)")
    ax.set_ylabel("local maxima of x")
    ax.set_title("Peak-map bifurcation diagram of the unforced Sprott/Linz jerk system")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_bif_rc_unforced.png"), dpi=150)
    plt.close(fig)
    np.savez_compressed(
        os.path.join(OUT, "bif_rc.npz"),
        rcs=rcs,
        peak_values=np.concatenate(peaks),
        peak_param_index=np.repeat(np.arange(n), [len(p) for p in peaks]),
    )
    return {"n_params": n, "n_transient": n_tr, "n_keep": n_keep, "peaks_total": int(sum(len(p) for p in peaks))}


# --- 4. bifurcation vs omega, forced -----------------------------------------


def stage_bif_omega():
    n = int(1200 * Q)
    omegas = np.linspace(0.25, 2.75, n)
    P = params_for(R_C, R_AMP, omegas)
    out = {"n_params": n}

    # (a) peak map
    n_tr, n_keep = int(40000 * Q), int(40000 * Q)
    t0 = time.perf_counter()
    peaks = peak_map_batch(X0, P, DT, n_tr, n_keep)
    out["peak_map_s"] = round(time.perf_counter() - t0, 1)

    # (b) stroboscopic: dt = T/steps_per_period; at omega=0.25, T=25.1 -> dt=0.049
    spp, n_tp, n_p = 512, int(200 * Q), int(300 * Q)
    t0 = time.perf_counter()
    strobe = stroboscopic_batch(X0, P, spp, n_tp, n_p)
    out["strobe_s"] = round(time.perf_counter() - t0, 1)
    out["strobe_cfg"] = {"steps_per_period": spp, "transient_periods": n_tp, "periods": n_p}

    fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    scatter_bifurcation(axes[0], omegas, peaks)
    axes[0].set_ylabel("local maxima of x")
    axes[0].set_title(f"Peak map vs omega, R_c={R_C}, R_amp={R_AMP}")
    scatter_bifurcation(axes[1], omegas, [s[:, 0] for s in strobe])
    axes[1].set_ylabel("x sampled once per forcing period")
    axes[1].set_xlabel("omega (rad/s)")
    axes[1].set_title("Stroboscopic map vs omega (same parameters)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_bif_omega.png"), dpi=150)
    plt.close(fig)
    np.savez_compressed(os.path.join(OUT, "bif_omega_strobe.npz"), omegas=omegas, strobe=strobe)
    return out


# --- 5. bifurcation vs R_amp -------------------------------------------------


def stage_bif_ramp():
    n = int(1200 * Q)
    ramps = np.linspace(0.0, 0.6, n)
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    for ax, w in zip(axes, [1.0, 2.0]):
        P = params_for(R_C, ramps, w)
        peaks = peak_map_batch(X0, P, DT, int(40000 * Q), int(40000 * Q))
        scatter_bifurcation(ax, ramps, peaks)
        ax.set_ylabel("local maxima of x")
        ax.set_title(f"Peak map vs R_amp, R_c={R_C}, omega={w}")
    axes[1].set_xlabel("R_amp")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_bif_ramp.png"), dpi=150)
    plt.close(fig)
    return {"n_params": n}


# --- 6. 2-D LLE map ------------------------------------------------------------


def stage_lle_map():
    nr, nw = int(60 * Q), int(80 * Q)
    steps = int(30000 * Q)
    ramps = np.linspace(0.0, 0.6, nr)
    omegas = np.linspace(0.0, 2.75, nw)
    RR, WW = np.meshgrid(ramps, omegas, indexing="ij")
    P = params_for(R_C, RR.ravel(), WW.ravel())
    lles = []
    chunk = 400
    for i in range(0, len(P), chunk):
        l, _ = batch_lle(X0, P[i : i + chunk], steps, DT, 10)
        lles.append(np.asarray(l.block_until_ready()))
        print(f"    lle_map {min(i + chunk, len(P))}/{len(P)}", flush=True)
    lles = np.concatenate(lles).reshape(nr, nw)
    np.savez_compressed(os.path.join(OUT, "lle_map.npz"), ramps=ramps, omegas=omegas, lle=lles)
    fig, ax = plt.subplots(figsize=(11, 7))
    vmax = np.nanmax(np.abs(lles))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("0.35")
    im = ax.pcolormesh(omegas, ramps, np.ma.masked_invalid(lles), cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im, ax=ax, label="LLE (nats / time unit)")
    ax.set_xlabel("omega (rad / time unit)")
    ax.set_ylabel("R_amp")
    ax.set_title(f"LLE over (R_amp, omega), R_c={R_C}, {steps} steps, dt={DT}.  Red = chaotic, grey = orbit escapes")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "06_lle_map.png"), dpi=150)
    plt.close(fig)
    escaped = np.isnan(lles)
    return {
        "grid": [nr, nw],
        "steps": steps,
        "escaped_fraction": float(escaped.mean()),
        "chaotic_fraction_of_bounded": float((lles[~escaped] > 0.005).mean()),
        "max_R_amp_fully_bounded": float(ramps[np.where(escaped.any(axis=1))[0][0] - 1]) if escaped.any() else float(ramps[-1]),
    }


# --- 7. thesis protocol grid -------------------------------------------------


def stage_thesis_grid():
    """
    The 2004 protocol verbatim: R in the seven listed values, omega on
    [0, 20] with d_omega = 2^-8 (5120 points), 2^15 RK4 steps at dt = 0.05,
    LLE in bits per time unit.  Reproduces Figs 5.1, 5.4 and 5.5.
    """
    LN2 = np.log(2)
    R_LIST = [0.15, 0.10, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    d_omega = 2.0 ** -8
    omegas = np.arange(0.0, 20.0, d_omega * (4 if args.quick else 1))
    steps = 2**15
    chunk = 512
    lle = np.zeros((len(R_LIST), len(omegas)))
    t_all = time.perf_counter()
    for ri, ramp in enumerate(R_LIST):
        P = params_for(R_C, ramp, omegas)
        t0 = time.perf_counter()
        for i in range(0, len(P), chunk):
            l, _ = batch_lle(X0, P[i : i + chunk], steps, DT, 10)
            lle[ri, i : i + chunk] = np.asarray(l.block_until_ready())
        print(f"    R={ramp}: {len(omegas)} omegas in {time.perf_counter() - t0:.1f}s", flush=True)
    secs = time.perf_counter() - t_all
    bits = lle / LN2
    np.savez_compressed(os.path.join(OUT, "thesis_grid.npz"), R=np.array(R_LIST), omegas=omegas, lle_nats=lle)

    # Fig 5.1 analogue
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(omegas, bits[0], lw=0.4, color="black")
    ax.set_xlabel("omega (rad / time unit)")
    ax.set_ylabel("LLE (bits / time unit)")
    ax.set_title(f"(R=0.15, omega): {len(omegas)} points, 2^15 steps, dt=0.05, R_c={R_C:.4f}  [thesis Fig 5.1]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "07_thesis_fig51.png"), dpi=130)
    plt.close(fig)

    # Fig 5.4 analogue: zoom on [0, 3], one curve per R, light smoothing
    fig, ax = plt.subplots(figsize=(11, 6))
    m = omegas <= 3.0
    k = 9 if not args.quick else 3
    kern = np.ones(k) / k
    for ri, ramp in enumerate(R_LIST):
        sm = np.convolve(bits[ri, m], kern, mode="same")
        ax.plot(omegas[m], sm, lw=0.9, label=f"R = {ramp}")
    ax.set_xlabel("omega (rad / time unit)")
    ax.set_ylabel(f"LLE (bits / time unit), {k}-point moving average")
    ax.set_title("Small-omega band, all seven R values  [thesis Fig 5.4]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "07_thesis_fig54.png"), dpi=130)
    plt.close(fig)

    # Fig 5.5 analogue: plateau statistics for omega > 2.75
    big = omegas > 2.75
    plateau = {}
    fig, ax = plt.subplots(figsize=(11, 5))
    for ri, ramp in enumerate(R_LIST):
        mu, sd = bits[ri, big].mean(), bits[ri, big].std()
        plateau[str(ramp)] = {"mean_bits": round(float(mu), 4), "std_bits": round(float(sd), 4)}
        ax.plot(omegas[big], np.convolve(bits[ri, big], np.ones(25) / 25, mode="same"), lw=0.8,
                label=f"{mu:.4f} +/- {sd:.4f} bits   R = {ramp}")
    ax.set_xlabel("omega (rad / time unit)")
    ax.set_ylabel("LLE (bits / time unit), 25-point moving average")
    ax.set_title("Averaging regime, omega > 2.75  [thesis Fig 5.5]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "07_thesis_fig55.png"), dpi=130)
    plt.close(fig)
    return {
        "n_R": len(R_LIST),
        "n_omega": len(omegas),
        "steps": steps,
        "aug_steps_per_s": round(len(R_LIST) * len(omegas) * steps / secs),
        "plateau_omega_gt_2.75": plateau,
    }


STAGES = [
    ("trajectories", stage_trajectories),
    ("lle_omega", stage_lle_omega),
    ("bif_rc", stage_bif_rc),
    ("bif_omega", stage_bif_omega),
    ("bif_ramp", stage_bif_ramp),
    ("lle_map", stage_lle_map),
    ("thesis_grid", stage_thesis_grid),
]

print(json.dumps({k: v for k, v in timings.items() if k != "stages"}, indent=2))
T0 = time.perf_counter()
for name, fn in STAGES:
    if want(name):
        timed(name, fn)
timings["total_seconds"] = round(time.perf_counter() - T0, 1)
with open(os.path.join(OUT, "timings.json"), "w") as f:
    json.dump(timings, f, indent=2)
print(f"\nTotal: {timings['total_seconds']}s  ->  {OUT}/")
