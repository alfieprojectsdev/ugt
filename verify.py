"""
Verification checks behind docs/thinkpad-test-run.md.  Writes to results/verify/.

  1. wolf        Wolf FET (thesis parameters) on x series vs Benettin exponent,
                 on periodic and chaotic orbits, with and without the transient
  2. dt          LLE convergence in dt at fixed total time, float64
  3. spectrum    full spectrum, sum-of-exponents check against -<A>, Kaplan-Yorke
  4. errors      per-parameter standard errors of the LLE from block averaging
  5. rc          plateau with R_c = 1.67 vs 1/0.6 against the thesis Fig 5.5 legend

    python verify.py [--checks wolf,dt,spectrum,errors] [--out results/verify]
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checks", default="rc,wolf,dt,spectrum,errors")
ap.add_argument("--out", default="results/verify")
args = ap.parse_args()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from lle_solver import batch_lle  # noqa: E402
from lyapunov_spectrum import (  # noqa: E402
    batch_spectrum,
    block_standard_error,
    kaplan_yorke,
    predicted_sum,
)
from sprott_solver import batch_simulate  # noqa: E402
from wolf_fet import wolf_fet  # noqa: E402

OUT = args.out
os.makedirs(OUT, exist_ok=True)
LN2 = np.log(2)
DT = 0.05
X0 = jnp.array([0.05, 0.0, 0.0])
RC = 1.0 / 0.6  # what the 2004 code used (see check_rc); the thesis text rounds it to 1.67
report = {}


def P(*rows):
    return jnp.asarray(np.array(rows, dtype=np.float64))


def save():
    with open(os.path.join(OUT, "verify.json"), "w") as f:
        json.dump(report, f, indent=2)


def lle_bits(params, steps, norm=10):
    l, h = batch_lle(X0, params, steps, DT, norm)
    return np.asarray(l) / LN2, np.asarray(h)


# --- 1. Wolf FET vs Benettin ----------------------------------------------------


def check_wolf():
    cases = {
        "A  Rc=1.67 R=0 (2004 baseline, unforced)": (1.67, 0.0, 0.0),
        "B  Rc=1.67 R=0.003125 w=10 (smallest thesis R, plateau)": (1.67, 0.003125, 10.0),
        "C  Rc=1.67 R=0.15 w=10 (largest thesis R, plateau)": (1.67, 0.15, 10.0),
        "D  Rc=1.6667 R=0 (Sprott A=0.6, chaotic)": (1.6667, 0.0, 0.0),
    }
    N = 2**15
    TRANS = 200_000
    params = P(*cases.values())
    # thesis-like series: from X0, first 2^15 samples (transient included)
    traj_short = np.asarray(batch_simulate(X0, params, N, DT))
    # transient-free series: 2^15 samples after 200k steps
    traj_long = np.asarray(batch_simulate(X0, params, TRANS + N, DT))[:, TRANS:]
    ben_short, _ = lle_bits(params, N)
    ben_long, _ = lle_bits(params, 2**19)
    out = {}
    print("  case | Wolf(2^15, transient in) | Wolf(2^15, after 200k) | Benettin 2^15 | Benettin 2^19   [bits]")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for k, (name, ax) in enumerate(zip(cases, axes.flat)):
        t0 = time.perf_counter()
        w_short, tr_s = wolf_fet(traj_short[k, :, 0], return_trace=True)
        w_long, tr_l = wolf_fet(traj_long[k, :, 0], return_trace=True)
        secs = time.perf_counter() - t0
        out[name] = {
            "wolf_bits_transient_included": float(w_short),
            "wolf_bits_after_transient": float(w_long),
            "benettin_bits_2^15": float(ben_short[k]),
            "benettin_bits_2^19": float(ben_long[k]),
            "wolf_seconds": round(secs, 1),
        }
        print(f"  {name[:2]} | {w_short:+.4f} | {w_long:+.4f} | {ben_short[k]:+.4f} | {ben_long[k]:+.4f}")
        ax.plot(tr_s[:, 0], tr_s[:, 1], lw=0.8, label="Wolf, transient included")
        ax.plot(tr_l[:, 0], tr_l[:, 1], lw=0.8, label="Wolf, after 200k-step transient")
        ax.axhline(ben_long[k], color="red", ls="--", lw=0.8, label=f"Benettin 2^19 = {ben_long[k]:+.3f}")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("time")
        ax.set_ylabel("running LLE (bits)")
        ax.legend(fontsize=7)
    fig.suptitle("Wolf FET (tau=15, dim=4, evolve=9, dismin=2%, dismax=15%, thmax=30) vs Benettin")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v1_wolf_vs_benettin.png"), dpi=130)
    plt.close(fig)

    # how long is the transient at Rc = 1.67, R = 0?
    x = np.asarray(batch_simulate(X0, P((1.67, 0.0, 0.0)), 400_000, DT))[0, :, 0]
    pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    vals = x[pk]
    settled = None
    for i in range(len(vals) - 60):
        if np.abs(vals[i : i + 50] - vals[i + 5 : i + 55]).max() < 1e-3:
            settled = i
            break
    t_settle = float(pk[settled] * DT) if settled is not None else None
    out["transient_to_period5_at_Rc1.67"] = {"time": t_settle, "steps": int(pk[settled]) if settled is not None else None}
    print(f"  transient to period-5 lock at Rc=1.67: t = {t_settle} (steps {out['transient_to_period5_at_Rc1.67']['steps']}); thesis series length = {N*DT}")
    report["wolf"] = out


# --- 2. dt convergence -----------------------------------------------------------


def check_dt():
    cases = [
        (1.67, 0.15, 0.38), (1.67, 0.15, 0.5), (1.67, 0.15, 1.0), (1.67, 0.15, 1.9),
        (1.67, 0.15, 2.68), (1.67, 0.15, 5.0), (1.67, 0.15, 10.0), (1.67, 0.05, 10.0),
        (1.67, 0.003125, 10.0), (1.6667, 0.0, 0.0), (1.67, 0.0, 0.0), (1.6, 0.0, 0.0),
    ]
    params = P(*cases)
    T = 3000.0
    dts = [0.05, 0.025, 0.0125]
    res = {}
    for dt in dts:
        steps = int(T / dt)
        norm = int(round(0.5 / dt))  # renormalise every 0.5 time units
        t0 = time.perf_counter()
        l, h = batch_lle(X0, params, steps, dt, norm)
        l = np.asarray(l) / LN2
        h = np.asarray(h)
        inc = np.diff(h, axis=1, prepend=0.0)  # lle_solver history is cumulative
        se = np.array([block_standard_error(inc[i][:, None], dt, norm)[0] for i in range(len(params))]) / LN2
        res[dt] = (l, se, time.perf_counter() - t0)
        print(f"  dt={dt}: {steps} steps, {res[dt][2]:.1f}s")
    table = []
    print("  (Rc, R, w) | " + " | ".join(f"dt={dt}" for dt in dts) + "  [bits, +/- block SE]")
    for i, c in enumerate(cases):
        row = {"params": c}
        for dt in dts:
            row[f"dt{dt}"] = [float(res[dt][0][i]), float(res[dt][1][i])]
        d = abs(res[0.05][0][i] - res[0.0125][0][i])
        row["diff_0.05_vs_0.0125"] = float(d)
        row["diff_in_SE_units"] = float(d / np.hypot(res[0.05][1][i], res[0.0125][1][i]))
        table.append(row)
        print(f"  {c} | " + " | ".join(f"{res[dt][0][i]:+.4f}+/-{res[dt][1][i]:.4f}" for dt in dts) + f" | diff {d:.4f} ({row['diff_in_SE_units']:.1f} SE)")
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(cases))
    for j, dt in enumerate(dts):
        ax.errorbar(xs + 0.25 * (j - 1), res[dt][0], yerr=res[dt][1], fmt="o", ms=4, capsize=2, label=f"dt = {dt}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c[0]},{c[1]},{c[2]}" for c in cases], rotation=45, fontsize=8)
    ax.set_ylabel("LLE (bits), T = 3000, float64")
    ax.axhline(0, color="k", lw=0.4)
    ax.legend()
    ax.set_title("dt convergence of the Benettin exponent; error bars = block standard error")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v2_dt_convergence.png"), dpi=130)
    plt.close(fig)
    report["dt"] = {"T": T, "rows": table, "seconds": {str(dt): round(res[dt][2], 1) for dt in dts}}


# --- 3. full spectrum ---------------------------------------------------------------


def check_spectrum():
    omegas = np.linspace(0.02, 3.0, 300)
    params = P(*[(RC, 0.15, w) for w in omegas])
    steps, norm, trans = 60000, 10, 10000
    t0 = time.perf_counter()
    lam, hist = batch_spectrum(X0, params, steps, DT, norm, trans)
    lam = np.asarray(lam)
    secs = time.perf_counter() - t0
    pred = predicted_sum(params)
    meas = lam.sum(axis=1)
    dky = kaplan_yorke(lam)
    print(f"  300 spectra x {steps} steps: {secs:.1f}s")
    print(f"  sum check: max |sum(lambda) + <A>| = {np.nanmax(np.abs(meas - pred)):.2e}, median = {np.nanmedian(np.abs(meas - pred)):.2e}")
    print(f"  lambda_2: median {np.nanmedian(lam[:,1]):+.2e}, max |lambda_2| = {np.nanmax(np.abs(lam[:,1])):.2e} (should be ~0 for a flow)")
    print(f"  D_KY range on chaotic points: {np.nanmin(dky[lam[:,0] > 0.005]):.3f} .. {np.nanmax(dky[lam[:,0] > 0.005]):.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(omegas, lam[:, 0], lw=0.8, label="lambda_1")
    axes[0].plot(omegas, lam[:, 1], lw=0.8, label="lambda_2")
    axes[0].plot(omegas, lam[:, 2], lw=0.8, label="lambda_3")
    axes[0].axhline(0, color="k", lw=0.4)
    axes[0].set_ylabel("nats / time unit")
    axes[0].legend()
    axes[0].set_title(f"Lyapunov spectrum, Rc=1/0.6, R=0.15, {steps} steps after {trans} transient")
    axes[1].plot(omegas, meas, lw=0.8, label="sum of exponents (measured)")
    axes[1].plot(omegas, pred, lw=0.8, ls="--", label="-1/sqrt(Rc^2 - R^2) (Jacobian trace average)")
    axes[1].set_ylabel("nats / time unit")
    axes[1].legend()
    axes[2].plot(omegas, dky, lw=0.8, color="k")
    axes[2].set_ylabel("Kaplan-Yorke dimension")
    axes[2].set_xlabel("omega")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v3_spectrum.png"), dpi=130)
    plt.close(fig)
    np.savez_compressed(os.path.join(OUT, "spectrum_omega.npz"), omegas=omegas, lambdas=lam, dky=dky)

    # plateau table with the sum check
    R_LIST = [0.15, 0.10, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    p2 = P(*[(RC, r, 10.0) for r in R_LIST])
    lam2, hist2 = batch_spectrum(X0, p2, 200000, DT, norm, trans)
    lam2 = np.asarray(lam2)
    se2 = np.array([block_standard_error(np.asarray(hist2[i]), DT, norm) for i in range(len(R_LIST))])
    rows = []
    print("  R | lambda_1 +/- SE | lambda_2 | lambda_3 | sum | -<A> | D_KY   [nats, w=10, 200k steps]")
    for i, r in enumerate(R_LIST):
        rows.append({"R": r, "lambda": lam2[i].tolist(), "se": se2[i].tolist(), "sum": float(lam2[i].sum()), "pred": float(predicted_sum(p2)[i]), "dky": float(kaplan_yorke(lam2[i]))})
        print(f"  {r:<9} | {lam2[i,0]:+.4f} +/- {se2[i,0]:.4f} | {lam2[i,1]:+.4f} | {lam2[i,2]:+.4f} | {lam2[i].sum():+.4f} | {predicted_sum(p2)[i]:+.4f} | {kaplan_yorke(lam2[i]):.3f}")
    report["spectrum"] = {
        "seconds_300x60k": round(secs, 1),
        "sum_check_max_abs": float(np.nanmax(np.abs(meas - pred))),
        "lambda2_max_abs": float(np.nanmax(np.abs(lam[:, 1]))),
        "plateau_w10": rows,
    }


# --- 4. error bars --------------------------------------------------------------------


def check_errors():
    R_LIST = [0.15, 0.10, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    omegas = np.linspace(3.0, 20.0, 200)
    steps, norm = 2**15, 10
    out = {}
    print("  R | across-omega std (thesis-style) | median per-parameter block SE | ratio   [bits, 2^15 steps]")
    for r in R_LIST:
        params = P(*[(RC, r, w) for w in omegas])
        l, h = lle_bits(params, steps, norm)
        inc = np.diff(np.asarray(h), axis=1, prepend=0.0)
        se = np.array([block_standard_error(inc[i][:, None], DT, norm, n_blocks=32)[0] for i in range(len(omegas))]) / LN2
        out[str(r)] = {"mean_bits": float(l.mean()), "std_across_omega": float(l.std()), "median_se_per_param": float(np.median(se)), "max_se_per_param": float(se.max())}
        print(f"  {r:<9} | {l.std():.4f} | {np.median(se):.4f} | {l.std()/np.median(se):.2f}")
    report["errors"] = out


# --- 5. which R_c did the 2004 code use? -----------------------------------------


def check_rc():
    """
    Plateau (omega > 2.75, 2^15 steps, thesis protocol) for R_c = 1.67 as
    written in the thesis and R_c = 1/0.6 = 1.6667 as in Sprott's papers,
    against the seven values printed in the thesis's Fig 5.5 legend.
    """
    thesis = {0.15: 0.0666, 0.10: 0.0625, 0.05: 0.0552, 0.025: 0.0547, 0.0125: 0.0559, 0.00625: 0.0551, 0.003125: 0.0544}
    omegas = np.arange(2.75 + 2**-8, 20.0, 2**-8 * 4)  # every 4th thesis omega
    out = {}
    print("  R | thesis Fig 5.5 | Rc=1.67 | Rc=1/0.6   [bits, mean +/- std over omega > 2.75, 2^15 steps]")
    for rc in [1.67, 1.0 / 0.6]:
        out[str(rc)] = {}
        for r in thesis:
            params = P(*[(rc, r, w) for w in omegas])
            l, _ = lle_bits(params, 2**15)
            out[str(rc)][str(r)] = [float(l.mean()), float(l.std())]
    for r, v in thesis.items():
        a, b = out["1.67"][str(r)], out[str(1.0 / 0.6)][str(r)]
        print(f"  {r:<9} | {v:.4f} | {a[0]:+.4f} +/- {a[1]:.4f} | {b[0]:+.4f} +/- {b[1]:.4f}")
    rms = {rc: float(np.sqrt(np.mean([(out[rc][str(r)][0] - v) ** 2 for r, v in thesis.items()]))) for rc in out}
    print(f"  rms deviation from thesis: Rc=1.67 -> {rms['1.67']:.4f} bits, Rc=1/0.6 -> {rms[str(1.0/0.6)]:.4f} bits")
    report["rc"] = {"thesis": thesis, "runs": out, "rms": rms, "n_omega": len(omegas)}


CHECKS = {"rc": check_rc, "wolf": check_wolf, "dt": check_dt, "spectrum": check_spectrum, "errors": check_errors}
for name in args.checks.split(","):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.perf_counter()
    CHECKS[name]()
    print(f"    ({time.perf_counter() - t0:.1f}s)", flush=True)
    save()
print(f"\n-> {OUT}/verify.json")
