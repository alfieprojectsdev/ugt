import os
import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

# Lyapunov exponents accumulate a log-norm over thousands of renormalizations,
# so float32 (JAX's default) is not enough precision. Must be set before any
# array is created.
jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402  (backend must be chosen before pyplot import)

# This project is expected to run headless over SSH, where plt.show() renders
# to nothing at all. Pick a file-writing backend when there is no display.
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

# --- Configuration ---
MODE = "PP"  # Options: "RT" (Real-Time Animation) or "PP" (Post-Processing Plot)
FIGURE_DIR = "figures"


def save_figure(filename):
    """Write the current figure to FIGURE_DIR; show it only if a display exists."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Figure written to {path}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return path


def is_headless():
    return matplotlib.get_backend().lower() == "agg"

# --- 1. The Physics Model (The "Logic") ---

@jit
def sprott_equations(state, t, params):
    """
    The Modified Sprott Circuit Equations.
    Corresponds to Eq 4.1 in the thesis: d3x/dt3 = -A(d2x/dt2) - (dx/dt) + |x| - 1
    Where A = 1 / (Rc + R*sin(omega*t))
    
    State vector: [x, dx/dt, d2x/dt2] -> [x, y, z]
    """
    x, y, z = state
    R_c, R_amp, omega = params
    
    # Calculate time-dependent A
    # A = 1.0 / (R_c + R_amp * jnp.sin(omega * t)) 
    # Note: Added epsilon for numerical stability if R_c + R_sin goes to 0
    denominator = R_c + R_amp * jnp.sin(omega * t)
    A = 1.0 / (denominator + 1e-9)

    # The system of 1st order ODEs
    # dx/dt = y
    # dy/dt = z
    # dz/dt = -A*z - y + |x| - 1
    
    dx_dt = y
    dy_dt = z
    dz_dt = -A * z - y + jnp.abs(x) - 1.0
    
    return jnp.array([dx_dt, dy_dt, dz_dt])

# --- 2. The Numerical Integrator (Runge-Kutta 4) ---

@jit
def rk4_step(state, t, dt, params):
    """
    Performs a single RK4 step.
    JIT-compiled for speed (replacing the Fortran loop).
    """
    k1 = sprott_equations(state, t, params)
    k2 = sprott_equations(state + 0.5 * dt * k1, t + 0.5 * dt, params)
    k3 = sprott_equations(state + 0.5 * dt * k2, t + 0.5 * dt, params)
    k4 = sprott_equations(state + dt * k3, t + dt, params)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- 3. The Simulation Loop (The "Time Evolution") ---

@partial(jit, static_argnums=(2,))
def simulate_trajectory(initial_state, params, time_steps, dt):
    """
    Simulates a full trajectory for a SINGLE set of parameters.
    Returns the full time series of states.

    `time_steps` is static: lax.scan needs a concrete length, and a plain @jit
    would trace it into a tracer. Each distinct value triggers a recompile.
    """
    def step_fn(carry, _):
        curr_state, curr_t = carry
        next_state = rk4_step(curr_state, curr_t, dt, params)
        return (next_state, curr_t + dt), next_state

    # scan is the JAX equivalent of a fast loop that carries state
    _, trajectory = jax.lax.scan(step_fn, (initial_state, 0.0), None, length=time_steps)
    return trajectory

# --- 4. The Vectorization (The "Completeness" Engine) ---

# This magic line creates a function that runs simulate_trajectory
# on a BATCH of parameters simultaneously.
# in_axes: (None, 0, None, None) means:
# - initial_state: same for all (None) or vectorized? Let's keep it same for now.
# - params: VECTORIZED (0) -> Each row of params gets its own simulation.
# - time_steps: same for all (None)
# - dt: same for all (None)
batch_simulate = vmap(simulate_trajectory, in_axes=(None, 0, None, None))

# --- Main Execution Block (Verification) ---

if __name__ == "__main__":
    # 1. Configuration
    DT = 0.05  # Matches Thesis Chapter 4
    STEPS = 32768 # 2^15, Matches Thesis Chapter 4
    R_C = 1.67 # Thesis value for chaos baseline
    
    # 2. Define Parameter Space (The "Sweep")
    # Let's create a batch of 10 different Omega values to test vectorization
    omegas = jnp.linspace(0, 2.75, 10) 
    R_amps = jnp.full_like(omegas, 0.15) # R = 0.15 (Thesis Chapter 5)
    R_base = jnp.full_like(omegas, R_C)
    
    # Stack into a (10, 3) matrix of parameters
    batch_params = jnp.stack([R_base, R_amps, omegas], axis=1)
    
    # Initial State (x=0, y=0, z=0 start)
    initial_state = jnp.array([0.05, 0.0, 0.0]) # Small perturbation to start

    print(f"--- 2026 Sprott Circuit Solver (JAX/Numba) ---")
    print(f"Mode: {MODE}")
    
    if MODE == "PP":
        print(f"Simulating {len(omegas)} parallel trajectories with {STEPS} steps each...")
        
        # 3. Run Simulation (with Warmup for JIT)
        start_time = time.time()
        # The first run compiles the graph (Might take a split second)
        _ = batch_simulate(initial_state, batch_params, STEPS, DT).block_until_ready()
        compile_time = time.time() - start_time
        print(f"JIT Compile Time: {compile_time:.4f}s")
        
        # The second run is the actual speed
        start_time = time.time()
        results = batch_simulate(initial_state, batch_params, STEPS, DT).block_until_ready()
        run_time = time.time() - start_time
        
        print(f"Execution Time: {run_time:.4f}s")
        print(f"Total Steps Computed: {len(omegas) * STEPS}")
        print(f"Steps per Second: {(len(omegas) * STEPS) / run_time:.2e}")
        
        # 4. Verify Output Shape
        # Should be (Batch_Size, Time_Steps, 3_State_Vars)
        print(f"Output Shape: {results.shape}") 
        
        # Basic Sanity Check: Check if it blew up (NaNs)
        if jnp.isnan(results).any():
            print("WARNING: NaNs detected! Check stiffness or dt.")
        else:
            print("Simulation stable.")
            
        # 5. Plotting (Post-Processing)
        print("Plotting results...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the first trajectory in the batch as an example
        # results[0, :, 0] -> 1st batch item, all time steps, x component
        x_vals = results[0, :, 0]
        y_vals = results[0, :, 1]
        z_vals = results[0, :, 2] # Usually derivative, for phase space
        
        # Actually, for phase space, usually we plot x vs dx/dt vs d2x/dt2
        # which corresponds to indices 0, 1, 2
        ax.plot(x_vals, y_vals, z_vals, lw=0.5, alpha=0.8)
        
        ax.set_xlabel("x")
        ax.set_ylabel("dx/dt (y)")
        ax.set_zlabel("d2x/dt2 (z)")
        ax.set_title(f"Sprott Circuit Phase Space (PP Mode)\nR={R_amps[0]}, Omega={omegas[0]:.2f}")
        save_figure("phase_space_pp.png")

    elif MODE == "RT":
        if is_headless():
            raise SystemExit(
                "MODE='RT' needs an interactive display; this session has none.\n"
                "Set MODE='PP' to render a static phase-space plot to "
                f"{FIGURE_DIR}/ instead."
            )
        print("Initializing Real-Time Animation...")
        # For RT, we usually simulate chunks or step-by-step.
        # Since JAX is fast, we can simulate a chunk, plot it, then simulate next.
        # Or, simulate whole thing and animate the reveal. 
        # Simulating step-by-step in Python loop kills JAX performance.
        # Better strategy: Simulate ALL first (since it takes <1s), then animate the array.
        
        # Run simulation for just ONE parameter set for clarity in animation
        single_params = batch_params[0] # Pick the first one
        # Reshape to (1, 3) to fit batch_simulate expectations
        single_batch_params = jnp.expand_dims(single_params, axis=0) 
        
        start_time = time.time()
        results = batch_simulate(initial_state, single_batch_params, STEPS, DT).block_until_ready()
        print(f"Simulation finished in {time.time() - start_time:.4f}s. Starting Animation...")
        
        # Extract data
        data = results[0] # (STEPS, 3)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize empty line
        line, = ax.plot([], [], [], lw=0.5)
        
        # Set limits based on data
        ax.set_xlim(jnp.min(data[:,0]), jnp.max(data[:,0]))
        ax.set_ylim(jnp.min(data[:,1]), jnp.max(data[:,1]))
        ax.set_zlim(jnp.min(data[:,2]), jnp.max(data[:,2]))
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Real-Time Phase Space Evolution\nR={single_params[1]}, Omega={single_params[2]:.2f}")

        # Animation Speed: Skip frames to make it look faster
        SKIP = 50 
        
        def update(frame):
            # Frame goes from 0 to STEPS/SKIP
            end_idx = frame * SKIP
            # Plot up to current index
            line.set_data(data[:end_idx, 0], data[:end_idx, 1])
            line.set_3d_properties(data[:end_idx, 2])
            return line,

        ani = FuncAnimation(fig, update, frames=int(STEPS/SKIP), interval=1, blit=False)
        plt.show()

    print("\nReady for Terraform deployment.")