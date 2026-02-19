import jax
import jax.numpy as jnp
from jax import jit, vmap, jacfwd
import time
import matplotlib.pyplot as plt

# Import the base physics equation
from sprott_solver import sprott_equations

# --- 1. The Tangent Dynamics (Variational Equations) ---

def augmented_equations(aug_state, t, params):
    """
    Evolves both the physical state and the tangent vector.
    State vector is size 6: [x, y, z, wx, wy, wz]
    """
    state = aug_state[:3]
    tangent_vec = aug_state[3:]
    
    # 1. Physical flow
    d_state = sprott_equations(state, t, params)
    
    # 2. Tangent flow: d(w)/dt = J * w
    # Use JAX to compute the Jacobian of sprott_equations w.r.t state (arg 0)
    # jacfwd returns a 3x3 matrix J(x, y, z)
    J = jacfwd(sprott_equations, argnums=0)(state, t, params)
    
    # Matrix multiplication: J @ w
    d_tangent = jnp.dot(J, tangent_vec)
    
    return jnp.concatenate([d_state, d_tangent])

# --- 2. The RK4 Integrator for Augmented System ---

@jit
def rk4_step_aug(aug_state, t, dt, params):
    """
    Standard RK4 but for the 6D augmented system.
    """
    k1 = augmented_equations(aug_state, t, params)
    k2 = augmented_equations(aug_state + 0.5 * dt * k1, t + 0.5 * dt, params)
    k3 = augmented_equations(aug_state + 0.5 * dt * k2, t + 0.5 * dt, params)
    k4 = augmented_equations(aug_state + dt * k3, t + dt, params)
    
    return aug_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- 3. The LLE Calculation Loop (Benettin's Algorithm) ---

@jit
def calculate_lle(initial_state, params, total_steps, dt, norm_interval=10):
    """
    Calculates the Largest Lyapunov Exponent (LLE) using the Jacobian method.
    Renormalizes the tangent vector every `norm_interval` steps to prevent overflow.
    """
    
    # Initial setup
    # Start with a random unit tangent vector
    key = jax.random.PRNGKey(42) # Deterministic seed
    w0 = jax.random.normal(key, (3,))
    w0 = w0 / jnp.linalg.norm(w0)
    
    # Augmented state: [x0, y0, z0, wx0, wy0, wz0]
    aug_state = jnp.concatenate([initial_state, w0])
    
    # Loop carry: (current_aug_state, current_t, cumulative_log_norm)
    initial_carry = (aug_state, 0.0, 0.0)

    # We process in chunks of `norm_interval` steps to handle renormalization efficiently
    num_chunks = total_steps // norm_interval

    def chunk_step(carry, _):
        curr_aug, curr_t, cum_log_norm = carry
        
        # 1. Evolve for norm_interval steps WITHOUT renormalization
        # Use a localized scan for the inner loop
        def inner_step(inner_carry, _):
            s, t = inner_carry
            ns = rk4_step_aug(s, t, dt, params)
            return (ns, t + dt), None
            
        (next_aug, next_t), _ = jax.lax.scan(inner_step, (curr_aug, curr_t), None, length=norm_interval)
        
        # 2. Renormalize the tangent vector
        state = next_aug[:3]
        tangent = next_aug[3:]
        
        norm = jnp.linalg.norm(tangent)
        tangent_normalized = tangent / norm
        
        # 3. Accumulate the log of the expansion rate
        new_cum_log = cum_log_norm + jnp.log(norm)
        
        # Re-pack augmented state
        next_aug_normalized = jnp.concatenate([state, tangent_normalized])
        
        return (next_aug_normalized, next_t, new_cum_log), new_cum_log

    # Run the chunks
    final_carry, log_history = jax.lax.scan(chunk_step, initial_carry, None, length=num_chunks)
    
    # Calculate Final LLE
    # LLE = (Sum of log norms) / Total Time
    final_log_sum = final_carry[2]
    total_time = total_steps * dt
    lle = final_log_sum / total_time
    
    return lle, log_history

# --- 4. Vectorization ---

# Calculate LLE for a batch of parameters
batch_lle = vmap(calculate_lle, in_axes=(None, 0, None, None, None))

# --- Main Execution ---

if __name__ == "__main__":
    # Configuration
    DT = 0.05 
    # Use fewer steps for quick demo, but enough to converge
    STEPS = 50000 
    R_C = 1.67
    
    # 1. Define Parameter Sweep (High Resolution for 1D)
    # Scanning Omega from 0 to 2.75 (The fluctuating region from your thesis)
    resolution = 200
    omegas = jnp.linspace(0, 2.75, resolution)
    R_amps = jnp.full_like(omegas, 0.15) # R = 0.15 curve
    R_base = jnp.full_like(omegas, R_C)
    
    batch_params = jnp.stack([R_base, R_amps, omegas], axis=1)
    initial_state = jnp.array([0.05, 0.0, 0.0])

    print(f"--- Jacobian LLE Solver (JAX) ---")
    print(f"Scanning {resolution} parameters... (Replacing Wolf's Algo)")
    
    # 2. Run
    start_time = time.time()
    
    # JIT Compile & Run
    lles, _ = batch_lle(initial_state, batch_params, STEPS, DT, 10).block_until_ready()
    
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.4f}s")
    
    # 3. Plotting (Recreating Thesis Figure 5.1)
    # If MODE is headless, we might skip plt.show() but here we assume local run
    plt.figure(figsize=(10, 6))
    plt.plot(omegas, lles, linewidth=0.8, color='black')
    plt.title("Largest Lyapunov Exponent vs Omega (JAX Jacobian Method)")
    plt.xlabel("Omega (radians)")
    plt.ylabel("LLE (bits/s or nats/s)")
    plt.axhline(0, color='red', linestyle='--', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    print("Plotting results...")
    plt.show()