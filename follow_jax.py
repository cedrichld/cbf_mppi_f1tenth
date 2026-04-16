import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, vmap, grad, jacobian, value_and_grad
import numpy as np
import matplotlib.pyplot as plt
import time

from typing import Callable

# MPPI Application: Compile the Dynamics Step
# Bicycle model again
L = 0.33 # wheelbase
DT = 0.05 # timestep 
state_0 = jnp.array([0.0, 0.0, 0.0, 1.0])
control_0 = jnp.array([0.1, 0.0]) 

def dynamics(state, control, dt = DT):
        ''' 
        Kinematic bicycle mode -- single step
        state: (4,), control: (2,)
        '''
        # state and control at t
        x_t, y_t, theta_t, v_t = state
        steer_t, accel_t = control

        # Updates t+1
        x_update = x_t + v_t * jnp.cos(theta_t) * dt
        y_update = y_t + v_t * jnp.sin(theta_t) * dt
        theta_update = theta_t + v_t / L * jnp.tan(steer_t) * dt
        v_update = v_t + accel_t * dt

        return jnp.array([x_update, y_update, theta_update, v_update])

# Basics
def section1():
    # Previous stuff done on 24.04...

    # Test the MPPI -- one step from rest
    state_0 = jnp.array([0.0, 0.0, 0.0, 1.0])  # at origin, heading east, v=1 m/s
    control_0 = jnp.array([0.1, 0.0])           # slight left steer, no accel

    state_1 = dynamics(state_0, control_0)
    print(f"State t=0: {state_0}")
    print(f"State t=1: {state_1}")

    def stage_cost(state, reference):
        "Quadratic position tracking cost"
        pos_error = state[:2] - reference[:2]
        return jnp.sum(pos_error**2)

    # Test
    ref = jnp.array([1.0, 0.0, 0.0, 1.0])
    c = stage_cost(state_1, ref)
    print(f"Cost from {state_1[:2]} to ref {ref[:2]}: {c:.4f}")

    # Plot the cost landscape
    fig, ax = plt.subplots(figsize=(6, 5))

    # Grid of positions
    xs = jnp.linspace(-1, 2, 100)
    ys = jnp.linspace(-1, 1, 100)
    X, Y = jnp.meshgrid(xs, ys)

    # Evaluate the cost at each grid point (only position matters)
    costs = (X - ref[0])**2 + (Y - ref[1])**2

    contour = ax.contourf(X, Y, costs, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Cost')

    ax.plot(ref[0], ref[1], 'r*', markersize=15, label='reference')
    ax.plot(float(state_1[0]), float(state_1[1]), 'wo', markersize=10,
            markeredgecolor='k', label=f'state (cost={c:.2f})')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Quadratic Tracking Cost Landscape')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

# jax.jit: Compilation
def section2():
    # jax.jit - compilation
    # compyles python function via optimzied machine code w XLA(Accelerate Lin Alg)

    # ---------------------------------------------------------
    # Simple fn without jit
    def slow_fn(x):
        for _ in range(10):
            x = jnp.sin(x) + 0.01 * x
        return x

    fast_fn = jit(slow_fn)

    x = jnp.ones(100_000)

    # Warm it up (first call icnludes compilation time)
    _ = fast_fn(x)

    # Time comparison
    n_runs = 100

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = slow_fn(x).block_until_ready()
    t_slow = (time.perf_counter() - start) / n_runs * 1000

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = fast_fn(x).block_until_ready()
    t_fast = (time.perf_counter() - start) / n_runs * 1000

    print(f"Without jit: {t_slow:.3f} ms per call")
    print(f"With jit:    {t_fast:.3f} ms per call")
    print(f"Speedup:     {t_slow/t_fast:.1f}x")

    # Visualize the speedup
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(['jit(fn)', 'fn (no jit)'], [t_fast, t_slow],
                color=['#2ecc71', '#e74c3c'], edgecolor='white', height=0.5)
    ax.set_xlabel('Time per call (ms)')
    ax.set_title('jit Speedup -- sin/cos loop on 100k elements')
    for bar, t in zip(bars, [t_fast, t_slow]):
        ax.text(bar.get_width() + 0.02 * t_slow, bar.get_y() + bar.get_height()/2,
                f'{t:.3f} ms', va='center', fontsize=10)
    ax.set_xlim(0, t_slow * 1.3)
    plt.tight_layout()
    plt.show()
    # ---------------------------------------------------------

    # Tracing and control flow gotchas:

    def clamp_speed_bad(state: jnp.array):
        "clamp vel to [0, 5] - BROKEN under jit"
        v = state[3]
        if v > 5.0:
            v = 5.0
        elif v < 0.0:
            v = 0.0
        return state.at[3].set(v)

    state_0 = jnp.array([0.0, 0.0, 0.0, 1.0])
    try:
        jit(clamp_speed_bad)(state_0)
    except Exception as e:
        print(f"Error: {type(e).__name__}")
        print(f"  {str(e)[:121]}...")

    # Fix - use jax.lax.cond, jnp.where or jnp.clip
    def clamp_speed_good(state: jnp.array):
        "Clamp vel for jit"
        v = jnp.clip(state[3], 0.0, 5.0)
        return state.at[3].set(v)

    result = jit(clamp_speed_good)(state_0)
    print(f"\nFixed version: v clamped to {result[3]:.2f}")

    # For more complax branching use jnp.where:
    def relu_dynamics(state: jnp.array):
        "Zero out neg vels"
        v = jnp.where(state[3] > 0.0, state[3], 0.0)
        return state.at[3].set(v)

    result2 = jit(relu_dynamics)(state_0)
    print(f"jnp.where version: v clamped to {result2[3]:.2f}\n")



    ## For auto Jitting on call: @jit decorator
    @jit
    def sum_squares(x):
        return jnp.sum(x**2)

    # First call: compiles for shape(100,)

    # Each new shape recompiles - watch timings
    for n in [100, 200, 300, 100]:
        x = jnp.ones(n)
        start = time.perf_counter()
        _ = sum_squares(x).block_until_ready()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  shape ({n:3d},): {elapsed:6.2f} ms  {'(recompile!)' if elapsed > 1.0 else '(cached)'}")

    print("========================================")
    print("Running AFTER cached compilation")
    for n in [100, 200, 300, 100]:
        x = jnp.ones(n)
        start = time.perf_counter()
        _ = sum_squares(x).block_until_ready()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  shape ({n:3d},): {elapsed:6.2f} ms  {'(recompile!)' if elapsed > 1.0 else '(cached)'}")

    print("========================================")
    print("Pure numpy (no jit, no compilation step)")
    def sum_squares_np(x):
        return np.sum(x**2)

    for n in [100, 200, 300, 100]:
        x = np.ones(n)
        start = time.perf_counter()
        _ = sum_squares_np(x)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  shape ({n:3d},): {elapsed:6.2f} ms")


    # MPPI Application: Compile the Dynamics Step
    # Bicycle model again
    dynamics_jit = jit(dynamics)
    _ = dynamics_jit(state_0, control_0) # warmup
    n_runs = 100

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = dynamics(state_0, control_0).block_until_ready()
    t_nojit = (time.perf_counter() - start) / n_runs * 1e6

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = dynamics_jit(state_0, control_0).block_until_ready()
    t_jit = (time.perf_counter() - start) / n_runs * 1e6

    print(f"dynamics (no jit): {t_nojit:.1f} us")
    print(f"dynamics (jit):    {t_jit:.1f} us")
    print(f"Speedup:           {t_nojit/t_jit:.1f}x")

# MPPI APPLICATkION: Roll out a Trajectory
def rollout(state_init, controls):
    "Roll out bicycle model over control sequence. Returns (H, 4) traj"
    def step(state, u):
        next_state = dynamics(state, u)
        return next_state, next_state
    
    _, trajectory = lax.scan(step, state_init, controls)
    return trajectory

# Compiled loops with lax.scan 3
def section3():
    # Minimal example: cumulative sum via scan 
    def cumsum_step(running_total, x_t):
        new_total = running_total + x_t
        return new_total, new_total # (new carry, output)

    xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    final, cumsum = lax.scan(cumsum_step, 0.0, xs)

    print(f"Input:          {xs}")
    print(f"Cumulative sum: {cumsum}")
    print(f"Final carry:    {final}")

    # analogous to:
    def scan_analogy(fn_step: Callable, init, xs):
        carry = init
        outputs = []
        for x_t in xs:
            carry, out = fn_step(carry, x_t)
            outputs.append(out)
        return carry, jnp.stack(outputs)
    # final_carry, stacked_out = scan_analogy(fn_step, init, xs)
    # same as:
    # final_carry, stacked_out = lax.scan(fn_step, init, xs)

    ## MPPI!

    # Test: constant slight steering for 40 steps
    H = 40
    controls_test = jnp.tile(jnp.array([0.15, 0.5]), (H, 1))
    traj = rollout(state_0, controls_test)
    print(f"Trajectory shape: {traj.shape}")  # (40, 4)
    print(f"Final position:   x={traj[-1, 0]:.2f}, y={traj[-1, 1]:.2f}")

    def viz_sec3():
        # Visualize the trajectory
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # XY path
        ax = axes[0]
        full_traj = jnp.vstack([state_0[None, :], traj])
        ax.plot(full_traj[:, 0], full_traj[:, 1], 'o-', markersize=3, color='#3498db')
        ax.plot(full_traj[0, 0], full_traj[0, 1], 'gs', markersize=10, label='start')
        ax.plot(full_traj[-1, 0], full_traj[-1, 1], 'r*', markersize=12, label='end')
        # Draw heading arrows every 5 steps
        for i in range(0, len(full_traj), 5):
            dx = 0.15 * jnp.cos(full_traj[i, 2])
            dy = 0.15 * jnp.sin(full_traj[i, 2])
            ax.arrow(float(full_traj[i, 0]), float(full_traj[i, 1]),
                    float(dx), float(dy), head_width=0.04, color='#2c3e50', alpha=0.6)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Trajectory (lax.scan rollout)')
        ax.legend()
        ax.set_aspect('equal')

        # State evolution over time
        ax = axes[1]
        t = jnp.arange(H) * DT
        ax.plot(t, traj[:, 0], label='x')
        ax.plot(t, traj[:, 1], label='y')
        ax.plot(t, traj[:, 2], label='theta')
        ax.plot(t, traj[:, 3], label='v')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('State')
        ax.set_title('State Evolution over Horizon')
        ax.legend(ncol=4, loc='upper left')

        plt.tight_layout()
        plt.show()

    viz_sec3()

    def rollout_fori(state_init: jnp.array, controls: jnp.array):
        "Rollout via fori_loop - manual trajectory storage"
        H = controls.shape[0]
        trajectory = jnp.zeros((H, 4))

        def body(t, carry):
            state, traj = carry
            next_state = dynamics(state, controls[t])
            traj = traj.at[t].set(next_state)
            return (next_state, traj)
        
        _, trajectory = lax.fori_loop(
            0, H, body, state_init)
        return trajectory

    def rollout_fori_final(state_init: jnp.array, controls: jnp.array):
        "Only returns the final state - no traj stored"
        def body(t, state):
            return dynamics(state, controls[t])
        return lax.fori_loop(
            0, controls.shape[0], body, state_init)

# jax.vmap: automatic vectorization
def section4():

    # Simple example: norm of a single vector
    def norm(x):
        return jnp.sqrt(jnp.sum(x**2))

    # batch it
    norm_batched = vmap(norm)

    key = jax.random.PRNGKey(42)
    points = jax.random.normal(key, (6, 3))

    # Without vmap -- you need a loop
    norms_loop = jnp.array([norm(points[i]) for i in range(len(points))])

    # With vmap --one line
    norms_vmap = norm_batched(points)

    print(f"Points shape: {points.shape}")
    print(f"Norms (loop): {norms_loop}")
    print(f"Norms (vmap): {norms_vmap}")
    print(f"Match: {jnp.allclose(norms_loop, norms_vmap)}")

    # distance from N points to a SINGLE reference
    def dist(point, ref):
        return jnp.linalg.norm(point - ref)

    # Batch over points (Axis 0), same ref for all (None)
    batched_dist = vmap(dist, in_axes=(0, None))

    ref = jnp.array([0.0, 0.0, 0.0])
    dists = batched_dist(points, ref)
    print(f"Distances to origin: {dists}")

    def rollout_with_noise(state, nominal_controls, noise):
        "Single rollout with perturbed controls."
        perturbed = nominal_controls + noise
        return rollout(state, perturbed)

    # vmap over noise (axis 0), broadcast state and nominal_controls
    batched_rollout = vmap(
        rollout_with_noise, in_axes=(None, None, 0))

    # Setup: K = 1024 Rollouts, H = 40 horizon
    K = 1024
    H = 40
    key = jax.random.PRNGKey(99)
    nominal = jnp.zeros((H, 2)) # start form 0 nominal
    noise_samples = jax.random.normal(key, (K, H, 2)) \
        * jnp.array([0.15, 0.8])

    # Run all K rollouts in parallel
    all_trajectories = batched_rollout(
        state_0, nominal, noise_samples)

    print(f"Trajectories shape: {all_trajectories.shape}") # (K, H, 4)

    def viz_s4():
        # Visualize all sampled trajectories
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all K trajectories with low alpha
        for k in range(K):
            ax.plot(all_trajectories[k, :, 0], all_trajectories[k, :, 1],
                    alpha=0.04, color='#3498db', linewidth=0.8)

        # Highlight a few random ones
        for k in [0, 50, 100, 200, 400]:
            ax.plot(all_trajectories[k, :, 0], all_trajectories[k, :, 1],
                    alpha=0.8, linewidth=1.5, label=f'sample {k}')

        ax.plot(state_0[0], state_0[1], 'gs', markersize=12, label='start', zorder=5)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'MPPI: {K} Sampled Trajectories via vmap')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    viz_s4()

# composing transforms: FULL MPPI PIPELINE
def section5():
    # Params
    K = 2**17 # number of samples: 2**15 = 32,765
    H = 40 # horizon length
    SIGMA_STEER = 0.15
    SIGMA_ACCEL = 0.8
    LAMBDA = 0.01 # temperature param

    # Reference: drive to a goal point
    goal = jnp.array([3.0, 1.5])

    def stage_cost_mppi(state, goal):
        "Position tracking cost for MPPI"
        return jnp.sum((state[:2] - goal)**2)

    def terminal_cost(state, goal):
        "Extra penalty at the end of the horizon"
        return 5.0 * jnp.sum((state[:2] - goal)**2)

    # -- Single rollout cost 
    def rollout_cost(state_init, controls, goal):
        "Cost of a single rollout"
        traj = rollout(state_init, controls)
        stage_costs = vmap(stage_cost_mppi, in_axes=(0, None))(traj, goal)
        return jnp.sum(stage_costs) + terminal_cost(traj[-1], goal)

    # -- Full MPPI Step (compiled)
    @jit
    def mppi_step(state, nominal_controls, key, goal):
        "One MPPI update -- fully compiled"
        # 1. Sample noise
        noise = jax.random.normal(key, (K, H, 2)) * jnp.array([SIGMA_STEER, SIGMA_ACCEL])

        # 2 and 3 Cost of each perturbed rollout
        def sample_cost(eps):
            return rollout_cost(state, nominal_controls + eps, goal)
        
        costs = vmap(sample_cost)(noise) # (K,)

        # 4 Importance weights (softmax)
        weights = jax.nn.softmax(-costs / LAMBDA)

        #5 weighted update
        update = jnp.sum(weights[:, None, None] * noise, axis=0) # (H, 2)
        new_nominal = nominal_controls + update

        return new_nominal, costs

    # --- Run MPPI in a receding-horizon loop ---
    state = jnp.array([0.0, 0.0, 0.0, 1.0])  # start: origin, heading east, v=1
    nominal = jnp.zeros((H, 2))
    key = jax.random.PRNGKey(0)

    n_steps = 60
    trajectory_history = [state]
    control_history = []
    cost_history = []

    for t in range(n_steps):
        key, subkey = jax.random.split(key)

        # MPPI update
        nominal, costs = mppi_step(state, nominal, subkey, goal)
        cost_history.append(float(jnp.min(costs)))

        # apply first control
        u = nominal[0]
        control_history.append(u)
        state = dynamics(state, u)
        trajectory_history.append(state)

        # shift nominal controls (warm start)
        nominal = jnp.concatenate([nominal[1:], nominal[-1:]], axis=0)

    trajectory_history = jnp.stack(trajectory_history)
    control_history = jnp.stack(control_history)
    print(f"Final position: ({trajectory_history[-1, 0]:.2f}, {trajectory_history[-1, 1]:.2f})")
    print(f"Goal:           ({goal[0]:.2f}, {goal[1]:.2f})")

    def visualize_mppi():
        # --- Visualize MPPI results ---
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. XY Trajectory with sampled rollouts from the final state
        ax = axes[0]
        noise_viz = jax.random.normal(jax.random.PRNGKey(123), (200, H, 2)) * jnp.array([SIGMA_STEER, SIGMA_ACCEL])
        for k in range(200):
            viz_traj = rollout(trajectory_history[-1], nominal + noise_viz[k])
            ax.plot(viz_traj[:, 0], viz_traj[:, 1], alpha=0.05, color="#3498db", linewidth=0.5)

        ax.plot(trajectory_history[:, 0], trajectory_history[:, 1],
                'k-', linewidth=2.5, label='executed path', zorder=4)
        ax.plot(trajectory_history[0, 0], trajectory_history[0, 1], 'gs', markersize=10, label='start', zorder=5)
        ax.plot(goal[0], goal[1], 'r*', markersize=15, label='goal', zorder=5)
        # heading arrows
        for i in range(0, len(trajectory_history), 16):
            s = trajectory_history[i]
            dx, dy = 0.12 * jnp.cos(s[2]), 0.12 * jnp.sin(s[2])
            ax.arrow(float(s[0]), float(s[1]), float(dx), float(dy),
                    head_width=0.03, color='#2c3e50', alpha=0.7)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('MPPI Closed-Loop Trajectory')
        ax.legend(fontsize=9)
        ax.set_aspect('equal')

        # 2. Controls over time
        ax = axes[1]
        t_ax = jnp.arange(n_steps) * DT
        ax.plot(t_ax, control_history[:, 0], label='steering', color='#e67e22')
        ax.plot(t_ax, control_history[:, 1], label='accel', color='#8e44ad')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Control')
        ax.set_title('Control Inputs')
        ax.legend()

        # 3. Min cost over iterations
        ax = axes[2]
        ax.plot(cost_history, color='#27ae60')
        ax.set_xlabel('MPPI iteration')
        ax.set_ylabel('Min sample cost')
        ax.set_title('Cost')

        plt.tight_layout()
        plt.show()
    visualize_mppi()

    # Time the full MPPI step
    key = jax.random.PRNGKey(0)
    _ = mppi_step(state_0, jnp.zeros((H, 2)), key, goal)  # warm up

    n_runs = 100
    # n_runs = 1 # Change n_runs when running with 10024 samples

    start = time.perf_counter()
    for i in range(n_runs):
        key, subkey = jax.random.split(key)
        result = mppi_step(state_0, jnp.zeros((H, 2)), subkey, goal)
        result[0].block_until_ready()
    t_mppi = (time.perf_counter() - start) / n_runs * 1000

    print(f"Full MPPI step ({K} samples x {H} horizon): {t_mppi:.2f} ms")
    print(f"Achievable rate: {1000/t_mppi:.0f} Hz")
    print(f"F1TENTH target:  >= 20 Hz {'pass' if 1000/t_mppi >= 20 else 'FAIL'}")

section5()

# jax.grad
def section6():
    # grad: scalar function -> gradient function
    def f(x):
        return jnp.sum(x**3)

    df = grad(f)
    x = jnp.array([1.0, 2.0, 3.0])
    print(f"f(x)     = {f(x)}")
    print(f"grad(f)  = {df(x)}")           # [3x^2] = [3, 12, 27]
    print(f"expected = {3 * x**2}")

    
    # jacobian: for vector -> vector functions
    def g(x):
        return jnp.array([x[0]**2 + x[1], jnp.sin(x[0]) * x[1]])

    J = jacobian(g)
    x0 = jnp.array([1.0, 2.0])
    print(f"g(x0) = {g(x0)}")
    print(f"Jacobian:\n{J(x0)}")


    # Beyond MPPI: Free Jacobians for MPC linearization
    # A = d(dynamics)/d(state), B = d(dynamics)/d(control)
    A_fn = jit(jacobian(dynamics, argnums=0))
    B_fn = jit(jacobian(dynamics, argnums=1))

    # Evaluate at a specific operating point
    state_op = jnp.array([1.0, 0.5, 0.3, 2.0])
    control_op = jnp.array([0.1, 0.0])

    A = A_fn(state_op, control_op)
    B = B_fn(state_op, control_op)

    print("A (df/dx) -- 4x4 state Jacobian:")
    print(A)
    print(f"\nB (df/du) -- 4x2 input Jacobian:")
    print(B)


    # Visualize the Jacobian structure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    state_labels = ['x', 'y', 'theta', 'v']
    control_labels = ['delta', 'a']

    ax = axes[0]
    im = ax.imshow(np.array(A), cmap='RdBu_r', aspect='equal', vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(4)); ax.set_xticklabels(state_labels)
    ax.set_yticks(range(4)); ax.set_yticklabels(state_labels)
    ax.set_title('A = df/dx (state Jacobian)')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{A[i,j]:.3f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(np.array(B), cmap='RdBu_r', aspect='equal', vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(2)); ax.set_xticklabels(control_labels)
    ax.set_yticks(range(4)); ax.set_yticklabels(state_labels)
    ax.set_title('B = df/du (input Jacobian)')
    for i in range(4):
        for j in range(2):
            ax.text(j, i, f'{B[i,j]:.3f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Automatic Linearization of Bicycle Model -- No Hand Derivation!', y=1.02)
    plt.tight_layout()
    plt.show()