# CBF Notes for the MPPI Final Project

This document is a working design note for adding obstacle awareness and
Control Barrier Function ideas to the current JAX MPPI controller.

The goal is not to rewrite the controller from scratch. The most practical
first extension is:

```text
raceline MPPI + fixed-size obstacle set + CBF-style obstacle cost
```

Then, if that works, we can add a final-command CBF-QP safety shield.

References that motivated this plan:

- `shaoanlu/mppi_cbf`: https://github.com/shaoanlu/mppi_cbf
- Bicycle MPPI/CBF notebook: https://github.com/shaoanlu/mppi_cbf/blob/main/bicycle_mppi_cbf_shielding.ipynb
- Shield-MPPI paper: https://arxiv.org/abs/2302.11719
- Ames CBF survey: https://arxiv.org/abs/1903.11199

## Why Add CBF Ideas?

Our current MPPI is good at tracking a reference line. Its basic loop is:

```text
sample control sequences
roll out vehicle dynamics
score each trajectory against the reference
average good controls
publish steering and speed
```

If the reward only measures reference tracking, then a longer horizon only
makes the controller look farther along the raceline. It does not make the car
aware of obstacles, other cars, blocked regions, or free space unless those
things appear in the reward or in a safety filter.

CBFs give us a mathematical way to say:

```text
stay inside a safe set
```

For racing, the safe set could mean:

- stay away from obstacle points from LiDAR
- stay inside track boundaries
- avoid opponent cars
- avoid entering regions where braking distance is too short

For a final race project, CBFs are useful because they give a clean story:

```text
MPPI handles performance. CBF terms handle safety.
```

## What Is A CBF?

Define a scalar function:

```math
h(x)
```

where `x` is the state. The safe set is:

```math
\mathcal{C} = \{x : h(x) \ge 0\}
```

Interpretation:

```text
h(x) > 0  safe
h(x) = 0  safety boundary
h(x) < 0  unsafe
```

For a circular obstacle centered at `o = (o_x, o_y)` with safety radius
`r_safe`, a common barrier is:

```math
h(x) = \|p(x) - o\|^2 - r_{safe}^2
```

where:

```math
p(x) = [x_{pos}, y_{pos}]^T
```

This says the car is safe when its position is outside the obstacle bubble.

## Continuous-Time CBF Condition

For continuous dynamics:

```math
\dot{x} = f(x) + g(x)u
```

a first-order CBF condition is:

```math
\dot{h}(x, u) \ge -\alpha h(x)
```

where:

```math
\alpha > 0
```

Equivalently, using Lie derivatives:

```math
L_f h(x) + L_g h(x)u + \alpha h(x) \ge 0
```

Intuition:

- Far from the obstacle, `h` is large, so the controller has more freedom.
- Near the boundary, `h` is small, so the controller must avoid decreasing
  `h` too quickly.
- If `h = 0`, the condition requires `\dot{h} >= 0`, meaning do not cross into
  the unsafe set.

This is the usual CBF-QP story:

```math
\begin{aligned}
u^* = \arg\min_u \quad & \|u - u_{nom}\|_R^2 \\
\text{s.t.} \quad & L_f h(x) + L_g h(x)u + \alpha h(x) \ge 0 \\
& u_{min} \le u \le u_{max}
\end{aligned}
```

Here `u_nom` would be the command from MPPI, and the QP minimally modifies it
to satisfy safety constraints.

## Why The Simple CBF-QP Math Is Not The First Step

For a car, obstacle distance usually does not depend directly on steering or
acceleration in the first derivative.

Example kinematic bicycle model from the notebook:

```math
q = (x, y, \theta, v)
```

with control:

```math
u = (\delta, a)
```

and dynamics:

```math
\begin{aligned}
\dot{x} &= v\cos(\theta) \\
\dot{y} &= v\sin(\theta) \\
\dot{\theta} &= \frac{v\tan(\delta)}{L} \\
\dot{v} &= a
\end{aligned}
```

For a circle centered at `(x_o, y_o)`:

```math
h(q) = (x - x_o)^2 + (y - y_o)^2 - r^2
```

The first derivative is:

```math
\dot{h}
= 2(x - x_o)v\cos(\theta)
+ 2(y - y_o)v\sin(\theta)
```

This does not contain `\delta` or `a` directly. To expose the controls, the
notebook differentiates again:

```math
\ddot{h}
= 2v^2
+ \frac{-2(x - x_o)v^2\sin(\theta) + 2(y - y_o)v^2\cos(\theta)}{L}
   \tan(\delta)
+ [2(x - x_o)\cos(\theta) + 2(y - y_o)\sin(\theta)]a
```

Now steering and acceleration appear.

To make this QP-friendly, the notebook linearizes:

```math
\tan(\delta)
\approx
\tan(\delta_0) + (1 + \tan^2(\delta_0))(\delta - \delta_0)
```

Then it uses a higher-order CBF style constraint such as:

```math
\ddot{h} + c_1\dot{h} + c_0 h \ge 0
```

The notebook uses a closely related form for a CBF-QP safety filter. This math
is useful, but it should be treated as a later extension for our code because:

- our dynamic single-track model is more complicated than the notebook's toy
  bicycle model
- our MPPI internal controls are steering rate and acceleration, while the ROS
  command interface publishes steering angle and speed
- solving a QP every control tick is possible, but it is another moving part
  on the Jetson

## Discrete CBFs Fit MPPI Better

MPPI already simulates the future states:

```math
x_0, x_1, \dots, x_H
```

So we do not need analytic derivatives at first.

A discrete CBF condition can be written as:

```math
h(x_{k+1}) - h(x_k) \ge -\gamma h(x_k)
```

or equivalently:

```math
h(x_{k+1}) \ge (1 - \gamma) h(x_k)
```

where:

```math
0 < \gamma \le 1
```

Interpretation:

- `gamma = 1.0`: only require `h(x_{k+1}) >= 0`
- smaller `gamma`: require more conservative progress near the obstacle

For MPPI, turn violations into a penalty:

```math
v_{cbf,k} =
\max\left(0, (1 - \gamma)h(x_k) - h(x_{k+1})\right)
```

Then:

```math
J_{cbf} = w_{cbf}\sum_{k=0}^{H-1} v_{cbf,k}^2
```

Since our implementation is reward-based, we subtract this penalty from reward:

```math
R_{new}
= R_{track}
- J_{obs}
- J_{cbf}
- J_{risk}
```

This is the math I would use first from the notebook: not the full QP shield,
but the discrete CBF cost idea inside MPPI.

## Basic Obstacle Cost

Before adding the discrete CBF progress term, add a simple obstacle penalty.

For each predicted car position `p_k` and obstacle point `o_i`:

```math
h_{i,k} = \|p_k - o_i\|^2 - r_{safe}^2
```

Penalty:

```math
J_{obs}
= w_{obs}\sum_{k=0}^{H}\sum_{i=1}^{M}
\max(0, -h_{i,k})^2
```

This says: if a rollout enters the obstacle bubble, punish it.

This is easier than a true CBF and already gives MPPI obstacle awareness.

## Risk-Aware Speed Cost

Obstacle avoidance is not just steering away. At speed, the car may need to
slow down before the collision becomes geometrically unavoidable.

A useful extra term is:

```math
J_{risk}
= w_{risk}\sum_k v_k^2 \exp\left(-\frac{d_{min,k}}{\sigma}\right)
```

where:

```math
d_{min,k} = \min_i \|p_k - o_i\|
```

This penalizes high speed near obstacles. It is not a formal CBF, but it is
practical and easy to tune.

## How This Maps To Our MPPI Code

Current reward is in `infer_env.py`:

```text
reward_fn_xy(...)
reward_fn_sey(...)
```

The extension should add obstacle terms there or in a helper called by those
reward functions.

Current JAX MPPI path:

```text
mppi_node.py
  receives odom
  builds state and reference
  calls self.mppi.update(...)

mppi_tracking.py
  samples controls
  rolls out states
  computes reward
  computes MPPI weights

infer_env.py
  steps dynamics
  computes tracking reward
```

The future extension should pass a fixed-size obstacle array into MPPI:

```math
O \in \mathbb{R}^{M \times 2}
```

and optionally a mask:

```math
m \in \{0, 1\}^{M}
```

Fixed shape matters because JAX recompiles when shapes change.

## Obstacle Input Plan

First source: `LaserScan`.

Pipeline:

```text
/scan
  -> convert ranges to points in base_link frame
  -> optionally keep only points in front of the car
  -> transform to map frame or keep everything in car frame
  -> choose closest M points
  -> pad to fixed shape
  -> pass to MPPI
```

Recommended first shape:

```yaml
cbf_obstacle_count: 32
```

Do not make this dynamic at runtime unless we intentionally rebuild the JAX
function or always keep the array padded to the same size.

## Proposed ROS Params

Startup-only or restart-recommended:

```yaml
cbf_obstacle_count: 32
```

Live-tunable:

```yaml
cbf_enabled: false
cbf_cost_enabled: true
cbf_shield_enabled: false

cbf_safe_radius: 0.45
cbf_gamma: 0.5
cbf_weight: 10.0
obstacle_weight: 20.0
risk_weight: 1.0
risk_distance_scale: 0.75

cbf_max_obstacle_range: 6.0
cbf_front_angle_min: -1.8
cbf_front_angle_max: 1.8
```

Suggested interpretation:

- `cbf_enabled`: master switch
- `cbf_cost_enabled`: add obstacle/CBF costs inside MPPI
- `cbf_shield_enabled`: later, enable final-action QP shield
- `cbf_safe_radius`: car plus obstacle buffer
- `cbf_gamma`: discrete CBF conservativeness
- `cbf_weight`: penalty for violating the discrete CBF condition
- `obstacle_weight`: penalty for entering obstacle bubbles
- `risk_weight`: slow down near obstacles
- `risk_distance_scale`: how quickly risk decays with distance

## Tuning Intuition

If the car ignores obstacles:

```yaml
obstacle_weight: increase
cbf_weight: increase
cbf_safe_radius: increase
```

If the car is too scared:

```yaml
obstacle_weight: decrease
cbf_weight: decrease
cbf_safe_radius: decrease
cbf_gamma: increase
```

If it avoids obstacles but stays too fast near them:

```yaml
risk_weight: increase
risk_distance_scale: increase
```

If rollouts cannot find a safe path:

```yaml
control_sample_std_steer: increase
control_sample_std_accel: increase
ref_vel: decrease
n_steps: increase carefully
```

## Jetson Feasibility

The MPPI-cost version should be feasible on the Jetson if the obstacle set is
small and fixed-shape.

Example work size:

```text
1024 samples * 10 horizon steps * 32 obstacle points
= 327,680 distance checks
```

That is reasonable for JAX if vectorized.

Avoid:

- Python loops over samples
- dynamic obstacle array shapes
- CVXPY inside the MPPI sample loop
- QP per sample

Maybe feasible later:

- one small OSQP-based safety filter after MPPI

Best first target:

```text
JAX-vectorized obstacle and discrete CBF cost inside MPPI
```

## Implementation Milestones

1. Add CBF params to sim and real YAMLs.
2. Add `cbf_enabled` and cost weights to `mppi_node.py`.
3. Add a `/scan` subscriber that stores fixed-size obstacle points.
4. Publish obstacle markers for debugging.
5. Pass `obstacles_xy` and `obstacle_mask` into `mppi.update`.
6. Add obstacle bubble cost.
7. Add discrete CBF progress cost.
8. Add risk-aware speed cost.
9. Compare baseline MPPI vs obstacle-aware MPPI in sim.
10. Only after that, consider a final-action CBF-QP shield.

## What To Implement First

Start with this reward extension:

```math
R_{new}
= R_{track}
- w_{obs}\sum_{k,i}\max(0, r_{safe}^2 - \|p_k - o_i\|^2)^2
```

Then add:

```math
- w_{cbf}\sum_{k,i}
\max(0, (1 - \gamma)h_i(x_k) - h_i(x_{k+1}))^2
```

Then add:

```math
- w_{risk}\sum_k v_k^2 \exp(-d_{min,k}/\sigma)
```

This gives a clean progression:

```text
obstacle avoidance
then CBF-style safety progress
then speed awareness near obstacles
```

That is enough for a strong final project extension before attempting a full
QP safety shield.
