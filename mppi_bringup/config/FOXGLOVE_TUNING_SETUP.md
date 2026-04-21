# MPPI Foxglove Tuning Setup

Use Foxglove as the primary tuning UI. It is sufficient for the current MPPI stack.
Do not build a custom dashboard yet. The current bottleneck is controller tuning and
debug signal selection, not frontend capability.

There is now a checked-in layout you can import directly:

- `/home/cedric/ros2_ws/roboracer_ws/src/mppi/mppi_bringup/config/foxglove/sim_mppi_tuning.json`

That file is based on your saved local `sim - MPPI` layout and updated to use the
current debug topics, including total cost.

## Goals

The layout should answer four questions at a glance:

1. Is the car following the reference cleanly?
2. Is the controller choosing safe trajectories?
3. Is speed timing correct before and after turns?
4. Are costs activating only when they should?

## Recommended Panels

### 1. 3D Panel: spatial view

Topics:
- `/mppi/reference`
- `/mppi/optimal_trajectory`
- `/mppi/sampled_trajectories`
- `/scan`
- `/map`
- `/ego_robot_description` if available

Purpose:
- verify reference vs chosen trajectory overlay
- see if sampled rollouts cluster tightly or spread badly
- verify wall-cost behavior near track boundaries

Notes:
- sampled trajectory count should stay modest for runtime sanity
- use the fixed frame `map`

### 2. Plot Panel: speed tracking

Topics:
- `/mppi/speed_debug`

Fields from the array:
- index 0: actual speed
- index 1: MPPI speed command
- index 2: profiled speed command
- index 3: final drive speed
- index 4: blend

Purpose:
- check whether the profile is braking early enough
- check whether final drive speed matches the intended policy
- catch late accel / late brake behavior

### 3. Plot Panel: reward and total cost

Topics:
- `/mppi/debug/reward_total_sum`
- `/mppi/debug/reward_total_mean`
- `/mppi/debug/cost_total_sum`
- `/mppi/debug/cost_total_mean`
- `/mppi/debug/cost_invalid_sum`

Purpose:
- compare overall reward vs total penalty
- verify invalid trajectory penalties stay near zero during healthy operation

Interpretation:
- `cost_total_sum` should spike near hazardous situations
- `cost_invalid_sum` should usually be near zero
- the absolute values matter less than how they change before instability

### 4. Plot Panel: reward breakdown

Topics:
- `/mppi/debug/reward_xy_sum`
- `/mppi/debug/reward_velocity_sum`
- `/mppi/debug/reward_yaw_sum`

Purpose:
- see which reward term is actually shaping decisions
- detect when one term is effectively inactive

### 5. Plot Panel: cost breakdown

Topics:
- `/mppi/debug/cost_wall_sum`
- `/mppi/debug/cost_slip_sum`
- `/mppi/debug/cost_latacc_sum`
- `/mppi/debug/cost_steer_sat_sum`

Purpose:
- verify which cost is actually active
- detect dead/no-op costs
- detect costs that are always active and therefore too strong

Interpretation:
- `cost_wall_sum` should be mostly near zero in clean operation and rise near walls
- `cost_slip_sum` should rise during unstable behavior, not constantly
- `cost_latacc_sum` should reflect aggressive high-speed turning, not dominate everywhere

### 6. Plot Panel: physical proxies

Topics:
- `/mppi/debug/min_wall_dist`
- `/mppi/debug/max_beta`
- `/mppi/debug/max_latacc`
- `/mppi/debug/max_abs_steer`
- `/mppi/debug/invalid_steps`

Purpose:
- tune physical thresholds directly
- compare raw physical proxy magnitudes against cost activation

Interpretation:
- `min_wall_dist` is useful for wall-margin tuning
- `max_beta` is useful for slip threshold tuning
- `max_latacc` is useful for lateral-acceleration threshold tuning
- `max_abs_steer` is useful if steer-saturation cost gets enabled later

### 7. Raw Messages Panel

Topics:
- `/mppi/debug/cost_total_sum`
- `/mppi/debug/cost_wall_sum`
- `/mppi/debug/max_beta`
- `/mppi/speed_debug`

Purpose:
- quick exact-number inspection while tuning

## Suggested Layout

If you want the fast path, import `sim_mppi_tuning.json` and start there.

Top row:
- left: 3D panel
- right: speed tracking plot

Middle row:
- left: reward + total cost plot
- right: cost breakdown plot

Bottom row:
- left: physical proxies plot
- right: raw messages

Optional extra plot:
- reward breakdown plot if you want to keep reward terms separate from total reward

The checked-in layout is a little different on purpose:
- left column: costs, rewards, physical proxies
- right column: parameters, 3D scene, raw messages

That mirrors the current `sim - MPPI` layout you already saved locally, while making
room for the new total-cost signal.

## Topics to Record in Bags

Record these at minimum:
- `/drive`
- `/ego_racecar/odom` or `/pf/pose/odom`
- `/mppi/speed_debug`
- `/mppi/reference`
- `/mppi/optimal_trajectory`
- `/mppi/debug/reward_total_sum`
- `/mppi/debug/reward_total_mean`
- `/mppi/debug/reward_xy_sum`
- `/mppi/debug/reward_velocity_sum`
- `/mppi/debug/reward_yaw_sum`
- `/mppi/debug/cost_total_sum`
- `/mppi/debug/cost_total_mean`
- `/mppi/debug/cost_invalid_sum`
- `/mppi/debug/cost_wall_sum`
- `/mppi/debug/cost_slip_sum`
- `/mppi/debug/cost_latacc_sum`
- `/mppi/debug/cost_steer_sat_sum`
- `/mppi/debug/min_wall_dist`
- `/mppi/debug/max_beta`
- `/mppi/debug/max_latacc`
- `/mppi/debug/max_abs_steer`
- `/mppi/debug/invalid_steps`

## What This Setup Does Not Show Yet

This setup only shows debug signals from the selected optimal trajectory.

That is enough for the next hardware step, but it is not enough for full tuning
of barrier-style costs. The next useful instrumentation upgrade is population
statistics over all sampled trajectories, for example:

- average sampled total cost
- weighted-average sampled total cost
- p90 / p95 sampled total cost
- fraction of rollouts with nonzero wall cost
- fraction of rollouts with nonzero slip cost
- effective sample size

Those should be added later if hardware tuning still feels blind.
