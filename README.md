# cbf_mppi_f1tenth

ROS 2 packages for running a JAX-based MPPI controller on F1TENTH cars in sim
and on the real car. Forked from
[mlab-upenn/mppi_example](https://github.com/mlab-upenn/mppi_example)
(MIT, © 2025 xLab for Safe Autonomous Systems).

Notes:
- `MPPI_GUIDE.md` — math, code map, and tuning notes.
- `CBF_MPPI_README.md` — design notes for adding obstacle awareness via CBF
  ideas (work in progress).
- `mppi_sim1.mp4` — sim run.

## Layout

- `mppi_example/` — the controller package. `mppi_node.py` subscribes to
  odometry and publishes to `/drive`; `mppi_tracking.py` is the JAX rollout
  loop; `dynamics_models/` holds the vehicle models.
- `mppi_bringup/` — launch files, params, and waypoint CSVs (Levine 9-column
  format under `waypoints/sim/`).

## Build

Drop into a ROS 2 workspace under `src/`:

```bash
cd ~/ros2_ws/roboracer_ws
colcon build --packages-select mppi_example mppi_bringup
source install/setup.bash
```

## Run (sim)

```bash
ros2 launch mppi_bringup mppi.launch.py
```

The launch publishes 20 zero-drive messages to keep the car still while JAX
warms up, then starts `lmppi_node` after a 1.6 s timer. Override with
`params_file:=...` or `drive_topic:=...`.

## Changes vs upstream

- `mppi_bringup` package: bundled launch + a single `params_sim.yaml`.
- `mppi_node.py`: `wpt_path` and `wpt_path_absolute` parameters that load a
  raceline directly from a CSV, bypassing the upstream `map_info.txt` flow.
- `Track.load_map_from_csv()`: standalone loader for the same.
- Reward weights (`xy_reward_weight`, `velocity_reward_weight`,
  `yaw_reward_weight`) and RViz marker publishers
  (`/mppi/reference`, `/mppi/optimal_trajectory`,
  `/mppi/sampled_trajectories`) added to the node.

## License

MIT — see [`LICENSE`](LICENSE). Includes upstream's MIT notice from xLab.