# Opponent Predictor

This package predicts where an opponent car is now and where it is likely to be over a short horizon. It is designed to feed future MPPI opponent-avoidance costs, while staying simple enough to tune and explain for the final project.

The key idea is to predict in **track progress coordinates** instead of raw Cartesian coordinates. A linear $x,y$ prediction drives straight through corners, while a raceline-progress prediction naturally follows the track curvature.

## State And Measurements

The filter state is:

2
\mathbf{x} =
\begin{bmatrix}
s \\
v
\end{bmatrix}
2

where:

2
s = \text{distance/progress along the raceline}
2

2
v = \dot{s}
2

The opponent odometry gives a Cartesian pose:

2
\mathbf{p} =
\begin{bmatrix}
x \\
y
\end{bmatrix}
2

That pose is projected onto the raceline to obtain:

2
s_{\text{meas}}, \quad e_y, \quad d_{\text{proj}}
2

where $e_y$ is the signed lateral offset from the raceline and $d_{\text{proj}}$ is the projection distance.

## Raceline Projection

For each raceline segment with endpoints $\mathbf{p}_i$ and $\mathbf{p}_{i+1}$, compute:

2
\alpha =
\text{clip}
\left(
\frac{(\mathbf{p} - \mathbf{p}_i)^T(\mathbf{p}_{i+1}-\mathbf{p}_i)}
{\|\mathbf{p}_{i+1}-\mathbf{p}_i\|^2},
0,
1
\right)
2

The projected point is:

2
\mathbf{p}_{\text{proj}} =
\mathbf{p}_i + \alpha(\mathbf{p}_{i+1}-\mathbf{p}_i)
2

The measured track progress is:

2
s_{\text{meas}} =
s_i + \alpha(s_{i+1}-s_i)
2

The signed lateral offset is:

2
e_y =
(\mathbf{p} - \mathbf{p}_{\text{proj}})^T\mathbf{n}_i
2

where $\mathbf{n}_i$ is the raceline normal.

Because the track is closed, $s$ is wrapped to:

2
s \in [0, L)
2

where $L$ is the total track length. When computing differences between two progress values, the predictor unwraps $s$ so crossing the start line does not look like a large jump.

## Speed Measurement

The primary speed measurement is progress speed:

2
v_{\text{meas}} =
\frac{s_t - s_{t-1}}{\Delta t}
2

This is preferred over raw odometry twist because the predictor cares about how quickly the opponent is moving along the track.

The measurement is rejected or clamped if it is physically unreasonable:

2
0 \le v_{\text{meas}} \le v_{\max}
2

and speed changes can be limited with:

2
|v_{\text{meas},t} - v_{\text{meas},t-1}|
\le
a_{\max}\Delta t
2

If progress speed is unavailable or invalid, the node falls back to twist magnitude:

2
v_{\text{twist}} =
\sqrt{v_x^2 + v_y^2 + v_z^2}
2

If both are poor, it can fall back to the raceline profile speed at the current progress.

## Kalman Filter

The predictor uses a constant-velocity Kalman filter in raceline progress:

2
\mathbf{x}_{k+1}
=
\mathbf{A}\mathbf{x}_k
2

2
\mathbf{A}
=
\begin{bmatrix}
1 & \Delta t \\
0 & 1
\end{bmatrix}
2

Equivalently:

2
s_{k+1} = s_k + v_k\Delta t
2

2
v_{k+1} = v_k
2

The measurement is:

2
\mathbf{z}_k =
\begin{bmatrix}
s_{\text{meas}} \\
v_{\text{meas}}
\end{bmatrix}
2

with measurement model:

2
\mathbf{z}_k =
\mathbf{H}\mathbf{x}_k
2

2
\mathbf{H}
=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
2

This is a standard linear Kalman filter, not an EKF. A constant-acceleration model with state $[s, v, a]^T$ would also still be linear:

2
\begin{bmatrix}
s_{k+1} \\
v_{k+1} \\
a_{k+1}
\end{bmatrix}
=
\begin{bmatrix}
1 & \Delta t & \frac{1}{2}\Delta t^2 \\
0 & 1 & \Delta t \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
s_k \\
v_k \\
a_k
\end{bmatrix}
2

We currently prefer the constant-velocity model because acceleration estimates would require differentiating noisy projected speed measurements. The raceline speed profile already provides a cleaner way to bias future prediction toward slowing for turns and speeding up on straights.

## Position Snapping

For visualization and obstacle-cost anchoring, the current predicted pose should not lag behind the measured opponent pose. After the Kalman update, the progress estimate is optionally snapped toward the latest measurement:

2
s_{\text{hat}}
\leftarrow
(1-\alpha_{\text{snap}})s_{\text{hat}}
+
\alpha_{\text{snap}}s_{\text{meas}}
2

With:

2
\alpha_{\text{snap}} = 1
2

the current marker is anchored to the newest odometry projection, while velocity remains filtered.

## Future Prediction

The prediction horizon starts from the filtered current state:

2
s_0 = s_{\text{hat}}
2

2
v_0 = v_{\text{hat}}
2

At each future step, the position is obtained by interpolating the raceline:

2
\mathbf{p}_k = \mathbf{r}(s_k)
2

The future speed is blended toward the raceline speed profile:

2
v_{k+1}
=
(1-\beta)v_k
+
\beta v_{\text{profile}}(s_k)
2

Then progress advances:

2
s_{k+1}
=
s_k + v_{k+1}\Delta t
2

When the opponent is visible, $\beta$ is `profile_speed_blend`. When the opponent is stale or temporarily out of sight, $\beta$ becomes `out_of_sight_profile_speed_blend`.

This means:

- with $\beta = 0$, prediction follows filtered opponent speed;
- with $\beta = 1$, prediction follows the raceline speed profile;
- with small $\beta$, prediction mostly follows the measured opponent but slowly returns toward expected track behavior.

## Lateral Offset

The opponent is not forced exactly onto the raceline. The node filters the measured lateral offset:

2
e_{y,\text{hat}}
\leftarrow
(1-\alpha_y)e_{y,\text{hat}}
+
\alpha_y e_{y,\text{meas}}
2

Future lateral offset decays back toward the raceline:

2
e_y(k)
=
e_{y,\text{hat}}
\exp
\left(
-\frac{k\Delta t}{\tau_y}
\right)
2

The final predicted position is:

2
\mathbf{p}^{\text{pred}}_k
=
\mathbf{r}(s_k) + e_y(k)\mathbf{n}(s_k)
2

This lets the prediction start from the opponent's measured side of the track, while assuming it gradually returns toward the reference.

## Main Parameters

`waypoint_path`
: Raceline CSV used for projection and prediction. This must match the current map and track.

`frame_id`
: Output frame, usually `map`.

`pose_source`
: `odom_pose` trusts map-frame odometry directly. `tf` uses the opponent child-frame transform.

`prediction_steps`
: Number of future points to publish.

`prediction_dt`
: Time spacing between predicted points.

`profile_speed_blend`
: Blend toward raceline speed profile while opponent odometry is fresh.

`out_of_sight_profile_speed_blend`
: Blend toward raceline speed profile when opponent odometry is stale.

`stale_timeout`
: Time after which prediction is considered stale.

`max_stale_prediction_time`
: Time after which the node stops publishing stale predictions.

`speed_profile_scale`
: Multiplier on raceline speed values.

`speed_profile_min_speed`
: Lower clamp on profile speed.

`speed_profile_max_speed`
: Upper clamp on profile speed.

`max_progress_speed`
: Maximum accepted progress-speed measurement.

`max_progress_accel`
: Maximum accepted change in measured progress speed.

`position_snap_alpha`
: How strongly fresh measurements anchor the current position estimate.

`lateral_offset_alpha`
: Filter gain for measured lateral offset.

`lateral_offset_decay_time`
: Time constant for predicted lateral offset to decay back toward the raceline.

`kf_process_var_s`
: Process noise for raceline progress.

`kf_process_var_v`
: Process noise for progress speed.

`kf_measurement_var_s`
: Measurement noise for projected raceline progress.

`kf_measurement_var_v`
: Measurement noise for progress speed.

## Published Topics

`/opponent/odom`
: Current filtered opponent estimate as `nav_msgs/Odometry`.

`/opponent/predicted_path`
: Future opponent path as `nav_msgs/Path`.

`/opponent/markers`
: Foxglove/RViz markers for current opponent estimate and future prediction.

`/opponent/debug`
: Debug vector:

2
\begin{bmatrix}
s_{\text{proj}} &
v_{\text{hat}} &
v_{\text{progress}} &
v_{\text{twist}} &
v_{\text{profile}} &
d_{\text{proj}} &
e_{y,\text{hat}} &
\text{stale}
\end{bmatrix}
2

## Algorithm Summary

1. Receive opponent odometry.
2. Transform or trust the pose in `map`.
3. Project the pose onto the raceline.
4. Compute progress speed from consecutive projected $s$ values.
5. Update the Kalman filter state $[s, v]^T$.
6. Snap current progress toward the fresh projected measurement for low-lag visualization.
7. Predict future progress along the raceline using filtered speed and profile-speed blending.
8. Apply lateral offset decay.
9. Publish odometry, path, markers, and debug values.

