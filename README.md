# active_slam_information

D-optimal information gain of candidate viewpoints, computed on the joint
covariance published by the OpenVINS fork (`/openvins/joint_covariance`).
Implements formulation draft v0.1, sections 3–5. See `MATH_TO_CODE.md` for the
derivation, the equation-to-code map, and the verification table.

## Status

| Piece | Status |
|---|---|
| Information gain (`information_gain.{hpp,cpp}`) | real, tested against a brute-force reference |
| Measurement Jacobians (`view_geometry.{hpp,cpp}`) | real, tested against finite differences |
| Match to OpenVINS conventions | **NOT YET VERIFIED** — see below |
| ROS node | not built yet; `active_slam_planner`'s `fisher_ig_estimator` is still a placeholder |

**Do not trust any score until the convention cross-check passes.** The
finite-difference test proves the code matches the convention written in
`view_geometry.hpp` (JPL `R_GtoI`, error `R = (I - [dθ]×) R̂`). It does not
prove that convention is OpenVINS's. That requires comparing `H` from
`predict_measurement()` against OpenVINS's own
`UpdaterHelper::get_feature_jacobian_full` on real or simulated data.

## Build and test

Without ROS (anywhere Eigen 3.4 is installed):

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

In a colcon workspace:

```bash
colcon build --packages-select active_slam_information
colcon test  --packages-select active_slam_information && colcon test-result --verbose
```

Expected: `134 passed, 0 failed`.

## Known gap before a node can use this

The current `JointCovariance` message carries covariance but no means. Scoring
needs the linearization point: IMU pose, each SLAM landmark's global position,
its representation Jacobian (from OpenVINS's
`UpdaterHelper::get_feature_jacobian_representation`), and camera
extrinsics/intrinsics — all captured in the same callback as the covariance.
