#pragma once
// ============================================================================
// joint_cov_problem.hpp  (ROS-dependent; header-only)
//
// Turns one active_slam_msgs/JointCovariance (with the linearization point the
// OpenVINS fork publishes since PATCHES s60) into the inputs of this library:
//   Sigma        the joint covariance, in the message's block order
//   imu_col      column of the IMU block (orientation error first)
//   current      the IMU pose (value, not FEJ: a planner predicts new
//                measurements, which OpenVINS will linearize at the then-current
//                clone, i.e. ~ the value)
//   cameras      one CameraModel per camera (OpenVINS JPL q_ItoC -> R_ItoC)
//   landmarks    LandmarkLinearization with rep blocks re-indexed from OpenVINS
//                state ids to MESSAGE columns
//   T_metric     scoring map: IMU state (15) + every usable landmark's global XYZ
//                (rows = sum of its representation Jacobians), MATH_TO_CODE.md
//
// Convention verified against OpenVINS by the s60 gate (docker/tools/
// ov_jacobian_gate): JPL q_GtoI, R = (I - [dth]x) R_hat, clone error [dth dp].
// ============================================================================

#include <map>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "active_slam_information/information_gain.hpp"
#include "active_slam_information/view_geometry.hpp"
#include "active_slam_msgs/msg/joint_covariance.hpp"

namespace active_slam_information {

// JPL quaternion [x y z w] -> rotation matrix, as ov_core::quat_2_Rot.
inline Eigen::Matrix3d jpl_quat_to_rot(const std::array<double, 4>& q) {
  const Eigen::Vector3d v(q[0], q[1], q[2]);
  const double w = q[3];
  return (2.0 * w * w - 1.0) * Eigen::Matrix3d::Identity() - 2.0 * w * skew(v) + 2.0 * v * v.transpose();
}

struct JointCovProblem {
  bool ok = false;
  std::string why;
  double stamp = 0.0;
  Eigen::MatrixXd Sigma;
  int imu_col = -1;
  Pose current;
  std::vector<CameraModel> cameras;
  std::vector<LandmarkLinearization> landmarks;  // usable ones only
  std::map<long long, int> landmark_index;       // feature id -> index in landmarks
  std::map<int, int> col_of_state;               // OpenVINS state id -> message column
  std::map<double, int> clone_col;               // clone timestamp -> message column

  // Scoring map: IMU (15) + metric XYZ of the landmarks whose ids are in `keep`
  // (all usable landmarks if keep is empty).
  Eigen::MatrixXd T_metric(const std::set<long long>& keep = {}) const {
    std::vector<int> sel;
    for (size_t j = 0; j < landmarks.size(); ++j)
      if (keep.empty() || keep.count(landmarks[j].id)) sel.push_back((int)j);
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(15 + 3 * (int)sel.size(), Sigma.cols());
    T.block(0, imu_col, 15, 15).setIdentity();
    for (size_t k = 0; k < sel.size(); ++k)
      for (const auto& r : landmarks[sel[k]].rep) T.block(15 + 3 * (int)k, r.col, 3, r.J.cols()) += r.J;
    return T;
  }
};

inline JointCovProblem problem_from_msg(const active_slam_msgs::msg::JointCovariance& m, double max_depth = 20.0,
                                        int width = 848, int height = 480) {
  JointCovProblem P;
  P.stamp = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec;
  const int n = m.dim;
  if (n <= 0 || (int)m.covariance.size() != n * n) { P.why = "bad covariance size"; return P; }
  if (m.cameras.empty()) { P.why = "no linearization point (pre-s60 publisher?)"; return P; }
  P.Sigma = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(m.covariance.data(), n, n);
  P.Sigma = 0.5 * (P.Sigma + P.Sigma.transpose());
  for (const auto& b : m.blocks) {
    P.col_of_state[b.state_id] = b.index;
    if (b.type == "imu") P.imu_col = b.index;
    if (b.type == "clone") P.clone_col[b.clone_timestamp] = b.index;
  }
  if (P.imu_col < 0) { P.why = "no imu block"; return P; }
  P.current.R_GtoI = jpl_quat_to_rot(m.imu_pose.q_gtoi);
  P.current.p_IinG = Eigen::Vector3d(m.imu_pose.p_iing[0], m.imu_pose.p_iing[1], m.imu_pose.p_iing[2]);
  for (const auto& c : m.cameras) {
    CameraModel cam;
    cam.R_ItoC = jpl_quat_to_rot(c.q_itoc);
    cam.p_IinC = Eigen::Vector3d(c.p_iinc[0], c.p_iinc[1], c.p_iinc[2]);
    cam.fx = c.intrinsics[0]; cam.fy = c.intrinsics[1]; cam.cx = c.intrinsics[2]; cam.cy = c.intrinsics[3];
    cam.width = width; cam.height = height; cam.max_depth = max_depth;
    P.cameras.push_back(cam);
  }
  for (const auto& L : m.landmarks) {
    if (!P.col_of_state.count(L.state_id)) continue;
    LandmarkLinearization lm;
    lm.id = (int)L.feature_id;
    lm.p_FinG = Eigen::Vector3d(L.p_fing[0], L.p_fing[1], L.p_fing[2]);
    Eigen::Matrix3d Hf;
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) Hf(r, c) = L.h_f[3 * r + c];
    lm.rep.push_back({P.col_of_state.at(L.state_id), Hf});
    int cols = 0;
    for (int s : L.hx_sizes) cols += s;
    bool usable = true;
    int c0 = 0;
    for (size_t k = 0; k < L.hx_state_ids.size(); ++k) {
      auto it = P.col_of_state.find(L.hx_state_ids[k]);
      if (it == P.col_of_state.end()) { usable = false; break; }  // dependency not in this matrix
      Eigen::MatrixXd J(3, L.hx_sizes[k]);
      for (int r = 0; r < 3; r++) for (int c = 0; c < L.hx_sizes[k]; c++) J(r, c) = L.h_x[(size_t)r * cols + c0 + c];
      lm.rep.push_back({it->second, J});
      c0 += L.hx_sizes[k];
    }
    if (!usable) continue;
    P.landmark_index[L.feature_id] = (int)P.landmarks.size();
    P.landmarks.push_back(lm);
  }
  P.ok = true;
  return P;
}

// Predicted measurements of the problem's landmarks from one candidate IMU pose,
// all cameras, common-prior mapping onto the current IMU pose (v1, as built).
inline std::vector<PredictedMeasurement> predict_all(const JointCovProblem& P, const Pose& candidate, const NoiseModel& noise,
                                                     const std::set<long long>* only = nullptr) {
  std::vector<PredictedMeasurement> out;
  for (const auto& lm : P.landmarks) {
    if (only && !only->count(lm.id)) continue;
    for (const auto& cam : P.cameras) {
      PredictedMeasurement pm;
      if (predict_measurement(P.current, candidate, P.imu_col, cam, lm, noise, nullptr, &pm)) out.push_back(pm);
    }
  }
  return out;
}

}  // namespace active_slam_information
