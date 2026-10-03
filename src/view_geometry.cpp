#include "active_slam_information/view_geometry.hpp"

#include <algorithm>
#include <limits>

namespace active_slam_information {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0, -v.z(), v.y(),
       v.z(), 0, -v.x(),
      -v.y(), v.x(), 0;
  return S;
}

Eigen::Vector3d camera_center_in_global(const Pose& imu_pose, const CameraModel& cam) {
  // p_C = R_ItoC p_I + p_IinC = 0  =>  p_I = -R_ItoC^T p_IinC
  const Eigen::Vector3d p_CinI = -cam.R_ItoC.transpose() * cam.p_IinC;
  return imu_pose.p_IinG + imu_pose.R_GtoI.transpose() * p_CinI;
}

Eigen::Vector2d project_normalized(const Pose& c, const CameraModel& cam,
                                   const Eigen::Vector3d& p_FinG) {
  const Eigen::Vector3d p_I = c.R_GtoI * (p_FinG - c.p_IinG);
  const Eigen::Vector3d p_C = cam.R_ItoC * p_I + cam.p_IinC;
  return p_C.head<2>() / p_C.z();
}

bool predict_measurement(const Pose& current, const Pose& candidate,
                         int imu_pose_col, const CameraModel& cam,
                         const LandmarkLinearization& lm, const NoiseModel& noise,
                         const OcclusionFn& occluded, PredictedMeasurement* out) {
  // --- geometry at the candidate --------------------------------------------
  const Eigen::Vector3d p_I = candidate.R_GtoI * (lm.p_FinG - candidate.p_IinG);
  const Eigen::Vector3d p_C = cam.R_ItoC * p_I + cam.p_IinC;
  const double X = p_C.x(), Y = p_C.y(), Z = p_C.z();

  // --- visibility ------------------------------------------------------------
  if (!(Z > cam.min_depth) || Z > cam.max_depth) return false;
  const double u = cam.fx * X / Z + cam.cx;
  const double v = cam.fy * Y / Z + cam.cy;
  if (u < cam.border_px || u > cam.width - cam.border_px ||
      v < cam.border_px || v > cam.height - cam.border_px)
    return false;
  if (occluded && occluded(camera_center_in_global(candidate, cam), lm.p_FinG))
    return false;

  // --- Jacobians -------------------------------------------------------------
  // dz/dp_C for normalized coordinates
  Eigen::Matrix<double, 2, 3> dz_dpc;
  dz_dpc << 1.0 / Z, 0.0, -X / (Z * Z),
            0.0, 1.0 / Z, -Y / (Z * Z);

  // Candidate pose error.  R_GtoI = (I - [dth]x) R_hat  =>  dp_I/ddth = [p_I]x
  const Eigen::Matrix<double, 2, 3> H_th_c = dz_dpc * cam.R_ItoC * skew(p_I);
  const Eigen::Matrix<double, 2, 3> H_p_c = -dz_dpc * cam.R_ItoC * candidate.R_GtoI;

  // Map to current IMU pose error:  dth_c = R_delta dth_now,  dp_c = dp_now
  const Eigen::Matrix3d R_delta = candidate.R_GtoI * current.R_GtoI.transpose();
  Eigen::Matrix<double, 2, 6> H_imu;
  H_imu.leftCols<3>() = H_th_c * R_delta;
  H_imu.rightCols<3>() = H_p_c;

  // Landmark:  dz/dp_FinG, chained through the representation Jacobian.
  const Eigen::Matrix<double, 2, 3> H_pf = dz_dpc * cam.R_ItoC * candidate.R_GtoI;

  out->blocks.clear();
  out->blocks.reserve(1 + lm.rep.size());
  out->blocks.push_back({imu_pose_col, H_imu});
  for (const auto& r : lm.rep) out->blocks.push_back({r.col, H_pf * r.J});

  // Noise: pixel sigma -> normalized units.
  const double f = 0.5 * (cam.fx + cam.fy);
  out->sigma = noise.sigma_px(lm, candidate, p_C) / f;
  return true;
}

std::vector<CandidateScore> score_candidates(
    const InformationPrior& prior, const Pose& current,
    const std::vector<Pose>& candidates, int imu_pose_col,
    const CameraModel& cam, const std::vector<LandmarkLinearization>& landmarks,
    const NoiseModel& noise, const OcclusionFn& occluded) {
  std::vector<CandidateScore> scores;
  scores.reserve(candidates.size());
  std::vector<PredictedMeasurement> meas;
  meas.reserve(landmarks.size());

  for (int c = 0; c < static_cast<int>(candidates.size()); ++c) {
    meas.clear();
    PredictedMeasurement pm;
    for (const auto& lm : landmarks)
      if (predict_measurement(current, candidates[c], imu_pose_col, cam, lm, noise,
                              occluded, &pm))
        meas.push_back(pm);

    CandidateScore s;
    s.index = c;
    s.visible = static_cast<int>(meas.size());
    s.gain = evaluate(prior, meas);
    scores.push_back(s);
  }

  std::stable_sort(scores.begin(), scores.end(),
                   [](const CandidateScore& a, const CandidateScore& b) {
                     if (a.gain.ok != b.gain.ok) return a.gain.ok;
                     return a.gain.posterior_logdet < b.gain.posterior_logdet;
                   });
  return scores;
}

}  // namespace active_slam_information
