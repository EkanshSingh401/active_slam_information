#pragma once
// ============================================================================
// view_geometry.hpp
//
// Builds the block-sparse measurement Jacobian H_j for a landmark predicted
// visible from a candidate viewpoint (Formulation section 3).
//
// CONVENTIONS -- must match OpenVINS.  Verify with the cross-check in the
// integration prompt before trusting any score; a mismatch here compiles and
// produces plausible but wrong rankings.
//
//   Orientation   R_GtoI (JPL q_GtoI).  Error:  R_GtoI = (I - [dth]x) R_GtoI_hat
//   Position      p_IinG, additive error
//   IMU error     [dth(3) dp(3) dv(3) dbg(3) dba(3)]   pose = first 6
//   Clone error   [dth(3) dp(3)]
//   Measurement   normalized image coordinates (x/z, y/z)
//
// CANDIDATE POSE (v1, common-prior approximation)
//   The candidate is a deterministic displacement of the current pose in the
//   global frame:   R_c = R_delta R_now,   p_c = p_now + dp_G
//   No process noise is added along the motion, so the candidate's pose error
//   is a linear function of the current IMU pose error:
//        dth_c = R_delta dth_now,    dp_c = dp_now
//   Every candidate is therefore scored against the same prior, and ranking by
//   gain is equivalent to ranking by posterior entropy.  When propagation
//   noise is added (non-common prior), rank by posterior_logdet, NOT by delta
//   -- see Atanasov et al. 2015, Table I.
//
// WHAT GEOMETRY VS NOISE CAPTURES
//   Range, viewing angle and parallax already enter through H (the 1/Z terms
//   and the projection geometry).  The noise model captures appearance
//   reliability only -- texture, blur, illumination.  Keep them separate: do
//   not inflate sigma for distance, the Jacobian already did.
// ============================================================================

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <vector>

#include "active_slam_information/information_gain.hpp"

namespace active_slam_information {

struct Pose {
  Eigen::Matrix3d R_GtoI = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_IinG = Eigen::Vector3d::Zero();
};

struct CameraModel {
  Eigen::Matrix3d R_ItoC = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_IinC = Eigen::Vector3d::Zero();
  double fx = 430.0, fy = 430.0, cx = 427.0, cy = 244.0;
  int width = 848, height = 480;
  double min_depth = 0.3, max_depth = 6.0;
  double border_px = 10.0;  // reject projections this close to the edge
};

// Linearization of a landmark's global position with respect to the state:
//   delta p_FinG = sum_b rep[b].J * delta x[rep[b].col : +k]
// Real OpenVINS landmark (ANCHORED_MSCKF_INVERSE_DEPTH):
//   rep = { (lambda_col, dp/dlambda 3x3), (anchor_clone_col, dp/danchor 3x6) }
//   computed inside OpenVINS by UpdaterHelper::get_feature_jacobian_representation
// Virtual frontier landmark:
//   rep = { (virtual_col, I_3) }
struct LandmarkLinearization {
  Eigen::Vector3d p_FinG = Eigen::Vector3d::Zero();
  std::vector<JacobianBlock> rep;
  int id = -1;
  bool is_virtual = false;
};

// Appearance-reliability noise model.  v1 baseline is constant and must
// match OpenVINS up_msckf_sigma_px.  The texture-dependent model (open
// question 4) plugs in here.
class NoiseModel {
 public:
  virtual ~NoiseModel() = default;
  virtual double sigma_px(const LandmarkLinearization& lm, const Pose& candidate,
                          const Eigen::Vector3d& p_C) const = 0;
};

class ConstantNoiseModel : public NoiseModel {
 public:
  explicit ConstantNoiseModel(double sigma_px) : sigma_px_(sigma_px) {}
  double sigma_px(const LandmarkLinearization&, const Pose&,
                  const Eigen::Vector3d&) const override {
    return sigma_px_;
  }
 private:
  double sigma_px_;
};

// Optional line-of-sight test (camera centre -> landmark), e.g. an ESDF
// raycast.  Return true if occluded.
using OcclusionFn =
    std::function<bool(const Eigen::Vector3d& from_G, const Eigen::Vector3d& to_G)>;

Eigen::Matrix3d skew(const Eigen::Vector3d& v);

Eigen::Vector3d camera_center_in_global(const Pose& imu_pose, const CameraModel& cam);

// Build the predicted measurement for one landmark seen from `candidate`.
// Returns false if the landmark is not visible (behind camera, outside depth
// range, outside image, or occluded).
//   imu_pose_col : column of the IMU orientation error in the state
//                  (IMU block id(); position follows at +3)
bool predict_measurement(const Pose& current, const Pose& candidate,
                         int imu_pose_col, const CameraModel& cam,
                         const LandmarkLinearization& lm, const NoiseModel& noise,
                         const OcclusionFn& occluded, PredictedMeasurement* out);

// Noise-free measurement function, for finite-difference tests.
Eigen::Vector2d project_normalized(const Pose& candidate, const CameraModel& cam,
                                   const Eigen::Vector3d& p_FinG);

// ---------------------------------------------------------------------------
struct CandidateScore {
  int index = -1;
  InformationGain gain;
  int visible = 0;
};

// Score every candidate against one prior.  Result sorted best-first
// (ascending posterior_logdet).  Candidates that fail (no PD factorization)
// are placed last.
std::vector<CandidateScore> score_candidates(
    const InformationPrior& prior, const Pose& current,
    const std::vector<Pose>& candidates, int imu_pose_col,
    const CameraModel& cam, const std::vector<LandmarkLinearization>& landmarks,
    const NoiseModel& noise, const OcclusionFn& occluded = nullptr);

}  // namespace active_slam_information
