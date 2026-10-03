#pragma once
// ============================================================================
// information_gain.hpp
//
// D-optimal information gain of a set of predicted measurements, evaluated on
// a scored linear function of the estimator state.  Implements Formulation
// draft v0.1, sections 3-5.
//
//   State covariance        Sigma            (n x n, may be rank deficient)
//   Scoring map             T                (s x n)
//   Scored covariance       Sigma_S = T Sigma T^T      (must be PD)
//   Measurement             z = H x + v,  v ~ N(0, V)
//
//   Kalman update           Sigma+  = Sigma - Sigma H^T A^-1 H Sigma
//                           A       = H Sigma H^T + V
//   Scored posterior        Sigma_S+ = Sigma_S - B A^-1 B^T,  B = T Sigma H^T
//
//   Information gain        dI = log det Sigma_S - log det Sigma_S+
//                              = log det A - log det D
//                           D  = H Sigma_{.|S} H^T + V
//                           Sigma_{.|S} = Sigma - Sigma T^T Sigma_S^-1 T Sigma
//
// The identity follows from Sylvester's determinant theorem.  Both A and D are
// m x m (m = 2 x number of predicted observations) and strictly PD because V
// is, so the gain is well defined even though the full Sigma is singular: the
// only matrix ever factored at state size is Sigma_S.
//
// T is a general linear map, not only a selection:
//   * selection    rows of I picking IMU + landmark blocks
//   * metric map   landmark rows = d p_FinG / d [lambda, anchor clone]
// A selection is the special case.  dI is invariant to T -> K T for any
// invertible K, so it does not depend on units or on how the scored quantity
// is coordinatized -- only on WHICH quantity is scored.
// ============================================================================

#include <Eigen/Dense>
#include <vector>

namespace active_slam_information {

// A dense (rows x k) Jacobian block located at column offset `col` of the
// state.  Measurement Jacobians use rows = 2.  Landmark representation
// Jacobians use rows = 3.
struct JacobianBlock {
  int col = 0;
  Eigen::MatrixXd J;
};

// One predicted 2D observation in normalized image coordinates, with
// isotropic noise sigma^2 I_2.  H is block sparse: the row pair is the sum of
// the listed blocks placed at their column offsets.
struct PredictedMeasurement {
  std::vector<JacobianBlock> blocks;
  double sigma = 1.0;
};

// Built once per planning cycle from the current joint covariance and scoring
// map.  All state-size work happens here; per-candidate evaluation is
// O(m * nnz(H)) plus two m x m Cholesky factorizations.
class InformationPrior {
 public:
  InformationPrior(const Eigen::MatrixXd& Sigma, const Eigen::MatrixXd& T);

  // False if Sigma_S = T Sigma T^T is not positive definite.  In that case the
  // scoring map includes a degenerate direction (e.g. both the IMU pose and
  // its bit-exact newest clone) and no candidate can be scored.
  bool ok() const { return ok_; }

  double prior_logdet() const { return prior_logdet_; }  // log det Sigma_S
  int state_dim() const { return static_cast<int>(Sigma_.rows()); }
  int scored_dim() const { return scored_dim_; }
  const Eigen::MatrixXd& Sigma() const { return Sigma_; }
  const Eigen::MatrixXd& Sigma_cond() const { return Sigma_cond_; }

 private:
  Eigen::MatrixXd Sigma_;
  Eigen::MatrixXd Sigma_cond_;
  double prior_logdet_ = 0.0;
  int scored_dim_ = 0;
  bool ok_ = false;
};

struct InformationGain {
  double delta = 0.0;             // log det Sigma_S - log det Sigma_S+  (>= 0)
  double posterior_logdet = 0.0;  // log det Sigma_S+   -- rank by this
  int num_measurements = 0;
  bool ok = false;
};

// Fast path.  Uses the precomputed prior.
InformationGain evaluate(const InformationPrior& prior,
                         const std::vector<PredictedMeasurement>& meas);

// Reference path.  Explicit full-state Kalman update, then log det of the
// scored posterior.  O(n^3).  For tests only.
InformationGain evaluate_reference(const Eigen::MatrixXd& Sigma,
                                   const Eigen::MatrixXd& T,
                                   const std::vector<PredictedMeasurement>& meas);

// Dense H (2N x n).  For tests and debugging.
Eigen::MatrixXd dense_jacobian(const std::vector<PredictedMeasurement>& meas,
                               int state_dim);

// Append num_virtual frontier landmarks with prior sigma_l^2 I_3, uncorrelated
// with the estimator state (Formulation section 6).  Returns
// blkdiag(Sigma, sigma_l^2 I_3, ..., sigma_l^2 I_3).  The first virtual
// landmark occupies column Sigma.rows().
Eigen::MatrixXd augment_with_virtual_landmarks(const Eigen::MatrixXd& Sigma,
                                               int num_virtual, double sigma_l);

// Pad a scoring map with identity rows for appended virtual landmarks, so they
// are part of the scored quantity.
Eigen::MatrixXd augment_scoring_map(const Eigen::MatrixXd& T, int old_state_dim,
                                    int num_virtual);

}  // namespace active_slam_information
