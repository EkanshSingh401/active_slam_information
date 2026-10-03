#include "active_slam_information/information_gain.hpp"

#include <cmath>

namespace active_slam_information {

namespace {

// log det of an SPD matrix via Cholesky.  Returns false if not PD.
bool logdet_spd(const Eigen::MatrixXd& M, double* out) {
  Eigen::LLT<Eigen::MatrixXd> llt(M);
  if (llt.info() != Eigen::Success) return false;
  const auto& L = llt.matrixLLT();
  double s = 0.0;
  for (int i = 0; i < L.rows(); ++i) {
    const double d = L(i, i);
    if (!(d > 0.0) || !std::isfinite(d)) return false;
    s += std::log(d);
  }
  *out = 2.0 * s;
  return true;
}

// H M H^T for symmetric M, exploiting block sparsity of H.
//   P = M H^T    (n x m):  column pair i gets  sum_b M[:, b] * J_b^T
//   Q = H P      (m x m):  row pair i gets     sum_b J_b * P[b, :]
Eigen::MatrixXd quad_form(const Eigen::MatrixXd& M,
                          const std::vector<PredictedMeasurement>& meas) {
  const int n = static_cast<int>(M.rows());
  const int m = 2 * static_cast<int>(meas.size());
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, m);
  for (int i = 0; i < static_cast<int>(meas.size()); ++i) {
    for (const auto& b : meas[i].blocks) {
      const int k = static_cast<int>(b.J.cols());
      P.middleCols(2 * i, 2).noalias() += M.middleCols(b.col, k) * b.J.transpose();
    }
  }
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(m, m);
  for (int i = 0; i < static_cast<int>(meas.size()); ++i) {
    for (const auto& b : meas[i].blocks) {
      const int k = static_cast<int>(b.J.cols());
      Q.middleRows(2 * i, 2).noalias() += b.J * P.middleRows(b.col, k);
    }
  }
  return 0.5 * (Q + Q.transpose());
}

void add_noise(Eigen::MatrixXd* Q, const std::vector<PredictedMeasurement>& meas) {
  for (int i = 0; i < static_cast<int>(meas.size()); ++i) {
    const double v = meas[i].sigma * meas[i].sigma;
    (*Q)(2 * i, 2 * i) += v;
    (*Q)(2 * i + 1, 2 * i + 1) += v;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
InformationPrior::InformationPrior(const Eigen::MatrixXd& Sigma,
                                   const Eigen::MatrixXd& T)
    : Sigma_(0.5 * (Sigma + Sigma.transpose())),
      scored_dim_(static_cast<int>(T.rows())) {
  const Eigen::MatrixXd TS = T * Sigma_;                    // s x n
  Eigen::MatrixXd Sigma_S = TS * T.transpose();             // s x s
  Sigma_S = 0.5 * (Sigma_S + Sigma_S.transpose());

  Eigen::LLT<Eigen::MatrixXd> llt(Sigma_S);
  if (llt.info() != Eigen::Success) return;                 // ok_ stays false
  if (!logdet_spd(Sigma_S, &prior_logdet_)) return;

  // W = L^-1 T Sigma,  so W^T W = Sigma T^T Sigma_S^-1 T Sigma.
  const Eigen::MatrixXd W = llt.matrixL().solve(TS);        // s x n
  Sigma_cond_ = Sigma_ - W.transpose() * W;
  Sigma_cond_ = 0.5 * (Sigma_cond_ + Sigma_cond_.transpose());
  ok_ = true;
}

// ---------------------------------------------------------------------------
InformationGain evaluate(const InformationPrior& prior,
                         const std::vector<PredictedMeasurement>& meas) {
  InformationGain g;
  g.num_measurements = static_cast<int>(meas.size());
  if (!prior.ok()) return g;
  if (meas.empty()) {
    g.posterior_logdet = prior.prior_logdet();
    g.ok = true;
    return g;
  }

  Eigen::MatrixXd A = quad_form(prior.Sigma(), meas);       // H Sigma H^T
  Eigen::MatrixXd D = quad_form(prior.Sigma_cond(), meas);  // H Sigma_.|S H^T
  add_noise(&A, meas);
  add_noise(&D, meas);

  double ldA = 0.0, ldD = 0.0;
  if (!logdet_spd(A, &ldA) || !logdet_spd(D, &ldD)) return g;

  g.delta = ldA - ldD;
  g.posterior_logdet = prior.prior_logdet() - g.delta;
  g.ok = true;
  return g;
}

// ---------------------------------------------------------------------------
Eigen::MatrixXd dense_jacobian(const std::vector<PredictedMeasurement>& meas,
                               int state_dim) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2 * meas.size(), state_dim);
  for (int i = 0; i < static_cast<int>(meas.size()); ++i)
    for (const auto& b : meas[i].blocks)
      H.block(2 * i, b.col, 2, b.J.cols()) += b.J;
  return H;
}

InformationGain evaluate_reference(const Eigen::MatrixXd& Sigma_in,
                                   const Eigen::MatrixXd& T,
                                   const std::vector<PredictedMeasurement>& meas) {
  InformationGain g;
  g.num_measurements = static_cast<int>(meas.size());
  const Eigen::MatrixXd Sigma = 0.5 * (Sigma_in + Sigma_in.transpose());
  const int n = static_cast<int>(Sigma.rows());

  Eigen::MatrixXd Sigma_S = T * Sigma * T.transpose();
  double ld_prior = 0.0;
  if (!logdet_spd(0.5 * (Sigma_S + Sigma_S.transpose()), &ld_prior)) return g;

  const Eigen::MatrixXd H = dense_jacobian(meas, n);
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(H.rows(), H.rows());
  for (int i = 0; i < static_cast<int>(meas.size()); ++i)
    V(2 * i, 2 * i) = V(2 * i + 1, 2 * i + 1) = meas[i].sigma * meas[i].sigma;

  const Eigen::MatrixXd A = H * Sigma * H.transpose() + V;
  const Eigen::MatrixXd K = Sigma * H.transpose() * A.inverse();
  // Joseph form for symmetry / PSD preservation.
  const Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(n, n) - K * H;
  Eigen::MatrixXd Sigma_post = IKH * Sigma * IKH.transpose() + K * V * K.transpose();

  Eigen::MatrixXd Sigma_S_post = T * Sigma_post * T.transpose();
  double ld_post = 0.0;
  if (!logdet_spd(0.5 * (Sigma_S_post + Sigma_S_post.transpose()), &ld_post)) return g;

  g.delta = ld_prior - ld_post;
  g.posterior_logdet = ld_post;
  g.ok = true;
  return g;
}

// ---------------------------------------------------------------------------
Eigen::MatrixXd augment_with_virtual_landmarks(const Eigen::MatrixXd& Sigma,
                                               int num_virtual, double sigma_l) {
  const int n = static_cast<int>(Sigma.rows());
  const int na = n + 3 * num_virtual;
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(na, na);
  out.topLeftCorner(n, n) = Sigma;
  out.bottomRightCorner(3 * num_virtual, 3 * num_virtual).diagonal().setConstant(
      sigma_l * sigma_l);
  return out;
}

Eigen::MatrixXd augment_scoring_map(const Eigen::MatrixXd& T, int old_state_dim,
                                    int num_virtual) {
  const int s = static_cast<int>(T.rows());
  const int va = 3 * num_virtual;
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(s + va, old_state_dim + va);
  out.topLeftCorner(s, old_state_dim) = T;
  out.bottomRightCorner(va, va).setIdentity();
  return out;
}

}  // namespace active_slam_information
