// Tests for information_gain + view_geometry.
// Synthetic state reproduces the OpenVINS layout AND the measured structural
// degeneracy: the newest clone is a bit-exact copy of the IMU pose, giving a
// rank deficit of exactly 6 (diagnostic, results/moving_camera).

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include "active_slam_information/information_gain.hpp"
#include "active_slam_information/view_geometry.hpp"

using namespace active_slam_information;
using Eigen::MatrixXd;
using Eigen::Vector3d;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                  \
  do {                                                    \
    if (cond) { ++g_pass; }                               \
    else { ++g_fail; std::printf("  FAIL: " __VA_ARGS__); std::printf("\n"); } \
  } while (0)

// ---------------------------------------------------------------------------
// Synthetic OpenVINS-like state
//   IMU 15 | calib_dt 1 | 11 clones x 6 | N landmarks x 3
// ---------------------------------------------------------------------------
struct SynthState {
  static constexpr int kImu = 0, kCalib = 15, kClone0 = 16, kNumClones = 11;
  int n_landmarks;
  int n;
  MatrixXd Sigma;
  std::vector<LandmarkLinearization> landmarks;  // real, anchored rep
  std::vector<int> lambda_col, anchor_col;
  int newest_clone_col() const { return kClone0 + 6 * (kNumClones - 1); }
  int landmark_col(int j) const { return kClone0 + 6 * kNumClones + 3 * j; }
};

static MatrixXd rand_spd(int d, std::mt19937& rng, const Eigen::VectorXd& scale) {
  std::normal_distribution<double> N(0, 1);
  MatrixXd A(d, d);
  for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) A(i, j) = N(rng);
  MatrixXd S = A * A.transpose() / d + 0.05 * MatrixXd::Identity(d, d);
  return scale.asDiagonal() * S * scale.asDiagonal();
}

static SynthState make_state(int n_landmarks, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-1, 1);
  SynthState s;
  s.n_landmarks = n_landmarks;
  s.n = SynthState::kClone0 + 6 * SynthState::kNumClones + 3 * n_landmarks;

  // Independent state = everything except the newest clone.
  const int ni = s.n - 6;
  Eigen::VectorXd scale(ni);
  for (int i = 0; i < ni; ++i) scale(i) = 0.02;          // default
  for (int i = 0; i < 6; ++i) scale(i) = 0.01;           // IMU pose
  scale(15) = 1e-3;                                      // time offset
  const MatrixXd Sigma0 = rand_spd(ni, rng, scale);

  // E : independent -> full.  Newest clone rows copy IMU pose rows.
  MatrixXd E = MatrixXd::Zero(s.n, ni);
  const int nc = s.newest_clone_col();
  for (int r = 0, c = 0; r < s.n; ++r) {
    if (r >= nc && r < nc + 6) continue;
    E(r, c++) = 1.0;
  }
  for (int k = 0; k < 6; ++k) E(nc + k, k) = 1.0;
  s.Sigma = E * Sigma0 * E.transpose();

  // Landmarks in front of an identity-pose camera, anchored to random clones
  // (including, sometimes, the newest).
  std::uniform_int_distribution<int> clone_pick(0, SynthState::kNumClones - 1);
  for (int j = 0; j < n_landmarks; ++j) {
    LandmarkLinearization lm;
    lm.id = j;
    lm.p_FinG = Vector3d(1.5 * U(rng), 0.8 * U(rng), 2.5 + 1.5 * U(rng));
    const int lcol = s.landmark_col(j);
    const int acol = SynthState::kClone0 + 6 * clone_pick(rng);
    Eigen::Matrix3d Jl = Eigen::Matrix3d::Identity() + 0.3 * Eigen::Matrix3d::Random();
    Eigen::Matrix<double, 3, 6> Ja = 0.5 * Eigen::Matrix<double, 3, 6>::Random();
    lm.rep = {{lcol, Jl}, {acol, Ja}};
    s.landmarks.push_back(lm);
    s.lambda_col.push_back(lcol);
    s.anchor_col.push_back(acol);
  }
  return s;
}

// Scoring map: selection of IMU state + landmark parameter blocks.
static MatrixXd T_selection(const SynthState& s) {
  const int sd = 15 + 3 * s.n_landmarks;
  MatrixXd T = MatrixXd::Zero(sd, s.n);
  T.block(0, 0, 15, 15).setIdentity();
  for (int j = 0; j < s.n_landmarks; ++j)
    T.block(15 + 3 * j, s.landmark_col(j), 3, 3).setIdentity();
  return T;
}

// Scoring map: IMU state + landmark positions in global XYZ (metric map).
static MatrixXd T_metric(const SynthState& s) {
  const int sd = 15 + 3 * s.n_landmarks;
  MatrixXd T = MatrixXd::Zero(sd, s.n);
  T.block(0, 0, 15, 15).setIdentity();
  for (int j = 0; j < s.n_landmarks; ++j)
    for (const auto& r : s.landmarks[j].rep)
      T.block(15 + 3 * j, r.col, 3, r.J.cols()) += r.J;
  return T;
}

static std::vector<PredictedMeasurement> measurements_from(
    const SynthState& s, const Pose& cur, const Pose& cand, double sigma_px) {
  CameraModel cam;
  ConstantNoiseModel noise(sigma_px);
  std::vector<PredictedMeasurement> out;
  PredictedMeasurement pm;
  for (const auto& lm : s.landmarks)
    if (predict_measurement(cur, cand, 0, cam, lm, noise, nullptr, &pm)) out.push_back(pm);
  return out;
}

static Pose displaced(const Pose& p, const Vector3d& rotvec, const Vector3d& dp_G) {
  Pose c;
  const double a = rotvec.norm();
  const Eigen::Matrix3d Rd = a > 0 ? Eigen::AngleAxisd(a, rotvec / a).toRotationMatrix()
                                   : Eigen::Matrix3d::Identity();
  c.R_GtoI = Rd * p.R_GtoI;
  c.p_IinG = p.p_IinG + dp_G;
  return c;
}

static double rel(double a, double b) { return std::abs(a - b) / std::max(1.0, std::abs(b)); }

// ===========================================================================
int main() {
  std::printf("== structural degeneracy ==\n");
  {
    const SynthState s = make_state(40, 1);
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(s.Sigma);
    const double emax = es.eigenvalues().maxCoeff();
    int deficit = 0;
    for (int i = 0; i < es.eigenvalues().size(); ++i)
      if (es.eigenvalues()(i) < 1e-12 * emax) ++deficit;
    CHECK(deficit == 6, "rank deficit = %d, expected 6", deficit);
    Eigen::LLT<MatrixXd> llt(s.Sigma);
    CHECK(llt.info() != Eigen::Success, "full Sigma unexpectedly Cholesky-factorable");

    // Scoring both the IMU pose and its bit-exact clone must be rejected.
    MatrixXd Tbad = MatrixXd::Zero(12, s.n);
    Tbad.block(0, 0, 6, 6).setIdentity();
    Tbad.block(6, s.newest_clone_col(), 6, 6).setIdentity();
    CHECK(!InformationPrior(s.Sigma, Tbad).ok(), "degenerate scoring map was accepted");

    CHECK(InformationPrior(s.Sigma, T_selection(s)).ok(), "selection map rejected");
    CHECK(InformationPrior(s.Sigma, T_metric(s)).ok(), "metric (XYZ) map rejected");
    std::printf("  deficit %d; IMU+newest-clone scoring correctly rejected\n", deficit);
  }

  std::printf("== fast path == reference (singular Sigma) ==\n");
  {
    double worst = 0;
    for (unsigned seed = 1; seed <= 20; ++seed) {
      const SynthState s = make_state(30, seed);
      Pose cur;
      const Pose cand = displaced(cur, Vector3d(0.05, -0.1, 0.03), Vector3d(0.2, 0.0, -0.1));
      const auto meas = measurements_from(s, cur, cand, 1.0);
      for (const MatrixXd& T : {T_selection(s), T_metric(s)}) {
        const InformationGain fast = evaluate(InformationPrior(s.Sigma, T), meas);
        const InformationGain ref = evaluate_reference(s.Sigma, T, meas);
        CHECK(fast.ok && ref.ok, "seed %u: evaluation failed", seed);
        const double e = rel(fast.delta, ref.delta);
        worst = std::max(worst, e);
        CHECK(e < 1e-7, "seed %u: fast %.12g vs ref %.12g", seed, fast.delta, ref.delta);
        CHECK(rel(fast.posterior_logdet, ref.posterior_logdet) < 1e-7,
              "seed %u: posterior logdet mismatch", seed);
      }
    }
    std::printf("  40 cases, worst relative error %.2e\n", worst);
  }

  std::printf("== invariance to reparameterizing the scored quantity ==\n");
  {
    const SynthState s = make_state(25, 7);
    Pose cur;
    const auto meas = measurements_from(s, cur, displaced(cur, Vector3d(0, 0.08, 0), Vector3d(0.1, 0, 0)), 1.0);
    const MatrixXd T = T_metric(s);
    // K: random invertible, rescales units (radians vs metres vs inverse depth)
    Eigen::VectorXd d = Eigen::VectorXd::Random(T.rows()).array().abs() * 100 + 0.01;
    const MatrixXd K = d.asDiagonal() * (MatrixXd::Identity(T.rows(), T.rows()) +
                                         0.1 * MatrixXd::Random(T.rows(), T.rows()));
    const double a = evaluate(InformationPrior(s.Sigma, T), meas).delta;
    const double b = evaluate(InformationPrior(s.Sigma, K * T), meas).delta;
    CHECK(rel(a, b) < 1e-7, "gain changed under T -> K T: %.12g vs %.12g", a, b);
    std::printf("  dI = %.9f  vs  %.9f after unit change\n", a, b);
  }

  std::printf("== monotonicity ==\n");
  {
    const SynthState s = make_state(40, 3);
    Pose cur;
    const auto all = measurements_from(s, cur, displaced(cur, Vector3d(0, 0, 0.05), Vector3d(0, 0.1, 0)), 1.0);
    const InformationPrior prior(s.Sigma, T_metric(s));
    double prev = 0;
    bool mono = true, nonneg = true;
    for (size_t k = 1; k <= all.size(); ++k) {
      const double g = evaluate(prior, {all.begin(), all.begin() + k}).delta;
      if (g < -1e-12) nonneg = false;
      if (g < prev - 1e-9) mono = false;
      prev = g;
    }
    CHECK(nonneg, "negative information gain");
    CHECK(mono, "gain decreased when adding a measurement");

    double last = 1e300;
    bool dec = true;
    for (double sp : {0.5, 1.0, 2.0, 4.0, 8.0}) {
      const double g = evaluate(prior, measurements_from(s, cur, cur, sp)).delta;
      if (g > last + 1e-12) dec = false;
      last = g;
    }
    CHECK(dec, "gain did not decrease with measurement noise");
    std::printf("  %zu obs: gain non-negative, monotone in #obs, decreasing in sigma\n", all.size());
  }

  std::printf("== virtual frontier landmarks ==\n");
  {
    const SynthState s = make_state(20, 5);
    const int nv = 5;
    const MatrixXd T = augment_scoring_map(T_metric(s), s.n, nv);
    std::vector<LandmarkLinearization> vl;
    for (int k = 0; k < nv; ++k) {
      LandmarkLinearization lm;
      lm.is_virtual = true;
      lm.p_FinG = Vector3d(-0.8 + 0.4 * k, 0.1, 4.0);
      lm.rep = {{s.n + 3 * k, Eigen::Matrix3d::Identity()}};
      vl.push_back(lm);
    }
    Pose cur;
    CameraModel cam;
    ConstantNoiseModel noise(1.0);
    double last = -1;
    bool inc = true;
    for (double sl : {0.1, 0.3, 1.0, 3.0}) {
      const MatrixXd Sa = augment_with_virtual_landmarks(s.Sigma, nv, sl);
      std::vector<PredictedMeasurement> meas;
      PredictedMeasurement pm;
      for (const auto& lm : vl)
        if (predict_measurement(cur, cur, 0, cam, lm, noise, nullptr, &pm)) meas.push_back(pm);
      const double g = evaluate(InformationPrior(Sa, T), meas).delta;
      if (g < last) inc = false;
      last = g;
    }
    CHECK(inc, "frontier pull did not increase with sigma_l");
    std::printf("  frontier gain increases with declared frontier uncertainty sigma_l\n");
  }

  std::printf("== virtual landmark gain vs sigma_l, closed form ==\n");
  {
    // One virtual landmark with prior sigma_l^2 I_3, uncorrelated with everything,
    // scored ALONE, one 2D measurement with Jacobian H (2x3) and noise s^2 I_2:
    //   delta = log det(I_2 + sigma_l^2 H R^-1 H^T),  R = s^2 I_2 + H_x Sigma_x H_x^T
    // where H_x is the measurement's block on the (uncertain) current pose: the pose
    // uncertainty is effective measurement noise for the scored landmark.
    // Checked over 7 decades of sigma_l, including the crossover sigma_l ~ s / |H|
    // below which the virtual landmark contributes almost nothing.
    const SynthState s = make_state(10, 3);
    Pose cur;
    CameraModel cam;
    LandmarkLinearization lm;
    lm.is_virtual = true;
    lm.p_FinG = Vector3d(0.3, -0.2, 4.0);
    lm.rep = {{s.n, Eigen::Matrix3d::Identity()}};
    ConstantNoiseModel noise(1.0);
    PredictedMeasurement pm;
    CHECK(predict_measurement(cur, cur, 0, cam, lm, noise, nullptr, &pm), "virtual landmark not visible");
    MatrixXd H, Hx;
    int cx = -1;
    for (const auto& b : pm.blocks) {
      if (b.col == s.n) H = b.J; else { Hx = b.J; cx = b.col; }
    }
    const Eigen::Matrix2d R = pm.sigma * pm.sigma * Eigen::Matrix2d::Identity() +
                              Hx * s.Sigma.block(cx, cx, Hx.cols(), Hx.cols()) * Hx.transpose();
    double prev = -1;
    bool mono = true, closed = true;
    for (double sl : {1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0}) {
      const MatrixXd Sa = augment_with_virtual_landmarks(s.Sigma, 1, sl);
      MatrixXd T = MatrixXd::Zero(3, Sa.cols());
      T.block(0, s.n, 3, 3).setIdentity();
      const double d = evaluate(InformationPrior(Sa, T), {pm}).delta;
      const double ref = std::log((Eigen::Matrix2d::Identity() + sl * sl * H * H.transpose() * R.inverse()).determinant());
      if (rel(d, ref) > 1e-6 && std::fabs(d - ref) > 1e-9) closed = false;
      if (d <= prev) mono = false;
      std::printf("  sigma_l %-7g gain %.6f nats (closed form %.6f)\n", sl, d, ref);
      prev = d;
    }
    CHECK(closed, "virtual-landmark gain does not match log det(I + sigma_l^2 H H^T / s^2)");
    CHECK(mono, "virtual-landmark gain not increasing in sigma_l");
  }

  std::printf("== pose-marginal objective: T = IMU pose (6), real + virtual landmarks ==\n");
  {
    // dI_pose scores only the 6-dof IMU pose; every landmark (real and virtual) is
    // marginalized out. Fast path (Sylvester) vs brute-force full Kalman update.
    double worst = 0;
    for (unsigned seed = 1; seed <= 20; ++seed) {
      const SynthState s = make_state(20 + seed % 7, seed);
      const int nv = 4;
      const MatrixXd Sa = augment_with_virtual_landmarks(s.Sigma, nv, 0.5);
      MatrixXd T = MatrixXd::Zero(6, Sa.cols());
      T.block(0, 0, 6, 6).setIdentity();
      Pose cur, cand;
      cand.p_IinG = Vector3d(0.2, -0.1, 0.05);
      CameraModel cam;
      ConstantNoiseModel noise(1.0);
      std::vector<PredictedMeasurement> meas;
      for (const auto& lm : s.landmarks) {
        PredictedMeasurement pm;
        if (predict_measurement(cur, cand, 0, cam, lm, noise, nullptr, &pm)) meas.push_back(pm);
      }
      for (int k = 0; k < nv; ++k) {
        LandmarkLinearization v;
        v.is_virtual = true;
        v.p_FinG = Vector3d(-0.6 + 0.4 * k, 0.2, 3.5);
        v.rep = {{s.n + 3 * k, Eigen::Matrix3d::Identity()}};
        PredictedMeasurement pm;
        if (predict_measurement(cur, cand, 0, cam, v, noise, nullptr, &pm)) meas.push_back(pm);
      }
      const InformationGain fast = evaluate(InformationPrior(Sa, T), meas);
      const InformationGain ref = evaluate_reference(Sa, T, meas);
      CHECK(fast.ok && ref.ok, "seed %u: pose-marginal evaluation failed", seed);
      worst = std::max(worst, std::fabs(fast.delta - ref.delta) / std::max(1.0, std::fabs(ref.delta)));
    }
    CHECK(worst < 1e-7, "pose-marginal fast vs reference rel err %.3g", worst);
    std::printf("  worst rel err %.2e over 20 states (pose block only, landmarks marginalized)\n", worst);
  }

  std::printf("== Jacobians vs central finite differences ==\n");
  {
    // Measurement as a function of [dth_now(3), dp_now(3), dp_F(3)], using the
    // exact JPL perturbation R = Exp(-dth) R_hat and the candidate defined as
    // a fixed global-frame displacement of the (perturbed) current pose.
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> U(-1, 1);
    CameraModel cam;
    cam.R_ItoC = Eigen::AngleAxisd(0.02, Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    cam.p_IinC = Vector3d(0.031, -0.006, -0.021);  // D455-like offset
    double worst = 0;
    int tested = 0;
    for (int trial = 0; trial < 50; ++trial) {
      Pose cur;
      cur.R_GtoI = Eigen::AngleAxisd(0.3 * U(rng), Vector3d(U(rng), U(rng), U(rng)).normalized())
                       .toRotationMatrix();
      cur.p_IinG = Vector3d(U(rng), U(rng), U(rng));
      const Vector3d rv(0.2 * U(rng), 0.2 * U(rng), 0.2 * U(rng));
      const Vector3d dpG(0.3 * U(rng), 0.3 * U(rng), 0.3 * U(rng));
      const Pose cand = displaced(cur, rv, dpG);
      const Eigen::Matrix3d R_delta = cand.R_GtoI * cur.R_GtoI.transpose();

      LandmarkLinearization lm;
      lm.p_FinG = cand.p_IinG + cand.R_GtoI.transpose() *
                                    (cam.R_ItoC.transpose() * (Vector3d(0.4 * U(rng), 0.3 * U(rng), 3.0) - cam.p_IinC));
      lm.rep = {{6, Eigen::Matrix3d::Identity()}};  // p_F at cols 6..8 for this test

      PredictedMeasurement pm;
      ConstantNoiseModel noise(1.0);
      if (!predict_measurement(cur, cand, 0, cam, lm, noise, nullptr, &pm)) continue;
      const MatrixXd H = dense_jacobian({pm}, 9);

      auto z_of = [&](const Eigen::Matrix<double, 9, 1>& d) {
        const Vector3d dth = d.segment<3>(0);
        const double a = dth.norm();
        const Eigen::Matrix3d Rp = a > 0 ? Eigen::AngleAxisd(-a, dth / a).toRotationMatrix()
                                         : Eigen::Matrix3d::Identity();
        Pose c2;
        c2.R_GtoI = R_delta * (Rp * cur.R_GtoI);
        c2.p_IinG = (cur.p_IinG + d.segment<3>(3)) + dpG;
        return project_normalized(c2, cam, lm.p_FinG + d.segment<3>(6));
      };

      Eigen::Matrix<double, 2, 9> Hn;
      const double eps = 1e-6;
      for (int k = 0; k < 9; ++k) {
        Eigen::Matrix<double, 9, 1> d = Eigen::Matrix<double, 9, 1>::Zero();
        d(k) = eps;
        const Eigen::Vector2d zp = z_of(d);
        d(k) = -eps;
        Hn.col(k) = (zp - z_of(d)) / (2 * eps);
      }
      const double e = (H - Hn).cwiseAbs().maxCoeff() / std::max(1e-9, Hn.cwiseAbs().maxCoeff());
      worst = std::max(worst, e);
      ++tested;
    }
    CHECK(tested > 30, "too few visible FD trials (%d)", tested);
    CHECK(worst < 1e-6, "analytic vs numeric Jacobian relative error %.2e", worst);
    std::printf("  %d random poses/extrinsics, worst relative error %.2e\n", tested, worst);
  }

  std::printf("== timing (realistic sizes) ==\n");
  {
    const int n_lm = 46, n_virtual = 10, n_cand = 50;
    const SynthState s = make_state(n_lm, 9);
    const MatrixXd Sa = augment_with_virtual_landmarks(s.Sigma, n_virtual, 1.0);
    const MatrixXd T = augment_scoring_map(T_metric(s), s.n, n_virtual);
    std::vector<LandmarkLinearization> lms = s.landmarks;
    for (int k = 0; k < n_virtual; ++k) {
      LandmarkLinearization lm;
      lm.is_virtual = true;
      lm.p_FinG = Vector3d(-1 + 0.2 * k, 0.2, 4.5);
      lm.rep = {{s.n + 3 * k, Eigen::Matrix3d::Identity()}};
      lms.push_back(lm);
    }
    std::vector<Pose> cands;
    Pose cur;
    std::mt19937 rng(4);
    std::uniform_real_distribution<double> U(-1, 1);
    for (int c = 0; c < n_cand; ++c)
      cands.push_back(displaced(cur, Vector3d(0.1 * U(rng), 0.25 * U(rng), 0.05 * U(rng)),
                                Vector3d(0.3 * U(rng), 0.1 * U(rng), 0.3 * U(rng))));
    CameraModel cam;
    ConstantNoiseModel noise(1.0);

    const auto t0 = std::chrono::steady_clock::now();
    const InformationPrior prior(Sa, T);
    const auto t1 = std::chrono::steady_clock::now();
    const auto scores = score_candidates(prior, cur, cands, 0, cam, lms, noise);
    const auto t2 = std::chrono::steady_clock::now();

    int vis = 0;
    for (const auto& sc : scores) vis += sc.visible;
    const double ms_prior = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_score = std::chrono::duration<double, std::milli>(t2 - t1).count();
    CHECK(prior.ok(), "prior failed at realistic size");
    CHECK(scores.front().gain.ok, "best candidate failed");
    std::printf("  state n=%d, scored s=%d, %d candidates, mean %.1f visible\n",
                prior.state_dim(), prior.scored_dim(), n_cand, double(vis) / n_cand);
    std::printf("  prior (once/cycle): %.2f ms   scoring: %.2f ms total, %.3f ms/candidate\n",
                ms_prior, ms_score, ms_score / n_cand);
    std::printf("  best candidate %d: dI = %.3f nats\n", scores.front().index,
                scores.front().gain.delta);
  }

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
