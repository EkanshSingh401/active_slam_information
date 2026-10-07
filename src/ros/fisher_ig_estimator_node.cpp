// fisher_ig_estimator (C++) -- candidate viewpoints scored by D-optimal
// information gain on the metric map (PATCHES s61). Drop-in replacement for the
// Python placeholder in active_slam_planner: same node name, same output topic
// (~/scored_viewpoints, active_slam_msgs/ScoredViewpointArray), same candidate
// ring parameters, is_placeholder = false.
//
// Per cycle (rate):
//   prior   InformationPrior(Sigma, T_metric) from the latest JointCovariance:
//           T = IMU state (15) + global XYZ of every SLAM landmark whose
//           representation dependencies are in the message (MATH_TO_CODE.md).
//   cands   ring of num_candidates x candidate_z_offsets at candidate_radius
//           around the current IMU position, yaw facing back at it.
//   score   evaluate(prior, predicted measurements from both cameras);
//           ScoredViewpoint.score = -posterior_logdet (the selector maximizes;
//           ranking by posterior_logdet, not by delta), information_gain = delta.
// v1 approximations (MATH_TO_CODE.md): common prior, horizon 1, constant pixel
// noise = up_slam_sigma_px, no occlusion model (no ESDF yet: is_reachable=true,
// clearance=-1).

#include <cmath>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "active_slam_information/ros/joint_cov_problem.hpp"
#include "active_slam_msgs/msg/joint_covariance.hpp"
#include "active_slam_msgs/msg/scored_viewpoint_array.hpp"

namespace asi = active_slam_information;

class FisherIGEstimator : public rclcpp::Node {
 public:
  FisherIGEstimator() : Node("fisher_ig_estimator") {
    rate_ = declare_parameter("rate", 2.0);
    num_ = declare_parameter("num_candidates", 24);
    radius_ = declare_parameter("candidate_radius", 2.0);
    z_offsets_ = declare_parameter("candidate_z_offsets", std::vector<double>{0.0, 0.5});
    sigma_px_ = declare_parameter("sigma_px", 1.0);  // = OpenVINS up_slam_sigma_px
    max_depth_ = declare_parameter("max_depth", 20.0);
    width_ = declare_parameter("image_width", 848);
    height_ = declare_parameter("image_height", 480);
    sub_ = create_subscription<active_slam_msgs::msg::JointCovariance>(
        "/openvins/joint_covariance", 10, [this](active_slam_msgs::msg::JointCovariance::SharedPtr m) { last_ = m; });
    pub_ = create_publisher<active_slam_msgs::msg::ScoredViewpointArray>("~/scored_viewpoints", 10);
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / std::max(rate_, 1e-3)), [this] { tick(); });
    RCLCPP_INFO(get_logger(), "fisher_ig_estimator (C++, D-opt metric map) up: %d x %zu candidates, r=%.2f m, sigma=%.2f px",
                num_, z_offsets_.size(), radius_, sigma_px_);
  }

 private:
  void tick() {
    if (!last_) return;
    const auto P = asi::problem_from_msg(*last_, max_depth_, width_, height_);
    if (!P.ok) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "skip: %s", P.why.c_str()); return; }
    const asi::InformationPrior prior(P.Sigma, P.T_metric());
    if (!prior.ok()) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "skip: scored marginal not PD"); return; }
    const asi::ConstantNoiseModel noise(sigma_px_);

    active_slam_msgs::msg::ScoredViewpointArray out;
    out.header = last_->header;
    out.header.frame_id = "global";
    out.scoring_method = "D_opt_metric_map_posterior_logdet_v1";
    out.is_placeholder = false;
    out.covariance_trace = P.Sigma.trace();
    const Eigen::Vector3d c = P.current.p_IinG;
    out.reference_pose = to_pose(P.current);
    for (double dz : z_offsets_) {
      for (int i = 0; i < num_; ++i) {
        const double th = 2.0 * M_PI * i / std::max(num_, 1);
        asi::Pose cand;
        cand.p_IinG = c + Eigen::Vector3d(radius_ * std::cos(th), radius_ * std::sin(th), dz);
        const double yaw = std::atan2(c.y() - cand.p_IinG.y(), c.x() - cand.p_IinG.x());
        const Eigen::Matrix3d R_ItoG = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        cand.R_GtoI = R_ItoG.transpose();
        const auto g = asi::evaluate(prior, asi::predict_all(P, cand, noise));
        active_slam_msgs::msg::ScoredViewpoint vp;
        vp.pose = to_pose(cand);
        vp.information_gain = g.ok ? g.delta : 0.0;
        vp.score = g.ok ? -g.posterior_logdet : -1e300;
        vp.coverage_gain = 0.0;
        vp.travel_cost = (cand.p_IinG - c).norm();
        vp.is_reachable = true;  // no ESDF in v1
        vp.clearance = -1.0;
        out.viewpoints.push_back(vp);
      }
    }
    pub_->publish(out);
  }

  static geometry_msgs::msg::Pose to_pose(const asi::Pose& p) {
    geometry_msgs::msg::Pose o;
    o.position.x = p.p_IinG.x(); o.position.y = p.p_IinG.y(); o.position.z = p.p_IinG.z();
    const Eigen::Quaterniond q(Eigen::Matrix3d(p.R_GtoI.transpose()));  // Hamilton q_ItoG for ROS
    o.orientation.x = q.x(); o.orientation.y = q.y(); o.orientation.z = q.z(); o.orientation.w = q.w();
    return o;
  }

  double rate_, radius_, sigma_px_, max_depth_;
  int num_, width_, height_;
  std::vector<double> z_offsets_;
  active_slam_msgs::msg::JointCovariance::SharedPtr last_;
  rclcpp::Subscription<active_slam_msgs::msg::JointCovariance>::SharedPtr sub_;
  rclcpp::Publisher<active_slam_msgs::msg::ScoredViewpointArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FisherIGEstimator>());
  rclcpp::shutdown();
  return 0;
}
