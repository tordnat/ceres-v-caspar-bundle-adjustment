#include <cstddef>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <random>
#include <utility>
#include <vector>

#include "ceres/solver.h"
#include "ceres_adapter.h"
#include "generated/solver.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/src/Core/EigenBase.h>
#include <Eigen/src/Core/GlobalFunctions.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/AngleAxis.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/Transform.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <ceres/types.h>
#include <solver_params.h>
#include <sophus/so3.hpp>
#include <stdio.h>

#define SPHERE_RADIUS 4
#define MEAN_POSE_DIST 4 * SPHERE_RADIUS

#define POINT_NOISE_MEAN 0.1
#define POINT_NOISE_STD 0.3

#define TRANSLATION_NOISE_MEAN 0.0
#define TRANSLATION_NOISE_STD 0.03

#define ROTATION_NOISE_MEAN 0.0
#define ROTATION_NOISE_STD 0.00

#define PIXEL_NOISE_MEAN 0.0
#define PIXEL_NOISE_STD 0  // 0.2  // 1-2 px typical, so this is low

std::vector<Eigen::Vector3d> generate_points_on_sphere(
    const size_t num_samples) {
  std::vector<Eigen::Vector3d> samples(num_samples);
  for (Eigen::Vector3d& pose : samples) {
    pose = Eigen::Vector3d::Random().normalized() * SPHERE_RADIUS;
  }
  return samples;
}

std::vector<Eigen::Isometry3d> generate_cam_pose(const size_t num_poses) {
  std::vector<Eigen::Isometry3d> sample_poses(num_poses);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist_distance(MEAN_POSE_DIST,
                                                 TRANSLATION_NOISE_STD);
  std::uniform_real_distribution<double> dist_dir(-1, 1);

  for (Eigen::Isometry3d& pose : sample_poses) {
    pose.setIdentity();

    Eigen::Vector3d cam_dir;
    cam_dir.x() = dist_dir(gen);
    cam_dir.y() = dist_dir(gen);
    cam_dir.normalize();
    cam_dir.z() = 0;
    pose.translation() = cam_dir * dist_distance(gen);

    Eigen::Vector3d forward =
        -cam_dir;  // Camera looks toward origin in XY plane
    forward.z() = 0;
    forward.normalize();

    double angle = std::atan2(forward.y(), forward.x());
    pose.linear() = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX())
                        .toRotationMatrix();
  }

  return sample_poses;
}

std::vector<Eigen::Vector3d> perturb_points(
    std::vector<Eigen::Vector3d> points) {
  std::random_device rd;
  std::mt19937 gen(rd());
  auto dist =
      std::normal_distribution<double>(POINT_NOISE_MEAN, POINT_NOISE_STD);
  for (auto& point : points) {
    auto noise = Eigen::Vector3d::Random().normalized() * dist(gen);
    point += noise;
  }
  return points;
}

std::vector<Eigen::Isometry3d> perturbe_poses(
    std::vector<Eigen::Isometry3d> poses) {
  std::random_device rd;
  std::mt19937 gen(rd());
  auto rot_dist =
      std::normal_distribution<double>(ROTATION_NOISE_MEAN, ROTATION_NOISE_STD);
  auto trans_dist = std::normal_distribution<double>(TRANSLATION_NOISE_MEAN,
                                                     TRANSLATION_NOISE_STD);
  for (auto& pose : poses) {
    Eigen::Vector3d rot_noise;
    rot_noise << rot_dist(gen), rot_dist(gen), rot_dist(gen);

    Eigen::Vector3d trans_noise;
    trans_noise << trans_dist(gen), trans_dist(gen), trans_dist(gen);

    Sophus::SO3<double> rot_pertubation =
        Sophus::SO3d::exp(rot_noise);  // Should be skew symmetric?

    pose.linear() = pose.linear() * rot_pertubation.matrix();
    pose.translation() += trans_noise;
  }
  return poses;
}

std::vector<Eigen::Vector2d> generate_pixels(
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::Matrix3d& K,
    const Eigen::Isometry3d& world_to_cam) {
  std::vector<Eigen::Vector2d> pixels(points.size());

  for (int i = 0; i < points.size(); ++i) {
    const Eigen::Vector3d point_in_cam = world_to_cam * points[i];
    if (point_in_cam.z() <= 0) {
      std::cout << "Warning, point behind camera: " << i << std::endl;
    }
    const double x_normalized = point_in_cam.x() / point_in_cam.z();
    const double y_normalized = point_in_cam.y() / point_in_cam.z();

    const Eigen::Vector3d pixel_homogeneous =
        K * Eigen::Vector3d(x_normalized, y_normalized, 1.0);
    pixels[i] = {pixel_homogeneous.x(), pixel_homogeneous.y()};
  }
  return pixels;
}

std::vector<Eigen::Vector2d> perturb_pixels(
    std::vector<Eigen::Vector2d> pixels) {
  std::random_device rd;
  std::mt19937 gen(rd());
  auto dist =
      std::normal_distribution<double>(PIXEL_NOISE_MEAN, PIXEL_NOISE_STD);
  for (auto& pixel : pixels) {
    const auto pertubation = Eigen::Vector2d::Random().normalized() * dist(gen);
    pixel += pertubation;
  }
  return pixels;
}

void exportToXYZ(const std::vector<Eigen::Vector3d>& points,
                 const std::string& filename) {
  std::ofstream file(filename);
  file << "x,y,z" << std::endl;
  for (const auto& point : points) {
    file << point.x() << "," << point.y() << "," << point.z() << std::endl;
  }
  file.close();
}

void exportPoseToBlender(const Eigen::Isometry3d& pose,
                         const std::string& filename) {
  std::ofstream file(filename);

  Eigen::AngleAxisd rot(M_PI, Eigen::Vector3d(0, 1, 0));
  const Eigen::Quaterniond quat(rot * pose.rotation());

  file << "x,y,z,R_w,R_x,R_y,R_z" << std::endl;
  file << std::fixed << std::setprecision(4) << pose.translation().x() << ","
       << pose.translation().y() << "," << pose.translation().z() << ","
       << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z()
       << std::endl;
  file.close();
}

// Ceres

struct SimpleReprojectionError {
  SimpleReprojectionError(double x_observed, double y_observed)
      : x_observed(x_observed), y_observed(y_observed) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point_world,
                  T* residuals) const {
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point_world, p);
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T x_pred = p[0] / p[2];
    T y_pred = p[1] / p[2];

    residuals[0] = x_pred - x_observed;
    residuals[1] = y_pred - y_observed;
    return true;
  }

  static ceres::CostFunction* Create(const double x_observed,
                                     const double y_observed) {
    return new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 6, 3>(
        new SimpleReprojectionError(x_observed, y_observed));
  }
  double x_observed;
  double y_observed;
};

int main() {
  printf("Testing Caspar \n");

  constexpr int num_points = 50;
  constexpr int num_poses = 2;
  constexpr int num_observations = num_poses * num_points;
  constexpr double epsilon_solver = 0.01;

  auto gt_points = generate_points_on_sphere(num_points);
  auto gt_poses = generate_cam_pose(num_poses);

  const auto cam1_pose_gt = gt_poses[0];
  const auto cam2_pose_gt = gt_poses[1];

  // Add after generating poses:
  for (int i = 0; i < 5; ++i) {  // Check first few poses
    Eigen::Vector3d cam_pos = gt_poses[i].translation();
    Eigen::Vector3d cam_dir = gt_poses[i].linear().col(2);  // Forward direction
    std::cout << "Camera " << i << ": pos=" << cam_pos.transpose()
              << ", dir=" << cam_dir.transpose() << std::endl;
  }

  auto initial_poses = perturbe_poses(gt_poses);
  auto cam1_pose_initial = initial_poses[0];
  auto cam2_pose_initial = initial_poses[1];

  auto initial_points = perturb_points(gt_points);
  auto pixels_cam1 = perturb_pixels(
      generate_pixels(gt_points, Eigen::Matrix3d::Identity(), cam1_pose_gt));
  auto pixels_cam2 = perturb_pixels(
      generate_pixels(gt_points, Eigen::Matrix3d::Identity(), cam2_pose_gt));

  std::unordered_map<size_t, Eigen::Vector2d> observationnr_to_pixel;
  std::unordered_map<size_t, size_t> observationnr_to_cam;
  std::vector<size_t> observationnr_to_point_index(num_observations);
  std::vector<Eigen::Matrix<double, 6, 1>> cam_params;

  for (int i = 0; i < num_poses; ++i) {
    Eigen::Isometry3d world_to_cam = initial_poses[i];

    Eigen::Matrix<double, 6, 1> params;
    params[3] = world_to_cam.translation().x();
    params[4] = world_to_cam.translation().y();
    params[5] = world_to_cam.translation().z();

    Eigen::AngleAxisd angle_axis(world_to_cam.rotation());
    Eigen::Vector3d axis_angle = angle_axis.angle() * angle_axis.axis();

    params[0] = axis_angle[0];
    params[1] = axis_angle[1];
    params[2] = axis_angle[2];

    cam_params.push_back(params);
  }

  for (int i = 0; i < num_points; ++i) {
    observationnr_to_cam[i] = 0;
    observationnr_to_cam[i + num_points] = 1;
    observationnr_to_pixel[i] = pixels_cam1[i];
    observationnr_to_pixel[i + num_points] = pixels_cam2[i];
    observationnr_to_point_index[i] = i;
    observationnr_to_point_index[i + num_points] = i;
  }

  exportToXYZ(gt_points, "sphere_points_gt.csv");
  exportToXYZ(initial_points, "sphere_points_noisy.csv");
  exportPoseToBlender(cam1_pose_gt, "cam1_pose_gt.csv");
  exportPoseToBlender(cam2_pose_gt, "cam2_pose_gt.csv");

  // Using Ceres Solver

  ceres::Problem ceres_problem;
  CeresToCasparAdapter adapter;
  caspar::SolverParams params;

  for (int i = 0; i < num_observations; ++i) {
    const auto& point_obs = observationnr_to_pixel[i];
    ceres::CostFunction* cost_function =
        SimpleReprojectionError::Create(point_obs.x(), point_obs.y());
    double pixel[2];
    pixel[0] = point_obs.x();
    pixel[1] = point_obs.y();
    ceres_problem.AddResidualBlock(
        cost_function,
        nullptr /* squared loss */,
        cam_params[observationnr_to_cam[i]].data(),
        initial_points[observationnr_to_point_index[i]].data());
    adapter.AddResidualBlock(
        CasparCostFnType::REPROJECTION_SIMPLE,
        CasparLossFnType::L2,
        pixel,
        cam_params[observationnr_to_cam[i]].data(),
        initial_points[observationnr_to_point_index[i]].data());
  }

  ceres_problem.SetParameterBlockConstant(cam_params[0].data());
  ceres_problem.SetParameterBlockConstant(initial_points[0].data());
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = 500;
  ceres_options.linear_solver_type = ceres::DENSE_SCHUR;
  ceres_options.minimizer_progress_to_stdout =
      true;  // Similar to logging=true in caspar
  ceres::Solver::Summary summary;

  ceres::Solve(ceres_options, &ceres_problem, &summary);
  exportToXYZ(initial_points, "sphere_points_solved.csv");
  std::cout << summary.FullReport() << std::endl;

  for (int i = 0; i < num_points; ++i) {
    double error = (initial_points[i] - gt_points[i]).norm();
    std::cout << "Point " << i << " error: " << error << std::endl;
  }

  std::cout << "Solving with Caspar" << std::endl;
  caspar::GraphSolver caspar_solver(
      params, num_poses * 100, num_points * 200, num_observations * 200);
  adapter.LoadToCaspar(caspar_solver);
  caspar_solver.solve(true);
  std::vector<float> caspar_solved_points(3 * num_points);
  std::vector<Eigen::Vector3d> caspar_solved_points_eig;
  caspar_solver.get_Point_nodes_to_stacked_host(
      caspar_solved_points.data(), 0, num_points);

  for (int i = 0; i < num_points; i++) {
    caspar_solved_points_eig.emplace_back(caspar_solved_points[i * 3],
                                          caspar_solved_points[i * 3 + 1],
                                          caspar_solved_points[i * 3 + 2]);
  }
  exportToXYZ(caspar_solved_points_eig, "sphere_points_solved_caspar.csv");
  return 0;
}