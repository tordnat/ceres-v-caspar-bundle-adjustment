#include <cstddef>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <random>
#include <utility>
#include <vector>

#include "../generated/solver.h"
#include "ceres/solver.h"
#include "include/ceres_adapter.h"
#include "include/generate_data.h"
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

// Logging

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
  const Eigen::Quaterniond quat(pose.rotation());

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
  bool operator()(const T* const camera_pose,
                  const T* const point_world,
                  T* residuals) const {
    T q[4] = {camera_pose[3], camera_pose[0], camera_pose[1], camera_pose[2]};
    T p[3];
    ceres::UnitQuaternionRotatePoint(q, point_world, p);
    p[0] += camera_pose[4];
    p[1] += camera_pose[5];
    p[2] += camera_pose[6];

    T x_pred = p[0] / p[2];
    T y_pred = p[1] / p[2];

    residuals[0] = x_pred - x_observed;
    residuals[1] = y_pred - y_observed;
    return true;
  }

  static ceres::CostFunction* Create(const double x_observed,
                                     const double y_observed) {
    return new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 7, 3>(
        new SimpleReprojectionError(x_observed, y_observed));
  }
  double x_observed;
  double y_observed;
};

bool is_point_visible(const Eigen::Vector3d point_world,
                      const Eigen::Isometry3d cam_T_world) {
  auto point_dir = point_world.normalized();
  auto cam_dir = cam_T_world.translation().normalized();
  return point_dir.dot(cam_dir) > 0;
}

int main() {
  printf("Comparing Ceres with Caspar \n");

  constexpr int num_points = 100;
  constexpr int num_poses = 10;
  int num_observations = 0;
  constexpr double epsilon_solver = 0.01;

  constexpr int sphere_radius = 4;
  constexpr int mean_cam_distance = 4 * sphere_radius;
  constexpr float std_cam_distance = 0.1;
  constexpr double point_noise_mean = 0.3;
  constexpr double point_noise_std = 0.5;
  constexpr double pose_noise_mean = 0.0;
  constexpr double pose_noise_std = 0.0;

  const auto gt_points = generate_points_on_sphere(num_points, sphere_radius);
  const auto gt_poses = generate_cam_pose_facing_origin(
      num_poses, mean_cam_distance, std_cam_distance);

  std::unordered_map<size_t, std::vector<Eigen::Vector2d>> cam_num_to_pixels;
  for (int i = 0; i < gt_poses.size(); i++) {
    auto cam_T_world = gt_poses[i].inverse();
    cam_num_to_pixels[i] = generate_pixels_from_points(
        gt_points, Eigen::Matrix3d::Identity(), cam_T_world);
  }

  // Initial poses and point estimates
  auto initial_poses =
      generate_perturbed_poses(gt_poses, pose_noise_mean, pose_noise_std);
  auto initial_points =
      generate_perturbed_points(gt_points, point_noise_mean, point_noise_std);

  std::unordered_map<size_t, size_t> observationnr_to_pixel_idx;
  std::unordered_map<size_t, size_t> observationnr_to_cam_idx;
  std::unordered_map<size_t, size_t> observationnr_to_point_idx;
  std::vector<Eigen::Matrix<double, 7, 1>> cam_params_ceres;
  std::vector<Eigen::Matrix<double, 7, 1>> cam_params_caspar;

  // The solvers expect the following memory layout for the cam parameters
  // cam: r_x, r_y, r_z, r_w, x, y, z
  for (int i = 0; i < num_poses; ++i) {
    Eigen::Isometry3d cam_T_world = initial_poses[i].inverse();
    Eigen::Quaterniond quat(cam_T_world.rotation());
    Eigen::Matrix<double, 7, 1> params;
    params[0] = quat.x();
    params[1] = quat.y();
    params[2] = quat.z();
    params[3] = quat.w();
    params[4] = cam_T_world.translation().x();
    params[5] = cam_T_world.translation().y();
    params[6] = cam_T_world.translation().z();

    cam_params_caspar.push_back(params);
    cam_params_ceres.push_back(params);
  }

  for (int cam_num = 0; cam_num < num_poses; cam_num++) {
    for (int point_num = 0; point_num < num_points; point_num++) {
      auto wold_T_cam = gt_poses[cam_num];  // we only use translation
      auto point_world = gt_points[point_num];
      if (is_point_visible(point_world, wold_T_cam)) {
        size_t observation_nr = num_observations;
        num_observations++;
        observationnr_to_cam_idx[observation_nr] = cam_num;
        observationnr_to_pixel_idx[observation_nr] = point_num;
        observationnr_to_point_idx[observation_nr] = point_num;
      }
    }
  }

  exportToXYZ(gt_points, "sphere_points_gt.csv");
  exportToXYZ(initial_points, "sphere_points_noisy.csv");
  exportPoseToBlender(gt_poses[0], "cam1_pose_gt.csv");
  exportPoseToBlender(gt_poses[1], "cam2_pose_gt.csv");

  ceres::Problem ceres_problem;
  CeresToCasparAdapter adapter;
  caspar::SolverParams params;

  for (int i = 0; i < num_observations; ++i) {
    size_t cam_idx = observationnr_to_cam_idx[i];
    size_t point_idx = observationnr_to_point_idx[i];
    size_t pixel_idx = observationnr_to_point_idx[i];
    const auto& point_observed = initial_points[point_idx];
    const auto& pixel_observed = cam_num_to_pixels[cam_idx][pixel_idx];
    double pixel[] = {pixel_observed.x(), pixel_observed.y()};
    ceres::CostFunction* cost_function =
        SimpleReprojectionError::Create(pixel_observed.x(), pixel_observed.y());
    ceres_problem.AddResidualBlock(cost_function,
                                   nullptr /* squared loss */,
                                   cam_params_ceres[cam_idx].data(),
                                   initial_points[point_idx].data());
    adapter.AddResidualBlock(CasparCostFnType::REPROJECTION_SIMPLE,
                             CasparLossFnType::L2,
                             pixel,
                             cam_params_caspar[cam_idx].data(),
                             initial_points[point_idx].data());
  }

  // Ceres Priors
  ceres_problem.SetParameterBlockConstant(cam_params_ceres[0].data());

  // Caspar Priors

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
  params.solver_iter_max = 500;
  caspar::GraphSolver caspar_solver(
      params, num_poses, num_points, num_observations, 1, 1);
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

  for (int i = 0; i < num_points; ++i) {
    double error = (caspar_solved_points_eig[i] - gt_points[i]).norm();
    std::cout << "Point " << i << " error: " << error << std::endl;
  }
  exportToXYZ(caspar_solved_points_eig, "sphere_points_solved_caspar.csv");
  return 0;
}