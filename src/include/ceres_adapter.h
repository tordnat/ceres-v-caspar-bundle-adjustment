#pragma once

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../../generated/solver.h"

enum class CasparCostFnType {
  REPROJECTION_SIMPLE,
  REPROJECTION,
  POSE_PRIOR,
  DISTANCE_PRIOR,
  UNKNOWN
};
enum class CasparLossFnType { L2, UNKNOWN };

class CeresToCasparAdapter {
 public:
  CeresToCasparAdapter() = default;

  // Caspar does not support the default cost functions of Ceres, so we have to
  // pass observations as a separate argument
  template <typename... Ts>
  bool AddResidualBlock(const CasparCostFnType& cost_fn,
                        const CasparLossFnType& loss_fn,
                        double* observed_values,
                        double* x0,
                        Ts... xs) {
    if (loss_fn != CasparLossFnType::L2) {
      std::cerr << "AddResidualBlock failed: Unknown or unsupported loss "
                   "function type."
                << std::endl;
      return false;
    }
    if (cost_fn == CasparCostFnType::REPROJECTION_SIMPLE) {
      if constexpr (sizeof...(Ts) == 1) {
        AddSimpleReprojectionFactor(observed_values, x0, xs...);
      } else {
        std::cerr << "REPROJECTION_SIMPLE expects 1 additional parameter"
                  << std::endl;
        return false;
      }
    } else if (cost_fn == CasparCostFnType::POSE_PRIOR) {
      if constexpr (sizeof...(Ts) == 1) {
        AddPosePrior(x0, xs...);
      } else {
        std::cerr << "POSITION_PRIOR expects 1 additional parameter"
                  << std::endl;
        return false;
      }
    } else if (cost_fn == CasparCostFnType::DISTANCE_PRIOR) {
      if constexpr (sizeof...(Ts) == 2) {
        AddDistancePrior(x0, xs...);
      } else {
        std::cerr << "DISTANCE_PRIOR expects 2 additional parameters"
                  << std::endl;
        return false;
      }
    } else {
      std::cerr << "Unknown or unsupported cost function type" << std::endl;
      return false;
    }
    return true;
  }

  void AddResidualBlockFixedCam(const CasparCostFnType& cost_fn,
                                const CasparLossFnType& loss_fn,
                                double* observed_values,
                                double* cam,
                                double* point) {}

  void LoadToCaspar(caspar::GraphSolver& solver);

 private:
  void AddSimpleReprojectionFactor(double* const pixel,
                                   double* const cam_ptr,
                                   double* const point_ptr) {
    int cam_idx = AddOrGetSimpleReprojectionCam(cam_ptr);
    int point_idx = AddOrGetSimpleReprojectionPoint(point_ptr);

    simple_reproj_pixels_.push_back(pixel[0]);
    simple_reproj_pixels_.push_back(pixel[1]);
    // Add the indices for this observation
    simple_reproj_cam_idx_.push_back(cam_idx);
    simple_reproj_point_idx_.push_back(point_idx);
    simple_reproj_count_++;
  }

  unsigned int AddOrGetSimpleReprojectionCam(double* const cam) {
    auto cam_it = simple_reproj_camera_idx_map_.find(cam);
    if (cam_it != simple_reproj_camera_idx_map_.end()) {
      return cam_it->second;
    }
    int cam_idx = simple_reproj_camera_idx_map_.size();
    simple_reproj_camera_idx_map_[cam] = cam_idx;

    for (int i = 0; i < SIMPLE_REPROJ_CAM_SIZE; i++) {
      simple_reproj_cam_.push_back(static_cast<float>(cam[i]));
    }
    return cam_idx;
  }

  unsigned int AddOrGetSimpleReprojectionPoint(double* const point) {
    auto point_it = simple_reproj_point_idx_map_.find(point);
    if (point_it != simple_reproj_point_idx_map_.end()) {
      return point_it->second;
    }
    int point_idx = simple_reproj_point_idx_map_.size();
    simple_reproj_point_idx_map_[point] = point_idx;
    for (int i = 0; i < POINT_SIZE; i++) {
      points_.push_back(static_cast<float>(point[i]));
    }
    return point_idx;
  }

  void LoadAllSimpleReprojectionFactors(caspar::GraphSolver& solver) {
    std::cout << "Loading problem data to caspar GPU" << std::endl;
    if (simple_reproj_count_ < 1) {
      std::cout << "No SimpleReprojection factors to load" << std::endl;
      return;
    };
    size_t num_cameras = simple_reproj_camera_idx_map_.size();

    // Tunable Camera parameters
    solver.set_PinholeIdeal_num(num_cameras);
    solver.set_PinholeIdeal_nodes_from_stacked_host(
        simple_reproj_cam_.data(), 0, num_cameras);

    // Number of factors
    solver.set_simple_pinhole_num(simple_reproj_count_);
    // Pixel data
    solver.set_simple_pinhole_pixel_data_from_stacked_host(
        simple_reproj_pixels_.data(), 0, simple_reproj_count_);
    // Cam idx
    solver.set_simple_pinhole_cam_indices_from_host(
        simple_reproj_cam_idx_.data(), simple_reproj_count_);
    // Point idx
    solver.set_simple_pinhole_point_indices_from_host(
        simple_reproj_point_idx_.data(), simple_reproj_count_);

    std::cout << "Finished loading data to Caspar" << std::endl;
  }

  void LoadAllSimpleReprojectionFactorsFixedCam(caspar::GraphSolver& solver) {
    std::cout << "Loading problem data to caspar GPU" << std::endl;
    if (simple_reproj_fixed_cam_count_ < 1) {
      std::cout << "No SimpleReprojectionFixedCam factors to load" << std::endl;
      return;
    };
    size_t num_cameras = simple_reproj_fixed_cam_camera_idx_map_.size();
    size_t num_points = simple_reproj_fixed_cam_point_idx_.size();
    size_t num_pixels = simple_reproj_fixed_cam_point_idx_.size();

    // Number of factors
    solver.set_simple_pinhole_fixed_cam_num(num_cameras);
    solver.set_simple_pinhole_fixed_cam_cam_data_from_stacked_host(
        simple_reproj_fixed_cam_cam_.data(), 0, num_cameras);
    solver.set_simple_pinhole_fixed_cam_pixel_data_from_stacked_device(
        simple_reproj_fixed_cam_pixels_.data(), 0, num_pixels);

    // Indices
    solver.set_simple_pinhole_fixed_cam_point_indices_from_host(
        simple_reproj_fixed_cam_cam_idx_.data(), num_cameras);
    solver.set_simple_pinhole_fixed_cam_point_indices_from_host(
        simple_reproj_fixed_cam_point_idx_.data(), num_points);

    std::cout << "Finished loading data to Caspar" << std::endl;
  }

  size_t num_observations_ = 0;

  static constexpr int PIXEL_SIZE = 2;
  static constexpr int POINT_SIZE = 3;
  static constexpr int POSITION_SIZE = 3;
  static constexpr int POSE_SIZE = 7;  // wxyz, xyz

  // Points to be refined by all factors
  std::vector<float> points_;

  // Simple Reprojection Factor
  size_t simple_reproj_count_ = 0;
  std::unordered_map<double*, unsigned int> simple_reproj_camera_idx_map_;
  std::unordered_map<double*, unsigned int> simple_reproj_point_idx_map_;
  std::vector<unsigned int> simple_reproj_cam_idx_;
  std::vector<unsigned int> simple_reproj_point_idx_;

  std::vector<float> simple_reproj_pixels_;
  std::vector<float> simple_reproj_cam_;

  // Simple Reprojection Factor Fixed Cam, points shared with
  size_t simple_reproj_fixed_cam_count_ = 0;
  std::unordered_map<double*, unsigned int>
      simple_reproj_fixed_cam_camera_idx_map_;
  std::unordered_map<double*, unsigned int>
      simple_reproj_fixed_cam_point_idx_map_;
  std::vector<unsigned int> simple_reproj_fixed_cam_cam_idx_;
  std::vector<unsigned int> simple_reproj_fixed_cam_point_idx_;

  std::vector<float> simple_reproj_fixed_cam_pixels_;
  std::vector<float> simple_reproj_fixed_cam_cam_;

  static constexpr int SIMPLE_REPROJ_CAM_SIZE = 7;
};