#pragma once

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "generated/solver.h"

enum class CasparCostFnType {
  REPROJECTION_SIMPLE,
  REPROJECTION,
  POSITION_PRIOR,
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
    } else if (cost_fn == CasparCostFnType::POSITION_PRIOR) {
      if constexpr (sizeof...(Ts) == 1) {
        AddPositionPrior(x0, xs...);
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
      simple_reproj_point_.push_back(static_cast<float>(point[i]));
    }
    return point_idx;
  }

  void LoadAllSimpleReprojectionFactors(caspar::GraphSolver& solver) {
    std::cout << "Loading problem data to caspar GPU" << std::endl;
    if (simple_reproj_count_ < 1) {
      std::cout << "No SimpleReprojection factors to load" << std::endl;
      return;
    };
    size_t num_points = simple_reproj_point_idx_map_.size();
    size_t num_cameras = simple_reproj_camera_idx_map_.size();

    solver.set_PinholeIdeal_num(num_cameras);
    solver.set_PinholeIdeal_nodes_from_stacked_host(
        simple_reproj_cam_.data(), 0, num_cameras);
    solver.set_Point_nodes_from_stacked_host(
        simple_reproj_point_.data(), 0, num_points);

    solver.set_simple_pinhole_num(simple_reproj_count_);
    // Indices
    solver.set_simple_pinhole_cam_indices_from_host(
        simple_reproj_cam_idx_.data(), simple_reproj_count_);
    solver.set_simple_pinhole_point_indices_from_host(
        simple_reproj_point_idx_.data(), simple_reproj_count_);
    // Pixel data
    solver.set_simple_pinhole_pixel_data_from_stacked_host(
        simple_reproj_pixels_.data(), 0, simple_reproj_count_);
    std::cout << "Finished loading data to Caspar" << std::endl;
  }

  void AddPositionPrior(double* cam, double* position_anchor) {
    auto cam_idx = AddOrGetSimpleReprojectionCam(cam);
    for (int i = 0; i < POSITION_SIZE; i++) {
      position_prior_positions_.push_back(position_anchor[0]);
    }
    position_prior_cam_idx_.push_back(cam_idx);
    num_position_priors_++;
  }

  void LoadAllPositionPriors(caspar::GraphSolver& solver) {
    if (num_position_priors_ < 1) {
      std::cout << "No position priors to be added" << std::endl;
      return;
    }
    std::cout << "Loading position priors" << std::endl;

    solver.set_position_prior_num(num_position_priors_);
    solver.set_position_prior_position_anchor_data_from_stacked_host(
        position_prior_positions_.data(), 0, num_position_priors_);
    solver.set_position_prior_cam1_indices_from_host(
        position_prior_cam_idx_.data(), num_position_priors_);
  }

  void AddDistancePrior(double* cam1,
                        double* cam2,
                        double* distance_constraint) {
    auto cam1_idx = AddOrGetSimpleReprojectionCam(cam1);
    auto cam2_idx = AddOrGetSimpleReprojectionCam(cam2);
    distance_prior_distances_.push_back(*distance_constraint);
    distance_prior_cam1_idx_.push_back(cam1_idx);
    distance_prior_cam2_idx_.push_back(cam2_idx);
    num_distance_priors_++;
  }

  void LoadAllDistancePriors(caspar::GraphSolver& solver) {
    if (num_distance_priors_ < 1) {
      std::cout << "No distance priors to be added" << std::endl;
      return;
    }
    std::cout << "Loading distance priors" << std::endl;
    solver.set_distance_prior_num(num_distance_priors_);
    solver.set_distance_prior_dist_data_from_stacked_host(
        distance_prior_distances_.data(), 0, num_distance_priors_);
    solver.set_distance_prior_cam1_indices_from_host(
        distance_prior_cam1_idx_.data(), num_distance_priors_);
    solver.set_distance_prior_cam2_indices_from_host(
        distance_prior_cam2_idx_.data(), num_distance_priors_);
    std::cout << "Done loading distance priors" << std::endl;
  }

  size_t num_observations_ = 0;

  static constexpr int PIXEL_SIZE = 2;
  static constexpr int POINT_SIZE = 3;
  static constexpr int POSITION_SIZE = 3;

  // Simple Reprojection Factor
  size_t simple_reproj_count_ = 0;
  std::unordered_map<double*, unsigned int> simple_reproj_camera_idx_map_;
  std::unordered_map<double*, unsigned int> simple_reproj_point_idx_map_;
  std::vector<unsigned int> simple_reproj_cam_idx_;
  std::vector<unsigned int> simple_reproj_point_idx_;

  std::vector<float> simple_reproj_pixels_;
  std::vector<float> simple_reproj_cam_;
  std::vector<float> simple_reproj_point_;

  static constexpr int SIMPLE_REPROJ_CAM_SIZE = 6;

  // Position prior
  size_t num_position_priors_ = 0;
  std::vector<float> position_prior_positions_;
  std::vector<unsigned int> position_prior_cam_idx_;

  // Distance prior
  size_t num_distance_priors_ = 0;
  std::vector<float> distance_prior_distances_;
  std::vector<unsigned int> distance_prior_cam1_idx_;
  std::vector<unsigned int> distance_prior_cam2_idx_;
};