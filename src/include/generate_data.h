#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

std::vector<Eigen::Vector3d> generate_points_on_sphere(const size_t num_samples,
                                                       unsigned int radius);
std::vector<Eigen::Isometry3d> generate_cam_pose_facing_origin(
    const size_t num_poses, const double mean_dist, const double std_dist);
std::vector<Eigen::Vector2d> generate_pixels_from_points(
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::Matrix3d& K,
    const Eigen::Isometry3d& cam_T_world);

std::vector<Eigen::Vector3d> generate_perturbed_points(
    std::vector<Eigen::Vector3d> points,
    const double noise_mean,
    const double noise_std);
std::vector<Eigen::Isometry3d> generate_perturbed_poses(
    std::vector<Eigen::Isometry3d> poses,
    const double noise_mean,
    const double noise_std);