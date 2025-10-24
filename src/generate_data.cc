#include "include/generate_data.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <sophus/se3.hpp>

std::vector<Eigen::Vector3d> generate_points_on_sphere(const size_t num_samples,
                                                       unsigned int radius) {
  std::vector<Eigen::Vector3d> sample_points(num_samples);
  std::random_device rd;
  std::mt19937 gen(rd());
  auto dist = std::uniform_real_distribution<double>(-1, 1);
  for (auto& point : sample_points) {
    point.setZero();
    point.x() = dist(gen);
    point.y() = dist(gen);
    point.z() = dist(gen);
    point = point / point.norm() * radius;
  }
  return sample_points;
}

std::vector<Eigen::Isometry3d> generate_cam_pose_facing_origin(
    const size_t num_poses, const double mean_dist, const double std_dist) {
  std::vector<Eigen::Isometry3d> sample_poses;
  sample_poses.reserve(num_poses);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> distance_dist(mean_dist, std_dist);
  std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

  for (size_t i = 0; i < num_poses; ++i) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

    // Sample distance and angle on XY plane
    double distance = distance_dist(gen);
    if (distance <= 0) continue;

    double theta = angle_dist(gen);

    // Position on XY plane
    Eigen::Vector3d position(
        distance * std::cos(theta), distance * std::sin(theta), 0.0);
    pose.translation() = position;

    // Build rotation matrix to point toward origin
    // Z-axis: points from camera to origin
    Eigen::Vector3d z_axis = -position.normalized();

    // X-axis: perpendicular to Z in the XY plane
    // Use world Z-axis (0,0,1) cross with camera Z-axis
    Eigen::Vector3d x_axis = Eigen::Vector3d::UnitZ().cross(z_axis);
    x_axis.normalize();

    // Y-axis: complete right-handed coordinate system
    Eigen::Vector3d y_axis = z_axis.cross(x_axis);

    // Construct rotation matrix from basis vectors
    Eigen::Matrix3d rotation;
    rotation.col(0) = x_axis;
    rotation.col(1) = y_axis;
    rotation.col(2) = z_axis;

    pose.linear() = rotation;

    sample_poses.push_back(pose);
  }

  return sample_poses;
}

std::vector<Eigen::Vector2d> generate_pixels_from_points(
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::Matrix3d& K,
    const Eigen::Isometry3d& cam_T_world) {
  std::vector<Eigen::Vector2d> pixels(points.size());

  for (int i = 0; i < points.size(); i++) {
    const Eigen::Vector3d point_in_cam = cam_T_world * points[i];
    // Positive depth = in front of camera
    if (point_in_cam.z() <= 0)
      std::cout << "Warning: Point behind camera" << point_in_cam << std::endl;

    const double x_hom = point_in_cam.x() / point_in_cam.z();
    const double y_hom = point_in_cam.y() / point_in_cam.z();
    const Eigen::Vector3d pixel_hom = K * Eigen::Vector3d(x_hom, y_hom, 1);
    pixels[i] = {pixel_hom.x(), pixel_hom.y()};
  }
  return pixels;
}

std::vector<Eigen::Vector3d> generate_perturbed_points(
    std::vector<Eigen::Vector3d> points,
    const double noise_mean,
    const double noise_std) {
  std::random_device rd;
  std::mt19937 gen(rd());
  auto dist = std::normal_distribution<double>(noise_mean, noise_std);
  for (auto& point : points) {
    Eigen::Vector3d noise;
    noise.x() = dist(gen);
    noise.y() = dist(gen);
    noise.z() = dist(gen);
    point += noise;
  }
  return points;
}

std::vector<Eigen::Isometry3d> generate_perturbed_poses(
    std::vector<Eigen::Isometry3d> poses,
    const double noise_mean,
    const double noise_std) {
  std::random_device rd;
  std::mt19937 gen(rd());
  auto dist = std::normal_distribution<double>(noise_mean, noise_std);
  for (auto& pose : poses) {
    Eigen::Matrix<double, 6, 1> noise;
    noise << dist(gen), dist(gen), dist(gen), dist(gen), dist(gen), dist(gen);
    Sophus::SE3d pertubation = Sophus::SE3d::exp(noise);
    pose = pose * Eigen::Isometry3d(pertubation.matrix());
  }
  return poses;
}
