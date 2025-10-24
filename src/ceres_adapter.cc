#include "include/ceres_adapter.h"

void CeresToCasparAdapter::LoadToCaspar(caspar::GraphSolver& solver) {
  // Points shared across all factors
  size_t num_points = points_.size();
  solver.set_Point_num(num_points);
  solver.set_Point_nodes_from_stacked_host(points_.data(), 0, num_points);
  // Load factors to Caspar
  CeresToCasparAdapter::LoadAllSimpleReprojectionFactorsFixedCam(solver);
  CeresToCasparAdapter::LoadAllSimpleReprojectionFactors(solver);
  solver.finish_indices();
}