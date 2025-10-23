#include "ceres_adapter.h"

void CeresToCasparAdapter::LoadToCaspar(caspar::GraphSolver& solver) {
  CeresToCasparAdapter::LoadAllSimpleReprojectionFactors(solver);
  solver.finish_indices();
}