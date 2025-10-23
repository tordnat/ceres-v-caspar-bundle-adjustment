#include "ceres_adapter.h"

void CeresToCasparAdapter::LoadToCaspar(caspar::GraphSolver& solver) {
  CeresToCasparAdapter::LoadAllSimpleReprojectionFactors(solver);
  CeresToCasparAdapter::LoadAllPositionPriors(solver);
  CeresToCasparAdapter::LoadAllDistancePriors(solver);
  solver.finish_indices();
}