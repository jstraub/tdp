#ifndef SKINNING_H
#define SKINNING_H

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>

#include <iostream>
#include <complex>
#include <vector>
#include <Eigen/Eigenvalues>

void test_getMean(const tdp::ManagedHostImage<tdp::Vector3fda>& pc);
tdp::Vector3fda getMean(const tdp::ManagedHostImage<tdp::Vector3fda>& pc);
tdp::Matrix3fda getCovariance(const tdp::ManagedHostImage<tdp::Vector3fda>& pc);
tdp::ManagedHostImage<tdp::Vector3fda> getSimplePc();
std::vector<tdp::Vector3fda> getMeanAndSpread(const tdp::ManagedHostImage<tdp::Vector3fda>& pc);
inline bool inBVoxel(const tdp::Vector3fda& p, const tdp::Vector3fda& topLeft, const tdp::Vector3fda& btmRight);
inline bool hasNan(tdp::Vector3fda p);
std::vector<tdp::Vector3fda> meanAndSpreadOfBVoxel(const tdp::ManagedHostImage<tdp::Vector3fda>& pc, const tdp::Vector3fda& p1, const tdp::Vector3fda& p2);
std::vector<tdp::Vector3fda> getMeans(const tdp::ManagedHostImage<tdp::Vector3fda>& pc, int nsteps);


#endif // SKINNING_H
