#ifndef SKINNING_H
#define SKINNING_H

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>

#include <iostream>
#include <complex>
#include <vector>
#include <Eigen/Eigenvalues>

tdp::Vector3fda getMean(const tdp::Image<tdp::Vector3fda> &pc, const Eigen::VectorXi& nnIds);
tdp::Matrix3fda getCovariance(const tdp::Image<tdp::Vector3fda>& pc, const Eigen::VectorXi& nnIds);
tdp::ManagedHostImage<tdp::Vector3fda> GetSimplePc();
void GetSphericalPc(tdp::Image<tdp::Vector3fda>& pc);


#endif // SKINNING_H
