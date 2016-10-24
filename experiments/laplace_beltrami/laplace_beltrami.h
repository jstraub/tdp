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
inline void getAxesIds(const std::vector<auto>& vec, std::vector<int>& axesIds);
Eigen::Matrix3f getLocalBasis(const tdp::Matrix3fda& cov, const Eigen::SelfAdjointEigenSolver<tdp::Matrix3fda>& es);
void getAllLocalBasis(const tdp::Image<tdp::Vector3fda>& pc, tdp::Image<tdp::Matrix3fda>& locals, tdp::ANN& ann, int knn, float eps);
inline float w(float d, int knn);
inline tdp::Vector6fda poly2Basis(const tdp::Vector2fda& vec);
void getThetas(const tdp::Image<tdp::Vector3fda>& pc, const tdp::Image<tdp::Matrix3fda>& locals, tdp::Image<tdp::Vector6fda>& thetas, tdp::ANN& ann, int knn, float eps);
void getZEstimates(const tdp::Image<tdp::Vector3fda>& pc, const tdp::Image<tdp::Matrix3fda>& locals, const tdp::Image<tdp::Vector6fda>& thetas, tdp::Image<tdp::Vector3fda>& estimates);

//tests
void test0();
void test1();
void test_getAxesIds();

#endif // SKINNING_H
