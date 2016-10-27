#ifndef SKINNING_H
#define SKINNING_H

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>
#include <tdp/manifold/SE3.h>

#include <iostream>
#include <complex>
#include <vector>
#include <Eigen/Eigenvalues>

tdp::Vector3fda getMean(const tdp::Image<tdp::Vector3fda> &pc, const Eigen::VectorXi& nnIds);
tdp::Matrix3fda getCovariance(const tdp::Image<tdp::Vector3fda>& pc, const Eigen::VectorXi& nnIds);
tdp::ManagedHostImage<tdp::Vector3fda> GetSimplePc();
void GetSphericalPc(tdp::Image<tdp::Vector3fda>& pc);
inline void getAxesIds(const std::vector<auto>& vec, std::vector<int>& axesIds);
Eigen::Matrix3f getLocalRot(const tdp::Matrix3fda& cov, const Eigen::SelfAdjointEigenSolver<tdp::Matrix3fda>& es);
void getAllLocalBasis(const tdp::Image<tdp::Vector3fda>& pc, tdp::Image<tdp::SE3f>& locals, tdp::ANN& ann, int knn, float eps);
inline float w(float d, int knn);
inline tdp::Vector6fda poly2Basis(const tdp::Vector3fda& p);
void getThetas(const tdp::Image<tdp::Vector3fda>& pc, const tdp::Image<tdp::SE3f>& locals, tdp::Image<tdp::Vector6fda>& thetas, tdp::ANN& ann, int knn, float eps);
void getZEstimates(const tdp::Image<tdp::Vector3fda>& pc, const tdp::Image<tdp::SE3f>& locals, const tdp::Image<tdp::Vector6fda>& thetas, tdp::Image<tdp::Vector3fda>& estimates);

void getThetas_F(const tdp::Image<tdp::Vector3fda>& pc_w,
               const tdp::Image<tdp::SE3f>& T_wls, tdp::Image<tdp::Vector6fda>& thetas,
               const auto& f, tdp::ANN& ann, int knn, float eps);

void getFEstimates(const tdp::Image<tdp::Vector3fda>& pc_w,
                   const tdp::Image<tdp::SE3f>& T_wl,
                   const tdp::Image<tdp::Vector6fda>& thetas,
                   tdp::Image<tdp::Vector3fda>& estimates_w);

//tests
void test_meanAndCov();
void test_getAllLocalBasis();
void test_getAxesIds();
void test_getLocalRot();
void test_getThetas_F();



#endif // SKINNING_H
