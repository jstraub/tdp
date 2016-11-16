#pragma once

#include <iostream>
#include <complex>
#include <vector>
#include <Eigen/Sparse>

#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>

#include <Eigen/Eigenvalues>

#include <pangolin/gl/gldraw.h>


#include <tdp/eigen/dense.h>
#include <tdp/eigen/std_vector.h>
#include <tdp/data/managed_image.h>
#include <tdp/nn/ann.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

Vector3fda getMean(const Image<Vector3fda> &pc, const Eigen::VectorXi& nnIds);

Matrix3fda getCovariance(const Image<Vector3fda>& pc,
                              const Eigen::VectorXi& nnIds);

ManagedHostImage<Vector3fda> GetSimplePc();


void GetSphericalPc(ManagedHostImage<Vector3fda>& pc);

void GetCylindricalPc(ManagedHostImage<Vector3fda>& pc);

void GetMtxPc(tdp::ManagedHostImage<Vector3fda>& pc, int w, int h, float step=0.1);

template<typename T>
inline void getAxesIds(const std::vector<T>& vec, std::vector<int>& sortIds){

    int hi(0), lo(0), mid;
    for (size_t i=0; i<vec.size(); ++i){
        if (vec[i] < vec[lo]){
            lo = i;
        } else if (vec[i] > vec[hi]){
            hi = i;
        }
    }

    for (size_t i=0; i<vec.size();++i){
        if (i!=hi&&i!=lo){
            mid=i;
        }
    }
    sortIds.push_back(hi);
    sortIds.push_back(mid);
    sortIds.push_back(lo);
}

Eigen::Matrix3f getLocalRot(const Matrix3fda& cov,
                            const Eigen::SelfAdjointEigenSolver<Matrix3fda>& es);

void getAllLocalBasis(const Image<Vector3fda>& pc, Image<SE3f>& locals,
                      ANN& ann, int knn, float eps);

inline float w(float d, int knn);

inline Vector6fda poly2Basis(const Vector3fda& p);

void getThetas(const Image<Vector3fda>& pc, const Image<SE3f>& T_wls,
               Image<Vector6fda>& thetas, ANN& ann, int knn, float eps);

void getZEstimates(const Image<Vector3fda>& pc, const Image<SE3f>& locals,
                   const Image<Vector6fda>& thetas, Image<Vector3fda>& estimates);

void getThetas_F(const Image<Vector3fda>& pc_w,const Image<SE3f>& T_wls,
                  const auto& f, Image<Vector6fda>& thetas, ANN& ann,
                 int knn, float eps);

void getFEstimates(const Image<Vector3fda>& pc_w,
                   const Image<SE3f>& T_wls,
                   const Image<Vector6fda>& thetas,
                   Image<Vector3fda>& estimates_w);

void  getSimpleLBEvector(Image<Vector3fda>& pc,
                         Eigen::VectorXf& evector_real,
                         ANN& ann,
                         const int knn,
                         const float eps,
                         float alpha,
                         int idEv);

Eigen::SparseMatrix<float> getLaplacian(Image<Vector3fda>& pc,
                                         ANN& ann,
                                         const int knn,
                                         const float eps,
                                         float alpha);

Eigen::VectorXf getLaplacianEvector(const Image<Vector3fda>& pc,
                         const Eigen::SparseMatrix<float>& L,
                         int idEv);

void getLaplacianEvectors(const Eigen::SparseMatrix<float>& L,
                          int numEv,
                          eigen_vector<Eigen::VectorXf>& evectors);

void getLaplacianBasis(const Eigen::SparseMatrix<float>& L,
                       int numEv,
                       Eigen::MatrixXf& basis);


Eigen::MatrixXf getMeanCurvature(const Image<Vector3fda>& pc,
                                 const Eigen::SparseMatrix<float>& L);


eigen_vector<Vector3fda> getLevelSetMeans(const Image<Vector3fda>& pc,
                                              const Eigen::VectorXf& evector,
                                              int nBins);
inline float rbf(const Vector3fda& p1,
                 const Vector3fda& p2,
                 const float alpha);


void f_rbf(const Image<Vector3fda>& pc,
                      const Vector3fda& p,
                      const float alpha,
           Eigen::VectorXf& f );

//std::vector<float> f_rbf(const Image<Vector3fda>& pc,
//           const Vector3fda& p,
//           const float alpha);

}
