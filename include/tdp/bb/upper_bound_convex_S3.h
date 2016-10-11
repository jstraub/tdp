/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tdp/bb/node_S3.h>
#include <tdp/bb/numeric_helpers.h>
#include <tdp/distributions/vmf.h>
#include <tdp/bb/bound.h>
#include <tdp/bb/upper_bound_indep_S3.h>

namespace tdp {

class UpperBoundConvexS3 : public Bound<NodeS3> {
 public:
  UpperBoundConvexS3(
      const std::vector<vMF3f>& vmf_mm_A, 
      const std::vector<vMF3f>& vmf_mm_B);
  virtual float Evaluate(const NodeS3& node);
  virtual float EvaluateAndSet(NodeS3& node);
  virtual float EvaluateRotationSet(const
      std::vector<Eigen::Quaternion<float>>& qs) const;
 private:
  const std::vector<vMF3f>& vmf_mm_A_;
  const std::vector<vMF3f>& vmf_mm_B_;
  static Eigen::Matrix<float,4,4> BuildM(const Eigen::Vector3f& u, const
    Eigen::Vector3f& v);
};

float FindMaximumQAQ(const Eigen::Matrix4f& A, const
  Eigen::Matrix<float,4,Eigen::Dynamic> Q, bool verbose);
float FindMaximumQAQ(const Eigen::Matrix4f& A, const Tetrahedron4D&
    tetrahedron, bool verbose);

template<uint32_t D>
bool FindLambda(const Eigen::Matrix<float, D,D>& A, const
    Eigen::Matrix<float, D,D>& B, float* lambda, bool verbose) {

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<float,D,D>>
    ges(A,B,Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
  
  Eigen::Matrix<float, D, 1> ev = ges.eigenvalues();
  Eigen::Matrix<float, D, D> V = ges.eigenvectors();

  if(verbose) {
    Eigen::Matrix<float,D,D> err = (A*V - B*V*ev.asDiagonal());
    if (err.norm() > 1e-6)
      std::cout << "D=" << D << " -------------" << std::endl;
      std::cout << "EVs " << ev.transpose() << std::endl;
      std::cout << V << std::endl;
      std::cout << "Error in GES " 
        << std::endl << err << std::endl
        << std::endl << A << std::endl
        << std::endl << B << std::endl;
//    std::cout << "VTBV:\n" << V.transpose()*B*V << std::endl;
  }

//  Eigen::LLT<Eigen::Matrix<float,D,D>> llt(B);
//  Eigen::Matrix<float,D,D> L = llt.matrixL();
//  Eigen::FullPivLU<Eigen::Matrix<float,D,D>> lu(L);
//
//  if (lu.rank() < D) {
//    std::cout << "FindLambda: cannot find eigen vector rank " 
//      << lu.rank() << " < " << D << std::endl;
//    return false;
//  }
//
//  Eigen::Matrix<float,D,D> A__ = lu.solve(A);
//  Eigen::Matrix<float,D,D> A_ = lu.solve(A__.transpose());
//
//  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,D,D>> es(A_, true);
//  Eigen::Matrix<float, D, 1> ev = es.eigenvalues();
////  if ((es.eigenvalues().imag().array() > 0.).any())
//
//  std::cout << "-- Find Lambda " << std::endl;
//  std::cout << A_ << std::endl;
////  std::cout << es.eigenvalues().transpose() << std::endl;
//
//  Eigen::FullPivLU<Eigen::Matrix<float,D,D>> lut(L.transpose());
//  if (lut.rank() < D) {
//    std::cout << "FindLambda: cannot find eigen vector rank " 
//      << lut.rank() << " < " << D << std::endl;
//    return false;
//  }
//  Eigen::Matrix<float, D, D> V = lut.solve(es.eigenvectors()); //.real();
////  std::cout << es.eigenvectors() << std::endl;
//  std::cout << es.eigenvectors().transpose() * es.eigenvectors() << std::endl;
////  std::cout << V << std::endl;
  
  // initialize original index locations
  std::vector<size_t> idx(ev.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&ev](size_t i1, size_t i2) {return ev(i1) > ev(i2);});
  
  for (auto i : idx) {
    if ((V.col(i).array() >= 0.).all() 
      || (V.col(i).array() <= 0.).all()) {
//      std::cout << "found ev with v all same sign:"
//<< ev(i) << "\tv: " << V.col(i).transpose() << std::endl;
      *lambda = ev(i);
      return true; 
    }
  } 
  return false;
}


template<uint32_t k> 
void ComputeLambdasOfSubset(const Eigen::MatrixXf& A, const
    Eigen::MatrixXf& B, const Eigen::Matrix<float,4,Eigen::Dynamic> Q,
    bool verbose,
    std::vector<float>& lambdas) {
  Combinations combNKs(Q.cols(),k);
  for (auto comb : combNKs.Get()) {
    Eigen::Matrix<float,k,k> A_; 
    Eigen::Matrix<float,k,k> B_;
    for (uint32_t i=0; i<k; ++i)
      for (uint32_t j=0; j<k; ++j) {
        A_(i,j) = A(comb[i],comb[j]);
        B_(i,j) = B(comb[i],comb[j]);
      }
    float lambda = 0.;
    if (FindLambda<k>(A_, B_, &lambda, verbose)) {
      lambdas.push_back(lambda);
      if(verbose) std::cout<<"lambda "<<k<<"x"<<k<<" "<< lambda << std::endl;
    }
  }
}

//template<uint32_t D>
//bool FindLambda(const Eigen::Matrix<float, D,D>& A, const
//    Eigen::Matrix<float, D,D>& B, float* lambda) {
//  Eigen::GeneralizedEigenSolver<Eigen::Matrix<float,D,D>> ges(A, B, true);
//  // eigenvalues are alphas/betas
//  if ((ges.betas().array().abs() > 1e-10).all()) {
////    std::cout << "FindLambda: non singular EVs" << std::endl;
//    uint32_t id_max = 0;
//    Eigen::Matrix<float, D, 1> ev = ges.eigenvalues().real();
//    float ev_max = ev.maxCoeff(&id_max);
////    Eigen::Matrix<float,D,1> alpha = ges.eigenvectors().col(id_max);
//    Eigen::FullPivLU<Eigen::Matrix<float,D,D>> qr(A-ev_max*B);
//    if (qr.rank() < D) {
////      std::cout << "FindLambda: cannot find eigen vector rank " << qr.rank() << " < " << D << std::endl;
//      return false;
//    }
////    std::cout << "FindLambda: can find eigen vector." << std::endl;
//    Eigen::Matrix<float,D,1> alpha = qr.solve(Eigen::Matrix<float,D,1>::Zero());
////    std::cout << "FindLambda: alphas = " << alpha.transpose() << std::endl;
//    if ((alpha.array() >= 0.).all() || (alpha.array() <= 0.).all()) {
////      std::cout << "FindLambda: lambda = " << ev_max << std::endl;
//      *lambda = ev_max;
//      return true; 
//    }
////  } else {
////    std::cout << "FindLambda: betas are to small: " << ges.betas().transpose() << std::endl;
//  }
//  return false;
//}

}
