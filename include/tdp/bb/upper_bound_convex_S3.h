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

template <typename T>
class UpperBoundConvexS3 : public Bound<T,NodeS3<T>> {
 public:
  UpperBoundConvexS3(
      const std::vector<vMF<T,3>>& vmf_mm_A, 
      const std::vector<vMF<T,3>>& vmf_mm_B);
  virtual T Evaluate(const NodeS3<T>& node);
  virtual T EvaluateAndSet(NodeS3<T>& node);
  virtual T EvaluateRotationSet(const
      std::vector<Eigen::Quaternion<T>>& qs) const;
 private:
  const std::vector<vMF<T,3>>& vmf_mm_A_;
  const std::vector<vMF<T,3>>& vmf_mm_B_;
  static Eigen::Matrix<T,4,4> BuildM(const Eigen::Matrix<T,3,1>& u, const
    Eigen::Matrix<T,3,1>& v);
};

typedef UpperBoundConvexS3<float>  UpperBoundConvexS3f;
typedef UpperBoundConvexS3<double> UpperBoundConvexS3d;

template <typename T>
T FindMaximumQAQ(const Eigen::Matrix<T,4,4>& A, const
  Eigen::Matrix<T,4,Eigen::Dynamic> Q, bool verbose);

template <typename T>
T FindMaximumQAQ(const Eigen::Matrix<T,4,4>& A, const Tetrahedron4D<T>&
    tetrahedron, bool verbose);

template<typename T, uint32_t D>
bool FindLambda(const Eigen::Matrix<T,D,D>& A, const
    Eigen::Matrix<T,D,D>& B, T* lambda, bool verbose) {

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<T,D,D>>
    ges(A,B,Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
  
  Eigen::Matrix<T, D, 1> ev = ges.eigenvalues();
  Eigen::Matrix<T, D, D> V = ges.eigenvectors();

  if(verbose) {
    Eigen::Matrix<T,D,D> err = (A*V - B*V*ev.asDiagonal());
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

//  Eigen::LLT<Eigen::Matrix<T,D,D>> llt(B);
//  Eigen::Matrix<T,D,D> L = llt.matrixL();
//  Eigen::FullPivLU<Eigen::Matrix<T,D,D>> lu(L);
//
//  if (lu.rank() < D) {
//    std::cout << "FindLambda: cannot find eigen vector rank " 
//      << lu.rank() << " < " << D << std::endl;
//    return false;
//  }
//
//  Eigen::Matrix<T,D,D> A__ = lu.solve(A);
//  Eigen::Matrix<T,D,D> A_ = lu.solve(A__.transpose());
//
//  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,D,D>> es(A_, true);
//  Eigen::Matrix<T, D, 1> ev = es.eigenvalues();
////  if ((es.eigenvalues().imag().array() > 0.).any())
//
//  std::cout << "-- Find Lambda " << std::endl;
//  std::cout << A_ << std::endl;
////  std::cout << es.eigenvalues().transpose() << std::endl;
//
//  Eigen::FullPivLU<Eigen::Matrix<T,D,D>> lut(L.transpose());
//  if (lut.rank() < D) {
//    std::cout << "FindLambda: cannot find eigen vector rank " 
//      << lut.rank() << " < " << D << std::endl;
//    return false;
//  }
//  Eigen::Matrix<T, D, D> V = lut.solve(es.eigenvectors()); //.real();
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


template<typename T, uint32_t k> 
void ComputeLambdasOfSubset(
    const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& A, 
    const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& B, 
    const Eigen::Matrix<T,4,Eigen::Dynamic> Q,
    bool verbose,
    std::vector<T>& lambdas) {
  Combinations combNKs(Q.cols(),k);
  for (auto comb : combNKs.Get()) {
    Eigen::Matrix<T,k,k> A_; 
    Eigen::Matrix<T,k,k> B_;
    for (uint32_t i=0; i<k; ++i)
      for (uint32_t j=0; j<k; ++j) {
        A_(i,j) = A(comb[i],comb[j]);
        B_(i,j) = B(comb[i],comb[j]);
      }
    T lambda = 0.;
    if (FindLambda<k>(A_, B_, &lambda, verbose)) {
      lambdas.push_back(lambda);
      if(verbose) std::cout<<"lambda "<<k<<"x"<<k<<" "<< lambda << std::endl;
    }
  }
}

//template<uint32_t D>
//bool FindLambda(const Eigen::Matrix<T, D,D>& A, const
//    Eigen::Matrix<T, D,D>& B, T* lambda) {
//  Eigen::GeneralizedEigenSolver<Eigen::Matrix<T,D,D>> ges(A, B, true);
//  // eigenvalues are alphas/betas
//  if ((ges.betas().array().abs() > 1e-10).all()) {
////    std::cout << "FindLambda: non singular EVs" << std::endl;
//    uint32_t id_max = 0;
//    Eigen::Matrix<T, D, 1> ev = ges.eigenvalues().real();
//    T ev_max = ev.maxCoeff(&id_max);
////    Eigen::Matrix<T,D,1> alpha = ges.eigenvectors().col(id_max);
//    Eigen::FullPivLU<Eigen::Matrix<T,D,D>> qr(A-ev_max*B);
//    if (qr.rank() < D) {
////      std::cout << "FindLambda: cannot find eigen vector rank " << qr.rank() << " < " << D << std::endl;
//      return false;
//    }
////    std::cout << "FindLambda: can find eigen vector." << std::endl;
//    Eigen::Matrix<T,D,1> alpha = qr.solve(Eigen::Matrix<T,D,1>::Zero());
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
