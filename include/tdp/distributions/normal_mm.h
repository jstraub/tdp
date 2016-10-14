/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <tdp/eigen/dense.h>
#include <tdp/eigen/std_vector.h>
#include <tdp/distributions/normal.h>
#include <tdp/clustering/dpmeans.hpp>

namespace tdp {

 
template<typename T>
bool ComputeGMMfromPC(
    const Image<Vector3fda>& x,
    const Image<Vector3fda>& cuX,
    DPmeans& dpmeans,
    size_t maxIt, 
    float minNchangePerc,
    Image<uint16_t>& z,
    Image<uint16_t>& cuZ,
    std::vector<Normal<T,3>>& gmm) {
  gmm.clear();
  // Run the clustering algorithm.
  dpmeans.Compute(x, cuX, cuZ, maxIt, minNchangePerc);
  z.CopyFrom(cuZ, cudaMemcpyDeviceToHost);
//  eigen_vector<Vector3fda>& centers = dpmeans.centers_;
  std::vector<size_t> Ns = dpmeans.Ns_;
  uint32_t K = dpmeans.K_;

  eigen_vector<Eigen::Matrix<T,3,1>> xSum(K,Eigen::Matrix<T,3,1>::Zero());
  eigen_vector<Eigen::Matrix<T,3,3>> Ss(K,Eigen::Matrix<T,3,3>::Zero());
  eigen_vector<Eigen::Matrix<T,3,1>> mus(K,Eigen::Matrix<T,3,1>::Zero());
  std::vector<T> ws(K,0.f);
  T W = 0.f;
  // Compute Gaussian statistics: 
  for (uint32_t i=0; i<x.Area(); ++i) 
    if(z[i] < K) {
      // TODO have no weighting right now!
      T w = 1;
      xSum[z[i]] += x[i].cast<T>()*w;
      ws[z[i]] += w;
      W += w;
    }
  for(uint32_t k=0; k<K; ++k) mus[k] = xSum[k]/ws[k];

  for (uint32_t i=0; i<x.Area(); ++i) 
    if(z[i] < K) {
      //TODO
      float w = 1.f;
      Ss[z[i]] += w*(x[i].cast<T>()-mus[z[i]])*(x[i].cast<T>()-mus[z[i]]).transpose();
    }
  const float maxEvFactor = 1e-2f;
  for(uint32_t k=0; k<K; ++k)
    if (Ns[k] > 5) {
      T pi = ws[k]/W;
      Eigen::Matrix<T,3,3> cov = Ss[k]/ws[k];
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,3,3>> eig(cov);
      Eigen::Matrix<T,3,1> e = eig.eigenvalues();
      uint32_t iMax = 0;
      T eMax = e.maxCoeff(&iMax);
      bool regularized = false;
      for (uint32_t i=0; i<3; ++i)
        if (i!=iMax && eMax*maxEvFactor > e(i)) {
          std::cout << "small eigenvalue: " << e(i) << " replaced by " 
            << eMax*maxEvFactor << std::endl;
          e(i) = eMax*maxEvFactor;
          regularized = true;
        }
      if (regularized) {
        Eigen::Matrix<T,3,3> V = eig.eigenvectors();
        cov = V*e.asDiagonal()*V.inverse();
      }
      gmm.push_back(Normal<T,3>(mus[k], cov, pi));
    }
  return true;
}

}
