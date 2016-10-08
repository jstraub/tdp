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

bool ComputeGMMfromPC(
    const Image<Vector3fda>& x,
    const Image<Vector3fda>& cuX,
    DPmeans& dpmeans,
    size_t maxIt, 
    float minNchangePerc,
    Image<uint16_t>& z,
    Image<uint16_t>& cuZ,
    std::vector<Normal<float,3>>& gmm) {
  // Run the clustering algorithm.
  dpmeans.Compute(x, cuX, cuZ, maxIt, minNchangePerc);
  z.CopyFrom(cuZ, cudaMemcpyDeviceToHost);
  eigen_vector<Vector3fda>& centers = dpmeans.centers_;
  std::vector<size_t> Ns = dpmeans.Ns_;
  uint32_t K = dpmeans.K_;
  eigen_vector<Eigen::Vector3f> xSum(K,Eigen::Vector3f::Zero());
  eigen_vector<Eigen::Matrix3f> Ss(K,Eigen::Matrix3f::Zero());
  eigen_vector<Eigen::Vector3f> mus(K,Eigen::Vector3f::Zero());
  std::vector<float> ws(K,0.f);
  float W = 0.f;
  // Compute Gaussian statistics: 
  for (uint32_t i=0; i<x.Area(); ++i) 
    if(z[i] < K) {
      // TODO have no weighting right now!
      float w = 1;
      xSum[z[i]] += x[i]*w;
      ws[z[i]] += w;
      W += w;
    }
  for(uint32_t k=0; k<K; ++k) mus[k] = xSum[k]/ws[k];

  for (uint32_t i=0; i<x.Area(); ++i) 
    if(z[i] < K) {
      //TODO
      float w = 1.f;
      Ss[z[i]] += w*(x[i]-mus[z[i]])*(x[i]-mus[z[i]]).transpose();
    }
  const float maxEvFactor = 1e-2f;
  for(uint32_t k=0; k<K; ++k)
    if (Ns[k] > 5) {
      float pi = ws[k]/W;
      Eigen::Matrix3f cov = Ss[k]/ws[k];
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
      Eigen::Vector3f e = eig.eigenvalues();
      uint32_t iMax = 0;
      float eMax = e.maxCoeff(&iMax);
      bool regularized = false;
      for (uint32_t i=0; i<3; ++i)
        if (i!=iMax && eMax*maxEvFactor > e(i)) {
          std::cout << "small eigenvalue: " << e(i) << " replaced by " 
            << eMax*maxEvFactor << std::endl;
          e(i) = eMax*maxEvFactor;
          regularized = true;
        }
      if (regularized) {
        Eigen::Matrix3f V = eig.eigenvectors();
        cov = V*e.asDiagonal()*V.inverse();
      }
      gmm.push_back(Normal3f(mus[k], cov, pi));
    }
  return true;
}

}
