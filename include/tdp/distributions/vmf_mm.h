/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <tdp/eigen/dense.h>
#include <tdp/eigen/std_vector.h>
#include <tdp/distributions/vmf.h>
#include <tdp/clustering/dpvmfmeans.hpp>

namespace tdp {

bool ComputevMFMM(
    const Image<Vector3fda>& n,
    const Image<Vector3fda>& cuN,
    DPvMFmeans& dpvmfmeans,
    size_t maxIt, 
    float minNchangePerc,
    Image<uint16_t>& z,
    Image<uint16_t>& cuZ,
    std::vector<vMF<float,3>>& vmfs) {
  // Run the clustering algorithm.
  dpvmfmeans.Compute(n, cuN, cuZ, maxIt, minNchangePerc);
  z.CopyFrom(cuZ, cudaMemcpyDeviceToHost);
  eigen_vector<Vector3fda>& centers = dpvmfmeans.centers_;
  std::vector<size_t> Ns = dpvmfmeans.Ns_;
  uint32_t K = dpvmfmeans.K_;
  eigen_vector<Eigen::Vector3f> xSum(K,Eigen::Vector3f::Zero());
  std::vector<float> ws(K,0.f);
  float W = 0.f;
  for (uint32_t i=0; i<n.Area(); ++i) 
    if(z[i] < K) {
      // TODO: have no weighting right now
      // Compute vMF statistics: area-weighted sum over surface normals
      // associated with respective cluster. 
      float w = 1.f;
      xSum[z[i]] += n[i]*w;
      ws[z[i]] += w;
      W += w;
    }
  for(uint32_t k=0; k<K; ++k) {
    if (Ns[k] > 5) {
      float pi = ws[k]/W;
      float tau = vMF3f::MLEstimateTau(xSum[k],xSum[k]/xSum[k].norm(),ws[k]);
      vmfs.push_back(vMF3f(xSum[k]/xSum[k].norm(),tau,pi));
    }
  return true;
}


}
