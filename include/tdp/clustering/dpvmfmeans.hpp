/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <tdp/eigen/dense.h>
#include <tdp/managed_image.h>
#include <tdp/image.h>
#include <tdp/sufficientStats.h>
#include <tdp/Stopwatch.h>

namespace tdp {

uint32_t dpvMFlabelsOptimistic( 
    Image<Vector3fda> n,
    Image<Vector3fda> mu,
    Image<uint16_t> z,
    float lambda, uint32_t i0, uint16_t K);

class DPvMFmeans {
 public: 
  DPvMFmeans(float lambda) : lambda_(lambda) {};
  ~DPvMFmeans() {};

  float Compute(const Image<Vector3fda>& n, const Image<Vector3fda>& cuN, 
      Image<uint16_t>& cuZ, size_t maxIt, float minNchangePerc);

  uint16_t K_;
  float lambda_;
  std::vector<Vector3fda> centers_;
  std::vector<size_t> Ns_;
 private:

  void UpdateLabels(
      const Image<Vector3fda>& n, 
      const Image<Vector3fda>& cuN, 
      Image<uint16_t>& cuZ
      );
  void UpdateCenters(
      const Image<Vector3fda>& cuN, 
      const Image<uint16_t>& cuZ
      );

  uint32_t optimisticLabelsAssign(
      const Image<Vector3fda>& cuN, 
      Image<Vector3fda>& cuCenters,
      Image<uint16_t>& cuZ, uint32_t i0
    );


  Eigen::Matrix<float,4,Eigen::Dynamic> computeSS(
      const Image<Vector3fda>& cuN, 
      const Image<uint16_t>& cuZ);

  uint16_t indOfClosestCluster(const Vector3fda& ni, float& sim_closest);
};

float DPvMFmeans::Compute(const Image<Vector3fda>& n, 
    const Image<Vector3fda>& cuN, 
    Image<uint16_t>& cuZ,
    size_t maxIt, float minNchangePerc) {
  centers_.clear();
  centers_.push_back(n[0]);
  Ns_.push_back(1);
  K_ = 1;
  uint16_t Kprev = 1;
  std::vector<size_t> Nsprev(1,1);
  for (size_t it=0; it<maxIt; ++it) {
    TICK("DPvMF means labels");
    UpdateLabels(n,cuN,cuZ);
    TOCK("DPvMF means labels");
    TICK("DPvMF means centers");
    UpdateCenters(cuN,cuZ);
    TOCK("DPvMF means centers");

    if (K_ == Kprev) {
      uint32_t Nchange = 0;
      uint32_t N = 0;
      for (uint16_t k=0; k<K_; ++k) {
        Nchange += abs((int32_t)Ns_[k] - (int32_t)Nsprev[k]); 
        N += Ns_[k];
      }
      std::cout << "K:" << K_ << " # " <<  N 
        << " change " << Nchange << " thr "
        << minNchangePerc*N << std::endl;
      if (Nchange < minNchangePerc*N)
        break;
    }

    Kprev = K_;
    Nsprev = Ns_;
  }
}

void DPvMFmeans::UpdateLabels(
    const Image<Vector3fda>& n, 
    const Image<Vector3fda>& cuN, 
    Image<uint16_t>& cuZ
    ) {
  const uint32_t UNASSIGNED = std::numeric_limits<uint32_t>::max();
  uint32_t i0 = 0;
  uint32_t idAction = UNASSIGNED;

  ManagedDeviceImage<Vector3fda> cuCenters(K_,1);
  cudaMemcpy(cuCenters.ptr_, &(centers_[0]), cuCenters.SizeBytes(),
      cudaMemcpyHostToDevice);

  for (size_t count = 0; count < n.Area(); count++){
    idAction = optimisticLabelsAssign(cuN,cuCenters,cuZ,i0);
    if(idAction == UNASSIGNED) {
      // cout<<"[ddpmeans] break." << endl;
      break;
    }
    float sim = 0.;
    uint16_t z_i = indOfClosestCluster(n[idAction],sim);
    if(z_i == K_) {
      centers_.push_back(n[idAction]);
      K_ ++;
      //std::cout << "# " << this->cls_.size() << " K=" << this->K_ << std::endl;
      //std::cout << this->cld_->x()->col(idAction) << std::endl;
    }
    i0 = idAction;
  }
}

uint16_t DPvMFmeans::indOfClosestCluster(const Vector3fda& ni, float& sim_closest)
{
  uint16_t z_i = K_;
  sim_closest = lambda_;
  for (uint16_t k=0; k<K_; ++k)
  {
    float sim_k = centers_[k].dot(ni);
    if(sim_k > sim_closest) {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
}

void DPvMFmeans::UpdateCenters(
    const Image<Vector3fda>& cuN, 
    const Image<uint16_t>& cuZ
    ) {

  Eigen::Matrix<float,4,Eigen::Dynamic> ss = computeSS(cuN,cuZ);
  std::cout << ss << std::endl;
  Ns_.clear();
  for(size_t k=0; k<K_; ++k) 
    Ns_.push_back(ss(3,k));

  for(size_t k=0; k<K_; ++k) {
    if(Ns_[k] == 0) {
      // reset centroid
      centers_[k] = Vector3fda::Random();
      centers_[k] /= centers_[k].norm();
    } else {
      centers_[k] = ss.block<3,1>(0,k) / ss(3,k);
    }
  }
}

Eigen::Matrix<float,4,Eigen::Dynamic> DPvMFmeans::computeSS(
    const Image<Vector3fda>& cuN, 
    const Image<uint16_t>& cuZ) {
  return SufficientStats1stOrder(cuN, cuZ, K_);
}

uint32_t DPvMFmeans::optimisticLabelsAssign(const Image<Vector3fda>& cuN, 
    Image<Vector3fda>& cuCenters,
    Image<uint16_t>& cuZ, uint32_t i0) {
  return dpvMFlabelsOptimistic(cuN,cuCenters,cuZ, lambda_, i0, K_);
}

}
