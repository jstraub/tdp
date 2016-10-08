/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <tdp/eigen/dense.h>
#include <tdp/eigen/std_vector.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/image.h>
#include <tdp/stats/sufficientStats.h>
#include <tdp/utils/Stopwatch.h>

namespace tdp {

uint32_t dpMeansLabelsOptimistic( 
    Image<Vector3fda> x,
    Image<Vector3fda> mu,
    Image<uint16_t> z,
    float lambda, uint32_t i0, uint16_t K);

class DPmeans {
 public: 
  DPmeans(float lambda) : lambda_(lambda) {};
  ~DPmeans() {};

  void Compute(const Image<Vector3fda>& x, const Image<Vector3fda>& cuX, 
      Image<uint16_t>& cuZ, size_t maxIt, float minNchangePerc);

  uint16_t K_;
  float lambda_;
  eigen_vector<Vector3fda> centers_;
  std::vector<size_t> Ns_;
 private:

  void UpdateLabels(
      const Image<Vector3fda>& x, 
      const Image<Vector3fda>& cuX, 
      Image<uint16_t>& cuZ
      );
  void UpdateCenters(
      const Image<Vector3fda>& cuX, 
      const Image<uint16_t>& cuZ
      );

  uint32_t optimisticLabelsAssign(
      const Image<Vector3fda>& cuX, 
      Image<Vector3fda>& cuCenters,
      Image<uint16_t>& cuZ, uint32_t i0
    );


  Eigen::Matrix<float,4,Eigen::Dynamic> computeSS(
      const Image<Vector3fda>& cuX, 
      const Image<uint16_t>& cuZ);

  uint16_t indOfClosestCluster(const Vector3fda& xi, float& sim_closest);
};

void DPmeans::Compute(const Image<Vector3fda>& x, 
    const Image<Vector3fda>& cuX, 
    Image<uint16_t>& cuZ,
    size_t maxIt, float minNchangePerc) {
  centers_.clear();
  centers_.push_back(x[0]);
  Ns_.push_back(1);
  K_ = 1;
  uint16_t Kprev = 1;
  std::vector<size_t> Nsprev(1,1);
  for (size_t it=0; it<maxIt; ++it) {
    TICK("DPmeans labels");
    UpdateLabels(x,cuX,cuZ);
    TOCK("DPmeans labels");
    TICK("DPmeans centers");
    UpdateCenters(cuX,cuZ);
    TOCK("DPmeans centers");

    if (K_ == Kprev) {
      uint32_t Nchange = 0;
      uint32_t x = 0;
      for (uint16_t k=0; k<K_; ++k) {
        Nchange += abs((int32_t)Ns_[k] - (int32_t)Nsprev[k]); 
        x += Ns_[k];
      }
      std::cout << "K:" << K_ << " # " <<  x 
        << " change " << Nchange << " thr "
        << minNchangePerc*x << std::endl;
      if (Nchange < minNchangePerc*x)
        break;
    }
    Kprev = K_;
    Nsprev = Ns_;
  }
}

void DPmeans::UpdateLabels(
    const Image<Vector3fda>& x, 
    const Image<Vector3fda>& cuX, 
    Image<uint16_t>& cuZ
    ) {
  const uint32_t UNASSIGNED = std::numeric_limits<uint32_t>::max();
  uint32_t i0 = 0;
  uint32_t idAction = UNASSIGNED;

  for (size_t count = 0; count < x.Area(); count++){

    ManagedDeviceImage<Vector3fda> cuCenters(K_,1);
    cudaMemcpy(cuCenters.ptr_, &(centers_[0]), cuCenters.SizeBytes(),
        cudaMemcpyHostToDevice);

    idAction = optimisticLabelsAssign(cuX,cuCenters,cuZ,i0);
    if(idAction == UNASSIGNED) {
      //std::cout<<"[ddpmeans] done." << std::endl;
      break;
    }
    float sim = 0.;
    uint16_t z_i = indOfClosestCluster(x[idAction],sim);
    if(z_i == K_) {
      centers_.push_back(x[idAction]);
      K_ ++;
      //std::cout << "K=" << K_ 
      //  << " idAction=" << idAction
      //  << " xi=" << x[idAction].transpose() << std::endl;
    }
    i0 = idAction;
  }
}

uint16_t DPmeans::indOfClosestCluster(const Vector3fda& xi, float& sim_closest)
{
  uint16_t z_i = K_;
  sim_closest = lambda_;
  for (uint16_t k=0; k<K_; ++k)
  {
    float sim_k = (centers_[k]-xi).norm();
    if(sim_k < sim_closest) {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
}

void DPmeans::UpdateCenters(
    const Image<Vector3fda>& cuX, 
    const Image<uint16_t>& cuZ
    ) {

  Eigen::Matrix<float,4,Eigen::Dynamic> ss = computeSS(cuX,cuZ);
  //std::cout << ss << std::endl;
  Ns_.clear();
  for(size_t k=0; k<K_; ++k) 
    Ns_.push_back(ss(3,k));

  for(size_t k=0; k<K_; ++k) {
    if(Ns_[k] == 0) {
      // reset centroid
      centers_[k] = Vector3fda::Random();
    } else {
      centers_[k] = ss.block<3,1>(0,k) / ss(3,k);
      //std::cout << centers_[k].transpose() << "; " << ss(3,k) << std::endl;
    }
  }
}

Eigen::Matrix<float,4,Eigen::Dynamic> DPmeans::computeSS(
    const Image<Vector3fda>& cuX, 
    const Image<uint16_t>& cuZ) {
  return SufficientStats1stOrder(cuX, cuZ, K_);
}

uint32_t DPmeans::optimisticLabelsAssign(const Image<Vector3fda>& cuX, 
    Image<Vector3fda>& cuCenters,
    Image<uint16_t>& cuZ, uint32_t i0) {
  //std::cout << "DPmeans::optimisticLabelsAssign " << i0 
  //  << " K=" << K_
  //  << " lambda=" << lambda_ << std::endl;
  return dpMeansLabelsOptimistic(cuX,cuCenters,cuZ, lambda_, i0, K_);
}

}
