/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <tdp/image.h>
#include <tdp/sufficientStats.h>

namespace tdp {

size_t dpvMFlabelsOptimistic( 
    Image<Vector3fda> n,
    Image<Vector3fda> mu,
    Image<uint16_t> z,
    float lambda, size_t i0, uint16_t K);

class DPvMFmeans {
 public: 
  DPvMFmeans(float lambda) : lambda_(lambda) {};
  ~DPvMFmeans() {};

  float Compute(const Image<Vector3fda>& n, const Image<Vector3fda>& cuN, 
      Image<uint16_t>& cuZ)

  uint16_t K_;
  float lambda_;
  std::vector<Vector3fda> centers_;
  std::vector<size_t> Ns_;
 private:

  void UpdateLabels(const Image<Vector3fda>& n, Image<uint16_t>& z);
  void UpdateCenters(const Image<Vector3fda>& n, const Image<uint16_t>& z);

  size_t optimisticLabelsAssign(const Image<Vector3fda>& cuN, 
    Image<uint16_t>& cuZ, size_t i0);
};

float DPvMFmeans::Compute(const Image<Vector3fda>& n, 
    const Image<Vector3fda>& cuN, 
    Image<uint16_t>& cuZ,
    float lambda, size_t maxIt) {

  for (size_t it=0; it<maxIt; ++it) {
    UpdateLabels(n,cuN,cuZ);
    UpdateCenters(cuN,cuZ);
  }
}

void DPvMFmeans::UpdateLabels(const Image<Vector3fda>& n, 
    const Image<Vector3fda>& cuN, 
    Image<uint16_t>& cuZ) {
  const size_t UNASSIGNED = std::numeric_limits<size_t>::max();
  size_t i0 = 0;
  size_t idAction = UNASSIGNED;

  ManagedDeviceImage<Vector3fda> cuCenters(K_,1);
  cudaMemcpy(cuCenters_.ptr_, &(centers_[0]), cuCenters.SizeBytes(), cudaMemcpyHostToDevice);

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

  // if a cluster runs out of labels reset it to the previous mean!
  for(size_t k=0; k<K_; ++k)
    if(Ns_[k] == 0)
    {
      std::cout << "ERROR ran  out of data in a cluster" << std::endl;
      //this->cls_[k]->centroid() = this->clsPrev_[k]->centroid();
      //this->cls_[k]->centroid() = this->cls_[k]->prevCentroid();
    }
}

uint16_t DPvMFmeans::indOfClosestCluster(Vector3fda& ni, float& sim_closest)
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

void DPvMFmeans::UpdateCenters(const Image<Vector3fda>& cuN, 
    const Image<uint16_t>& cuZ) {

  Eigen::Matrix<float,4,Eigen::Dynamic> ss = computeSS(cuN,cuZ);

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
    Image<uint16_t>& cuZ) {
  return SufficientStats1stOrder(cuN, cuZ, K_);
}

size_t DPvMFmeans::optimisticLabelsAssign(const Image<Vector3fda>& cuN, 
    Image<Vector3fda>& cuCenters,
    Image<uint16_t>& cuZ, size_t i0) {
  return dpvMFlabelsOptimistic(cuN,cuCenters,cuZ, lambda, i0, K_);
}

}
