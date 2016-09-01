#pragma once

#include <Eigen/Dense>
#include <tdp/image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void MMFvMFCostFctAssignmentGPU( Image<Vector3fda> cuN, 
    Image<uint32_t> cuZ, Image<Vector3fda>cuMu, Image<float> cuPi, 
    int K, float& cost, float& W);

void MMFvMFCostFctAssignmentGPU(
    Image<Vector3fda> cuN, Image<float> cuWeights,
    Image<uint32_t> cuZ, Image<Vector3fda>cuMu, Image<float> cuPi, 
    int K, float& cost, float& W);

template<int K>
class vMFMMF {
 public:
   vMFMF(size_t w, size_t h) 
    : cuZ(w,h), cuMu(6*K,1), cuPi(6*K,1) {};
   ~vMFMF() {};

   float Compute(Image<Vector3fda>& cuN)

 private:
   float UpdateMF();
   void UpdateAssociation();

   ManagedDeviceImage<uint32_t> cuZ;
   ManagedDeviceImage<Vector3fda> cuMu;
   ManagedDeviceImage<float> cuPi;
};

float vMFMF::Compute(Image<Vector3fda>& cuN, 
    size_t maxIt) {

  for (size_t it=0; it<maxIt; ++it) {
    UpdateAssociation(cuN);
    UpdateMF(cuN);
  }
}

float vMFMF::UpdateMF(Image<Vector3fda>& cuN) {
  Eigen::Matrix3f N = Eigen::Matrix3f::Zero();
  // tauR_*R^T is the contribution of the motion prior between two
  // frames to regularize solution in case data exists only on certain
  // axes
  if (this->t_ >= 1) N += tauR_*R.transpose();
  for (uint32_t j=0; j<6; ++j) { 
    Eigen::Vector3f m = Eigen::Vector3f::Zero();
    m(j/2) = j%2==0?1.:-1.;
    N += m*this->cld_.xSums().col(j).transpose();
  }
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(N,Eigen::ComputeThinU|Eigen::ComputeThinV);
  if (svd.matrixV().determinant()*svd.matrixU().determinant() > 0)
    R = svd.matrixV()*svd.matrixU().transpose();
  else
    R = svd.matrixV()*Eigen::Vector3f(1.,1.,-1.).asDiagonal()*svd.matrixU().transpose();
  return (N*R).trace();
}

void vMFMF::UpdateAssociation(Image<Vector3fda>& cuN) {
  float cost = 0.;
  float W = 0.;
  MMFvMFCostFctAssignmentGPU(cuN,cuZ,cuMu,cuPi,K,cost,W);
  return cost/W;
}


}
