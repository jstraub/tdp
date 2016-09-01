#pragma once

#include <Eigen/Dense>
#include <tdp/image.h>

namespace tdp {

class vMFMF {
 public:
   vMFMF() {};
   ~vMFMF() {};

 private:
   float UpdateMF();
   void UpdateAssociation();

};


float vMFMF::UpdateMF() {
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

void vMFMF::UpdateAssociation() {
  Rot2Device(R);
  float residuals[6]; // for all 6 different axes
  vMFCostFctAssignmentGPU(residuals, d_cost, &N, d_N_, cld_.d_x(),
      d_weights_, cld_.d_z(), d_mu_, pi_.data(), cld_.N());
  float residual = 0.0f;
  for (uint32_t i=0; i<6; ++i) residual +=  residuals[i];
  return residual;
}


}
