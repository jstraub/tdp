#pragma once

#include <Eigen/Dense>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>
#include <tdp/reductions/vectorSum.h>

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
   vMFMMF(float tauR) 
    : t_(0), tauR_(tauR), cuMu_(6*K,1), cuPi_(6*K,1) 
   {Reset();};
   ~vMFMMF() {};

   float Compute(const Image<Vector3fda>& cuN, size_t maxIt, bool verbose);
   void Reset();

   std::vector<Eigen::Matrix3f> Rs_; // rotations of MFs
   std::vector<float> cs_; // costs MFs
   std::vector<float> Ns_; // counts
   int64_t t_;
   float tauR_;
 private:
   float UpdateMF(const Image<Vector3fda>& cuN);
   float UpdateAssociation(const Image<Vector3fda>& cuN);

   Eigen::Matrix<float,4,6> ComputeSums(const Image<Vector3fda>& cuN, uint32_t k);

   ManagedDeviceImage<uint32_t> cuZ_;
   ManagedDeviceImage<Vector3fda> cuMu_;
   ManagedDeviceImage<float> cuPi_;
};

template<int K>
void vMFMMF<K>::Reset() { 
  Rs_.clear(); 
  Rs_.resize(K, Eigen::Matrix3f::Identity());
  cs_.clear(); 
  cs_.resize(K, 0.f);
  Ns_.clear(); 
  Ns_.resize(K, 0.f);
  t_ = 0;
}

template<int K>
float vMFMMF<K>::Compute(const Image<Vector3fda>& cuN, 
    size_t maxIt, bool verbose) {
  cuZ_.Reinitialise(cuN.w_, cuN.h_);
  float assocCost = 0;
  Eigen::VectorXf csPrev(K);
  csPrev.fill(1e16);
  for (size_t it=0; it<maxIt; ++it) {
    assocCost = UpdateAssociation(cuN);
    UpdateMF(cuN);
    if (verbose) {
      std::cout << "rtmf @" << it << ":\t" << assocCost << " costs: ";
      for (size_t k=0; k<K; ++k) {
        std::cout << "\t" << cs_[k];
      }
      std::cout << " Ns: ";
      for (size_t k=0; k<K; ++k) {
        std::cout << "\t" << Ns_[k];
      }
      std::cout << std::endl;
//      std::cout << Rs_[0] << std::endl;
//      std::cout << Rs_[0].determinant() << std::endl;
    }
    Eigen::Map<Eigen::VectorXf> cs(&cs_[0],K);
    if (((cs - csPrev).array().abs() < 1e-6).all())
      break;
    csPrev = cs;
  }
  return assocCost;
}

template<int K>
float vMFMMF<K>::UpdateMF(const Image<Vector3fda>& cuN) {
  for (size_t k=0; k<K; ++k) {
    Ns_[k] = 0;
    Eigen::Matrix3f N = Eigen::Matrix3f::Zero();
    // tauR_*R^T is the contribution of the motion prior between two
    // frames to regularize solution in case data exists only on certain
    // axes
    if (t_ > 0) N += tauR_*Rs_[k].transpose();
    Eigen::Matrix<float,4,6> nSums = ComputeSums(cuN, k);
//    std::cout << nSums << std::endl;
    for (uint32_t j=0; j<6; ++j) { 
      Eigen::Vector3f m = Eigen::Vector3f::Zero();
      m(j/2) = j%2==0?1.:-1.;
      N += m*nSums.block<3,1>(0,j).transpose();
      Ns_[k] += nSums(3,j);
    }
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(N,Eigen::ComputeFullU|Eigen::ComputeFullV);
//    if (svd.matrixV().determinant()*svd.matrixU().determinant() > 0)
//      Rs_[k] = svd.matrixV()*svd.matrixU().transpose();
//    else
//      Rs_[k] = svd.matrixV()*Eigen::Vector3f(1.,1.,-1.).asDiagonal()*svd.matrixU().transpose();
    Rs_[k] = svd.matrixV()*Eigen::Vector3f(1.,1.,(svd.matrixV()*svd.matrixU()).determinant()).asDiagonal()*svd.matrixU().transpose();
    cs_[k] = (N*Rs_[k]).trace();
  }
  float C = 0.;
  for (auto& c : cs_) C += c;
  return C;
}

template<int K>
Eigen::Matrix<float,4,6> vMFMMF<K>::ComputeSums(const
    Image<Vector3fda>& cuN, uint32_t k) {
  ManagedDeviceImage<Vector4fda> cuSSs(6,1);
  cudaMemset(cuSSs.ptr_,0,cuSSs.SizeBytes());
  VectorSum(cuN, cuZ_, k*6, 6, cuSSs);
  ManagedHostImage<Vector4fda> SSs(6,1);
  SSs.CopyFrom(cuSSs,cudaMemcpyDeviceToHost);
  Eigen::Matrix<float,4,6> ss;
  for (int j=0; j<6; ++j)
    ss.col(j) = SSs[j];
  return ss;
}

template<int K>
float vMFMMF<K>::UpdateAssociation(const Image<Vector3fda>& cuN) {
  float cost = 0.;
  float W = 0.;

  ManagedHostImage<Vector3fda> mu(K*6,1);
  ManagedHostImage<float> pi(K*6,1);
  for (size_t k=0; k<K; ++k) {
    for (uint32_t j=0; j<6; ++j) { 
      mu[k*6+j] = Rs_[k].col(j/2) * (j%2==0?1.:-1.);
      pi[k*6+j] = log(1./(6.*K));
//      std::cout << mu[k*6+j].transpose() << " ---- " << pi[k*6+j] << std::endl;
    }
  }
  cuMu_.CopyFrom(mu,cudaMemcpyHostToDevice);
  cuPi_.CopyFrom(pi,cudaMemcpyHostToDevice);

  MMFvMFCostFctAssignmentGPU(cuN,cuZ_,cuMu_,cuPi_,K,cost,W);
//  ManagedHostImage<uint32_t> z(cuZ_.w_);
//  z.CopyFrom(cuZ_, cudaMemcpyDeviceToHost);
//  for (size_t i=0; i<z.w_; ++i) {
//    std::cout << z[i] << " ";
//  }
//  std::cout << std::endl;
  
  return cost/W;
}


}
