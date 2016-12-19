/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <tdp/eigen/std_vector.h>

namespace tdp {
/// This is a simple version of the DPvMFmeansSimple algorithm in dpmeans.hpp
/// without inheritance or the use of CLData structures which can make
/// it a bit hard to read the other algorithm.
///
/// This implementation is ment as a lightweight alternative for small
/// number of datapoints or if you just want to have a look at how the
/// algorithm works.
template<class T, int D>
class DPvMFmeansSimple
{
public:
  /// Constructor
  /// 
  /// lambda = cos(lambda_in_degree * M_PI/180.) - 1.
  DPvMFmeansSimple(T lambda);
  virtual ~DPvMFmeansSimple();

  /// Adds an observation (adds obs, computes label, and potentially
  /// adds new cluster depending on label assignment).
  virtual void addObservation(const Eigen::Matrix<T,D,1>& x);
  /// Updates all labels of all data currently stored with the object.
  virtual void updateLabels();
  /// Updates all centers based on the current data and label
  /// assignments.
  virtual void updateCenters();

  /// Iterate updates for centers and labels until cost function
  /// convergence.
  virtual bool iterateToConvergence(uint32_t maxIter, T eps);
  /// Compuyte the current cost function value.
  virtual T cost();

  uint32_t GetK() const {return K_;};
  const std::vector<int32_t>& GetNs() const {return Ns_;};
  bool GetCenter(uint32_t k, Eigen::Matrix<T,D,1>& mu) const {
    if (k<K_) {mu = mus_[k]; return true; } else { return false; } };
  const Eigen::Matrix<T,D,1>& GetCenter(uint32_t k) const {
    if (k<K_) {return mus_[k]; } else { return Eigen::Matrix<T,D,1>::Zero(); } };
  const std::vector<uint32_t>& GetZs() const { return zs_;};
  bool GetX(uint32_t i, Eigen::Matrix<T,D,1>& x) const {
    if (i<xs_.size()) {x=xs_[i]; return true; } else { return false; } };

protected:
  T lambda_;
  uint32_t K_;
  eigen_vector<Eigen::Matrix<T,D,1>> xs_;
  eigen_vector<Eigen::Matrix<T,D,1>> mus_;
  eigen_vector<Eigen::Matrix<T,D,1>> xSums_;
  std::vector<uint32_t> zs_;
  std::vector<int32_t> Ns_;

  /// resets all clusters (mus_ and Ks_) and resizes them to K_
  void resetClusters();
  /// Removes all empty clusters.
  void removeEmptyClusters();
  /// Computes the index of the closest cluster (may be K_ in which
  /// case a new cluster has to be added).
  uint32_t indOfClosestCluster(const Eigen::Matrix<T,D,1>& x, T& sim_closest,
      uint32_t* zExclude=nullptr);
};

typedef DPvMFmeansSimple<float,3> DPvMFmeansSimple3f; 

// -------------------------------- impl ----------------------------------
template<class T, int D>
DPvMFmeansSimple<T,D>::DPvMFmeansSimple(T lambda)
  : lambda_(lambda), K_(0)
{}
template<class T, int D>
DPvMFmeansSimple<T,D>::~DPvMFmeansSimple()
{}

template<class T, int D>
void DPvMFmeansSimple<T,D>::addObservation(const Eigen::Matrix<T,D,1>& x) {
  xs_.push_back(x); 
  T sim_closest = 0;
  uint32_t z = indOfClosestCluster(x, sim_closest);
  if (z == K_) {
    mus_.push_back(x);
    xSums_.push_back(x);
    Ns_.push_back(0);
    ++K_;
//    std::cout << "adding cluster " << mus_.size() << " "
//      << xSums_.size() << " " << K_ << " "
//      << x.transpose() << std::endl;
  }
  zs_.push_back(z);
  Ns_[z] ++;
};

template<class T, int D>
uint32_t DPvMFmeansSimple<T,D>::indOfClosestCluster(const
    Eigen::Matrix<T,D,1>& x, T& sim_closest, uint32_t* zExclude)
{
  uint32_t z_i = K_;
  sim_closest = lambda_;
  for (uint32_t k=0; k<K_; ++k) {
    if (zExclude && k == *zExclude)
      continue;
    T sim_k = mus_[k].dot(x);
    if(sim_k > sim_closest) {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
};

template<class T, int D>
void DPvMFmeansSimple<T,D>::updateLabels()
{
  if (xs_.size() == 0) return;
  for(uint32_t i=0; i<xs_.size(); ++i) {
    T sim_closest = 0;
    uint32_t zPrev = zs_[i];
    uint32_t z = indOfClosestCluster(xs_[i], sim_closest);
    if (z==zPrev && Ns_[z] == 1) {
      z = indOfClosestCluster(xs_[i], sim_closest, &z);
//      std::cout << "single cluster " << z << " " << zPrev << std::endl;
    }
    if (z == K_) {
//      std::cout << "adding cluster " << mus_.size() << " "
//        << xSums_.size() << " " << K_ << " "
//        << xs_[i].transpose() << std::endl;
      mus_.push_back(  xs_[i]);
      xSums_.push_back(Eigen::Matrix<T,D,1>::Zero());
      Ns_.push_back(0);
      ++K_;
    }
    if (z != zPrev) {
      Ns_[zPrev] --;
      xSums_[zPrev] -= xs_[i];
      Ns_[z] ++; 
      xSums_[z] += xs_[i];
    }
    zs_[i] = z;
  }
};

// General update centers assumes Euclidean
template<class T, int D>
void DPvMFmeansSimple<T,D>::updateCenters()
{
  if (xs_.size() == 0) return;
//  resetClusters();
//  for(uint32_t i=0; i<xs_.size(); ++i) {
//    ++Ns_[zs_[i]]; 
//    mus_[zs_[i]] += xs_[i];
//  }
  // Spherical mean computation
  for(uint32_t k=0; k<K_; ++k) {
//    mus_[k] /= mus_[k].norm();
    mus_[k] = xSums_[k].normalized();
  }
  removeEmptyClusters();
};

template<class T, int D>
void DPvMFmeansSimple<T,D>::resetClusters() {
  Ns_.resize(K_, 0);
  for(uint32_t k=0; k<K_; ++k) {
    mus_[k].fill(0);
    Ns_[k] = 0;
  }
};

template<class T, int D>
void DPvMFmeansSimple<T,D>::removeEmptyClusters() {
  if (K_ < 1) return;
  uint32_t kNew = K_;
  std::vector<bool> toDelete(K_,false);
  for(int32_t k=K_-1; k>-1; --k)
    if(Ns_[k] == 0) {
      toDelete[k] = true;
//      std::cout<<"cluster k "<<k<<" empty"<<std::endl;
//#pragma omp parallel for 
      for(uint32_t i=0; i<xs_.size(); ++i)
        if(static_cast<int32_t>(zs_[i]) >= k) zs_[i] -= 1;
      kNew --;
    }
  uint32_t j=0;
  for(uint32_t k=0; k<K_; ++k) {
    mus_[j] = mus_[k];
    xSums_[j] = xSums_[k];
    Ns_[j] = Ns_[k];
    if(!toDelete[k]) { 
      ++j;
    }
  }
//  std::cout << "K " << K_ << " -> " << kNew << std::endl;
  K_ = kNew;
  Ns_.resize(K_);
  mus_.resize(K_);
  xSums_.resize(K_);
//  for(uint32_t k=0; k<K_; ++k) 
//    std::cout << mus_[k].transpose() << std::endl;
};

template<class T, int D>
T DPvMFmeansSimple<T,D>::cost() {
  T f = lambda_*K_; 
//  std::cout << "f="<<f<< std::endl;
  for(uint32_t i=0; i<xs_.size(); ++i)  {
    f += mus_[zs_[i]].dot(xs_[i]);
//    std::cout << zs_[i] << ", " << xs_[i].transpose() << ", " 
//      << mus_[zs_[i]].transpose();
//    std::cout << " f="<<f<< std::endl;
  }
  return f;
}

template<class T, int D>
bool DPvMFmeansSimple<T,D>::iterateToConvergence(uint32_t maxIter, T eps) {
  uint32_t iter = 0;
  T fPrev = 1e9;
  T f = cost();
//  std::cout << "f=" << f << " fPrev=" << fPrev << std::endl;
  while (iter < maxIter && fabs(fPrev - f)/f > eps) {
    updateCenters();
    updateLabels();
    fPrev = f;
    f = cost();
    ++iter;
//    std::cout << iter << ": f=" << f << " fPrev=" << fPrev << ": ";
//    int32_t Nall = 0;
//    for (const auto& N : Ns_) {
//      std::cout << N << " ";
//      Nall += N;
//    } std::cout << " sum= " << Nall << std::endl;
  }
//  if (f != f || fPrev != fPrev || f > 1e9 || iter == maxIter-1) {
  std::cout << iter << ": f=" << f << " fPrev=" << fPrev << ": ";
  int32_t Nall = 0;
  for (const auto& N : Ns_) {
    std::cout << N << " ";
    Nall += N;
  } std::cout << " sum= " << Nall << std::endl;
//  }
  return iter < maxIter;
}

}
