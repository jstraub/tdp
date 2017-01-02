/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <tdp/eigen/std_vector.h>
#include <tdp/clustering/dpvmfmeans_simple.hpp>

namespace tdp {
/// This is a simple version of the ManagedDPvMFmeansSimple algorithm in dpmeans.hpp
/// without inheritance or the use of CLData structures which can make
/// it a bit hard to read the other algorithm.
///
/// This implementation is ment as a lightweight alternative for small
/// number of datapoints or if you just want to have a look at how the
/// algorithm works.
template<class T, int D>
class ManagedDPvMFmeansSimple : public DPvMFmeansSimple<T,D>
{
public:
  /// Constructor
  /// 
  /// lambda = cos(lambda_in_degree * M_PI/180.) - 1.
  ManagedDPvMFmeansSimple(T lambda);
  virtual ~ManagedDPvMFmeansSimple();

  /// Adds an observation (adds obs, computes label, and potentially
  /// adds new cluster depending on label assignment).
  virtual void addObservation(const Eigen::Matrix<T,D,1>& x);

protected:
};

typedef ManagedDPvMFmeansSimple<float,3> ManagedDPvMFmeansSimple3f; 
typedef ManagedDPvMFmeansSimple<float,4> ManagedDPvMFmeansSimple4f; 

// -------------------------------- impl ----------------------------------
template<class T, int D>
ManagedDPvMFmeansSimple<T,D>::ManagedDPvMFmeansSimple(T lambda)
  : DPvMFmeansSimple(lambda)
{}
template<class T, int D>
ManagedDPvMFmeansSimple<T,D>::~ManagedDPvMFmeansSimple()
{
  for (auto xi : xs_) delete xi;
  for (auto zi : zs_) delete zi;
}

template<class T, int D>
void ManagedDPvMFmeansSimple<T,D>::addObservation(const Eigen::Matrix<T,D,1>& x) {
  Eigen::Matrix<T,D,1>* xi = new Eigen::Matrix<T,D,1>(x); 
  uint32_t* zi = new uint32_t(0); 
  DPvMFmeansSimple<T,D>::addObservation(xi, zi);
};

}
