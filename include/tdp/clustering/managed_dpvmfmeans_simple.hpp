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
template<class T, int D, int Options>
class ManagedDPvMFmeansSimple : public DPvMFmeansSimple<T,D,Options>
{
public:
  /// Constructor
  /// 
  /// lambda = cos(lambda_in_degree * M_PI/180.) - 1.
  ManagedDPvMFmeansSimple(T lambda);
  virtual ~ManagedDPvMFmeansSimple();

  /// Adds an observation (adds obs, computes label, and potentially
  /// adds new cluster depending on label assignment).
  virtual void addObservation(const Eigen::Matrix<T,D,1,Options>& x);

protected:
};

typedef ManagedDPvMFmeansSimple<float,3,Eigen::DontAlign> ManagedDPvMFmeansSimple3fda; 
typedef ManagedDPvMFmeansSimple<float,4,Eigen::DontAlign> ManagedDPvMFmeansSimple4fda; 

// -------------------------------- impl ----------------------------------
template<class T, int D, int Options>
ManagedDPvMFmeansSimple<T,D,Options>::ManagedDPvMFmeansSimple(T lambda)
  : DPvMFmeansSimple<T,D,Options>(lambda)
{}

template<class T, int D, int Options>
ManagedDPvMFmeansSimple<T,D,Options>::~ManagedDPvMFmeansSimple()
{
  for (auto xi : this->xs_) delete xi;
  for (auto zi : this->zs_) delete zi;
}

template<class T, int D, int Options>
void ManagedDPvMFmeansSimple<T,D,Options>::addObservation(const Eigen::Matrix<T,D,1,Options>& x) {
  Eigen::Matrix<T,D,1,Options>* xi = new Eigen::Matrix<T,D,1,Options>(x); 
  uint16_t* zi = new uint16_t(0); 
  DPvMFmeansSimple<T,D,Options>::addObservation(xi, zi);
};

}
