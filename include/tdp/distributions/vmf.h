/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <tdp/bb/numeric_helpers.h>

namespace tdp {

/// Compute log((exp(z) - exp(-z)) / z)
template <typename T>
T ComputeLog2SinhOverZ(T z) {
  if (fabs(z) < 1.e-3) 
    return log(2.);
  else if (z < 50.) 
    return log(exp(z) - exp(-z)) - log(z);
//  else
//    return z - log(z) + log(1.- exp(-2.*z) );
  else
    return z - log(z);
};

/// vMF distribution templated on the dimension.
template <typename T, uint32_t D>
class vMF {
 public:
  vMF(const Eigen::Matrix<T, D, 1>& mu, T tau, T pi);
  vMF(const vMF<T,D>& vmf) = default;
  ~vMF() = default;
  T GetPi() const {return pi_;}
  void SetPi(T pi) {pi_ = pi;}
  T GetTau() const {return tau_;}
  const Eigen::Matrix<T, D, 1>& GetMu() const {return mu_;}
  T GetLogZ() const;
  void Print() const {
    std::cout << "vMF tau= " << tau_ << "\tmu= " << mu_.transpose() 
      << "\tpi= " << pi_ << std::endl;
  }
  static T MLEstimateTau(const Eigen::Matrix<T,3,1>& xSum, const
      Eigen::Matrix<T,3,1>& mu, T count);
 private:
  Eigen::Matrix<T, D, 1> mu_;
  T tau_;
  T pi_;
};

typedef vMF<float,3> vMF3f;
typedef vMF<double,3> vMF3d;

template<uint32_t D>
T ComputeLogvMFtovMFcost(const vMF<T,D>& vmf_A, const vMF<T,D>& vmF_B, 
  const Eigen::Matrix<T, D, 1>& mu_B_prime);

//T MLEstimateTau(const Eigen::Vector3d& xSum, const
//    Eigen::Vector3d& mu, T count);

}
#include <tdp/distributions/vmf_impl.h>
