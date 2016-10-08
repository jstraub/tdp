/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <tdp/bb/numeric_helpers.h>

namespace tdp {

/// Compute log((exp(z) - exp(-z)) / z)
double ComputeLog2SinhOverZ(double z) {
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
template <uint32_t D>
class vMF {
 public:
  vMF(const Eigen::Matrix<double, D, 1>& mu, double tau, double pi);
  vMF(const vMF<D>& vmf) = default;
  ~vMF() = default;
  double GetPi() const {return pi_;}
  void SetPi(double pi) {pi_ = pi;}
  double GetTau() const {return tau_;}
  const Eigen::Matrix<double, D, 1>& GetMu() const {return mu_;}
  double GetLogZ() const;
  void Print() const {
    std::cout << "vMF tau= " << tau_ << "\tmu= " << mu_.transpose() 
      << "\tpi= " << pi_ << std::endl;
  }
  static double MLEstimateTau(const Eigen::Vector3d& xSum, const
      Eigen::Vector3d& mu, double count);
 private:
  Eigen::Matrix<double, D, 1> mu_;
  double tau_;
  double pi_;
};

template<uint32_t D>
double ComputeLogvMFtovMFcost(const vMF<D>& vmf_A, const vMF<D>& vmF_B, 
  const Eigen::Matrix<double, D, 1>& mu_B_prime);

//double MLEstimateTau(const Eigen::Vector3d& xSum, const
//    Eigen::Vector3d& mu, double count);

}
#include <tdp/distributions/vmf_impl.h>
