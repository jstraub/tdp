/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>

namespace tdp {

template <typename T>
T LogSumExp(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
  const T x_max = x.maxCoeff();
  return log((x.array() - x_max).exp().sum()) + x_max;
}

template <typename T>
T SumExp(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
  const T x_max = x.maxCoeff();
  return (x.array() - x_max).exp().sum() * exp(x_max);
}

}
