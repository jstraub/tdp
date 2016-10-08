/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/bb/numeric_helpers.h>

namespace tdp {

float LogSumExp(const Eigen::VectorXf& x) {
  const float x_max = x.maxCoeff();
  return log((x.array() - x_max).exp().sum()) + x_max;
};

float SumExp(const Eigen::VectorXf& x) {
  const float x_max = x.maxCoeff();
  return (x.array() - x_max).exp().sum() * exp(x_max);
};

}
