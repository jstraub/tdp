/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/bb/numeric_helpers.h>

namespace tdp {

double LogSumExp(const Eigen::VectorXd& x) {
  const double x_max = x.maxCoeff();
  return log((x.array() - x_max).exp().sum()) + x_max;
};

double SumExp(const Eigen::VectorXd& x) {
  const double x_max = x.maxCoeff();
  return (x.array() - x_max).exp().sum() * exp(x_max);
};

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

}
