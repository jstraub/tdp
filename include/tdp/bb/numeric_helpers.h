/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>

namespace tdp {

float LogSumExp(const Eigen::VectorXf& x);
float SumExp(const Eigen::VectorXf& x);

}
