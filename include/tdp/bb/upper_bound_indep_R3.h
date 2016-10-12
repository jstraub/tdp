/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tdp/bb/node_R3.h>
#include <tdp/bb/numeric_helpers.h>
#include <tdp/distributions/normal.h>
#include <tdp/bb/bound.h>
#include <tdp/bb/lower_bound_R3.h>

namespace tdp {

template <typename T>
class UpperBoundIndepR3 : public Bound<T,NodeR3<T>> {
 public:
  UpperBoundIndepR3(const std::vector<Normal<T,3>>& gmm_A, const
      std::vector<Normal<T,3>>& gmm_B, const Eigen::Quaternion<T>& q);
  virtual ~UpperBoundIndepR3() = default;
  virtual T Evaluate(const NodeR3<T>& node);
  virtual T EvaluateAndSet(NodeR3<T>& node);
 private:
  std::vector<Normal<T,3>> gmmT_;
};

typedef UpperBoundIndepR3<float>  UpperBoundIndepR3f;
typedef UpperBoundIndepR3<double> UpperBoundIndepR3d;

template <typename T>
Eigen::Matrix<T,3,1> FindMinTranslationInNode(const Eigen::Matrix<T,3,3>& A, 
    const Eigen::Matrix<T,3,1>& b, const NodeR3<T>& node);

}
