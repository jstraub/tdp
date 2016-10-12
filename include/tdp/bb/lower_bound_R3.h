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

namespace tdp {

template <typename T>
class LowerBoundR3 : public Bound<T,NodeR3<T>> {
 public:
  LowerBoundR3(const std::vector<Normal<T,3>>& gmmA, const
      std::vector<Normal<T,3>>& gmmB, const Eigen::Quaternion<T>& q);
  virtual ~LowerBoundR3() = default;
  virtual T Evaluate(const NodeR3<T>& node);
  virtual T EvaluateAndSet(NodeR3<T>& node);
 private:
  void Evaluate(const NodeR3<T>& node, Eigen::Matrix<T,3,9>& xs,
      Eigen::Matrix<T,9,1>& lbs);
  std::vector<Normal<T,3>> gmmT_;
};

typedef LowerBoundR3<float>  LowerBoundR3f;
typedef LowerBoundR3<double> LowerBoundR3d;

template <typename T>
void ComputeGmmT( const std::vector<Normal<T,3>>& gmmA, const
    std::vector<Normal<T,3>>& gmmB, std::vector<Normal<T,3>>& gmmT, const
    Eigen::Quaternion<T>& q);

}
