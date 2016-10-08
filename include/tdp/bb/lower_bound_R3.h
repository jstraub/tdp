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

class LowerBoundR3 : public Bound<NodeR3> {
 public:
  LowerBoundR3(const std::vector<Normal3f>& gmmA, const
      std::vector<Normal3f>& gmmB, const Eigen::Quaternion<float>& q);
  virtual ~LowerBoundR3() = default;
  virtual float Evaluate(const NodeR3& node);
  virtual float EvaluateAndSet(NodeR3& node);
 private:
  void Evaluate(const NodeR3& node, Eigen::Matrix<float,3,9>& xs,
      Eigen::Matrix<float,9,1>& lbs);
  std::vector<Normal3f> gmmT_;
};

void ComputeGmmT( const std::vector<Normal3f>& gmmA, const
    std::vector<Normal3f>& gmmB, std::vector<Normal3f>& gmmT, const
    Eigen::Quaternion<float>& q);

}
