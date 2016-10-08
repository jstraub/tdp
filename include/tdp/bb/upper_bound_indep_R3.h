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

class UpperBoundIndepR3 : public Bound<NodeR3> {
 public:
  UpperBoundIndepR3(const std::vector<Normal3f>& gmm_A, const
      std::vector<Normal3f>& gmm_B, const Eigen::Quaternion<float>& q);
  virtual ~UpperBoundIndepR3() = default;
  virtual float Evaluate(const NodeR3& node);
  virtual float EvaluateAndSet(NodeR3& node);
 private:
  std::vector<Normal3f> gmmT_;
};

Eigen::Vector3f FindMinTranslationInNode(const Eigen::Matrix3f& A, 
    const Eigen::Vector3f& b, const NodeR3& node);

}
