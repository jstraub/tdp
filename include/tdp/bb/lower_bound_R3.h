/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tdp/bb/node_R3.h>
#include <tdp/bb/numeric_helpers.h>
#include <tdp/bb/normal.h>
#include <tdp/bb/bound.h>

namespace tdp {

class LowerBoundR3 : public Bound<NodeR3> {
 public:
  LowerBoundR3(const std::vector<Normal<3>>& gmmA, const
      std::vector<Normal<3>>& gmmB, const Eigen::Quaterniond& q);
  virtual ~LowerBoundR3() = default;
  virtual double Evaluate(const NodeR3& node);
  virtual double EvaluateAndSet(NodeR3& node);
 private:
  void Evaluate(const NodeR3& node, Eigen::Matrix<double,3,9>& xs,
      Eigen::Matrix<double,9,1>& lbs);
  std::vector<Normal<3>> gmmT_;
};

void ComputeGmmT( const std::vector<Normal<3>>& gmmA, const
    std::vector<Normal<3>>& gmmB, std::vector<Normal<3>>& gmmT, const
    Eigen::Quaterniond& q);

}
