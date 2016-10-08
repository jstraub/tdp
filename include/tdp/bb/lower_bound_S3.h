/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tdp/bb/node_S3.h>
#include <tdp/bb/numeric_helpers.h>
#include <tdp/distributions/vmf.h>
#include <tdp/bb/bound.h>

namespace tdp {

class LowerBoundS3 : public Bound<NodeS3> {
 public:
  LowerBoundS3(
      const std::vector<vMF3f>& vmf_mm_A, 
      const std::vector<vMF3f>& vmf_mm_B);
  virtual ~LowerBoundS3() = default;
  virtual float Evaluate(const NodeS3& node);
  virtual float EvaluateAndSet(NodeS3& node);

  void EvaluateRotationSet(const std::vector<Eigen::Quaternion<float>>& qs,
      Eigen::VectorXf& lbs) const;
 private:
//  void Evaluate(const NodeS3& node, std::vector<Eigen::Quaternion<float>>& qs,
//      Eigen::Matrix<float,5,1>& lbs);
  const std::vector<vMF3f>& vmf_mm_A_;
  const std::vector<vMF3f>& vmf_mm_B_;
};

}
