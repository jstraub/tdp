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

template <typename T>
class LowerBoundS3 : public Bound<T,NodeS3<T>> {
 public:
  LowerBoundS3(
      const std::vector<vMF<T,3>>& vmf_mm_A, 
      const std::vector<vMF<T,3>>& vmf_mm_B);
  virtual ~LowerBoundS3() = default;
  virtual T Evaluate(const NodeS3<T>& node);
  virtual T EvaluateAndSet(NodeS3<T>& node);

  void EvaluateRotationSet(const std::vector<Eigen::Quaternion<T>>& qs,
      Eigen::Matrix<T,Eigen::Dynamic,1>& lbs) const;
 private:
//  void Evaluate(const NodeS3& node, std::vector<Eigen::Quaternion<T>>& qs,
//      Eigen::Matrix<T,5,1>& lbs);
  const std::vector<vMF<T,3>>& vmf_mm_A_;
  const std::vector<vMF<T,3>>& vmf_mm_B_;
};

typedef LowerBoundS3<float>  LowerBoundS3f;
typedef LowerBoundS3<double> LowerBoundS3d;

}
