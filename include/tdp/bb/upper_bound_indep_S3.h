
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

class UpperBoundIndepS3 : public Bound<NodeS3> {
 public:
  UpperBoundIndepS3(
      const std::vector<vMF3f>& vmf_mm_A, 
      const std::vector<vMF3f>& vmf_mm_B);
  virtual float Evaluate(const NodeS3& node);
  virtual float EvaluateAndSet(NodeS3& node);
  virtual float EvaluateRotationSet(const
      std::vector<Eigen::Quaternion<float>>& qs) const;
 protected:
  const std::vector<vMF3f>& vmf_mm_A_;
  const std::vector<vMF3f>& vmf_mm_B_;

};

Eigen::Vector3f ComputeExtremumOnGeodesic(const Eigen::Vector3f& q1,
    const Eigen::Vector3f& q2, const Eigen::Vector3f& p, bool verbose);

Eigen::Vector3f ClosestPointInRotationSet(const vMF3f& vmf_A, const
    vMF3f& vmf_B, const std::vector<Eigen::Quaternion<float>>& qs, bool
    furthest=false, bool verbose=false);

Eigen::Vector3f FurthestPointInRotationSet(const vMF3f& vmf_A, const
    vMF3f& vmf_B, const std::vector<Eigen::Quaternion<float>>& qs, 
    bool verbose);

/// This function just calls ClosestPointInRotationSet with the
/// rotations implied by Tetrahedron.
Eigen::Vector3f ClosestPointInTetrahedron(const vMF3f& vmf_A, const
    vMF3f& vmf_B, const Tetrahedron4D& tetrahedron, bool
    furthest=false, bool verbose = false);

Eigen::Vector3f FurthestPointInTetrahedron(const vMF3f& vmf_A, const
    vMF3f& vmf_B, const Tetrahedron4D& tetrahedron, bool verbose = false);

}
