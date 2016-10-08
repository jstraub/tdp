
/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tdp/bb/node_S3.h>
#include <tdp/bb/numeric_helpers.h>
#include <tdp/distributions/vmf.h>
#include <tdp/distributions/vmf_mm.h>
#include <tdp/bb/bound.h>

namespace tdp {

class UpperBoundIndepS3 : public Bound<NodeS3> {
 public:
  UpperBoundIndepS3(const vMFMM<3>& vmf_mm_A, const vMFMM<3>& vmf_mm_B);
  virtual double Evaluate(const NodeS3& node);
  virtual double EvaluateAndSet(NodeS3& node);
  virtual double EvaluateRotationSet(const
      std::vector<Eigen::Quaterniond>& qs) const;
 protected:
  const vMFMM<3>& vmf_mm_A_;
  const vMFMM<3>& vmf_mm_B_;

};

Eigen::Vector3d ComputeExtremumOnGeodesic(const Eigen::Vector3d& q1,
    const Eigen::Vector3d& q2, const Eigen::Vector3d& p, bool verbose);

Eigen::Vector3d ClosestPointInRotationSet(const vMF<3>& vmf_A, const
    vMF<3>& vmf_B, const std::vector<Eigen::Quaterniond>& qs, bool
    furthest=false, bool verbose=false);

Eigen::Vector3d FurthestPointInRotationSet(const vMF<3>& vmf_A, const
    vMF<3>& vmf_B, const std::vector<Eigen::Quaterniond>& qs, 
    bool verbose);

/// This function just calls ClosestPointInRotationSet with the
/// rotations implied by Tetrahedron.
Eigen::Vector3d ClosestPointInTetrahedron(const vMF<3>& vmf_A, const
    vMF<3>& vmf_B, const Tetrahedron4D& tetrahedron, bool
    furthest=false, bool verbose = false);

Eigen::Vector3d FurthestPointInTetrahedron(const vMF<3>& vmf_A, const
    vMF<3>& vmf_B, const Tetrahedron4D& tetrahedron, bool verbose = false);

}
