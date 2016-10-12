
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
class UpperBoundIndepS3 : public Bound<T,NodeS3<T>> {
 public:
  UpperBoundIndepS3(
      const std::vector<vMF<T,3>>& vmf_mm_A, 
      const std::vector<vMF<T,3>>& vmf_mm_B);
  virtual T Evaluate(const NodeS3<T>& node);
  virtual T EvaluateAndSet(NodeS3<T>& node);
  virtual T EvaluateRotationSet(const
      std::vector<Eigen::Quaternion<T>>& qs) const;
 protected:
  const std::vector<vMF<T,3>>& vmf_mm_A_;
  const std::vector<vMF<T,3>>& vmf_mm_B_;

};

typedef UpperBoundIndepS3<float>  UpperBoundIndepS3f;
typedef UpperBoundIndepS3<double> UpperBoundIndepS3d;

template <typename T>
Eigen::Matrix<T,3,1> ComputeExtremumOnGeodesic(const Eigen::Matrix<T,3,1>& q1,
    const Eigen::Matrix<T,3,1>& q2, const Eigen::Matrix<T,3,1>& p, bool verbose);

template <typename T>
Eigen::Matrix<T,3,1> ClosestPointInRotationSet(const vMF<T,3>& vmf_A, const
    vMF<T,3>& vmf_B, const std::vector<Eigen::Quaternion<T>>& qs, bool
    furthest=false, bool verbose=false);

template <typename T>
Eigen::Matrix<T,3,1> FurthestPointInRotationSet(const vMF<T,3>& vmf_A, const
    vMF<T,3>& vmf_B, const std::vector<Eigen::Quaternion<T>>& qs, 
    bool verbose) {
  return ClosestPointInRotationSet<T>(vmf_A, vmf_B, qs, true, verbose);
}

/// This function just calls ClosestPointInRotationSet with the
/// rotations implied by Tetrahedron.
template <typename T>
Eigen::Matrix<T,3,1> ClosestPointInTetrahedron(const vMF<T,3>& vmf_A, const
    vMF<T,3>& vmf_B, const Tetrahedron4D<T>& tetrahedron, bool
    furthest=false, bool verbose = false) {
  std::vector<Eigen::Quaternion<T>> qs(4);
  for (uint32_t i=0; i<4; ++i)
    qs[i] = tetrahedron.GetVertexQuaternion(i);
  return ClosestPointInRotationSet<T>(vmf_A, vmf_B, qs, furthest, verbose);
}

template <typename T>
Eigen::Matrix<T,3,1> FurthestPointInTetrahedron(const vMF<T,3>& vmf_A, const
    vMF<T,3>& vmf_B, const Tetrahedron4D<T>& tetrahedron, bool verbose = false) {
  return ClosestPointInTetrahedron<T>(vmf_A, vmf_B, tetrahedron, true, verbose);
}

}
