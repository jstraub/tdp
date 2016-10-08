/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tdp/bb/node_TpS3.h>
#include <tdp/bb/node_AA.h>
#include <tdp/bb/numeric_helpers.h>
#include <tdp/bb/bound.h>
#include <tdp/bb/upper_bound_indep_S3.h>
#include <tdp/bb/upper_bound_convex_S3.h>

namespace tdp {

template<class UpperBound, class NodeLin>
class UpperBoundLin : public Bound<NodeLin> {
 public:
  UpperBoundLin(UpperBound& boundS3);
  virtual ~UpperBoundLin() = default;
  virtual float Evaluate(const NodeLin& node);
  virtual float EvaluateAndSet(NodeLin& node);
 private:
  UpperBound& boundS3_;
};
typedef  UpperBoundLin<UpperBoundConvexS3,NodeTpS3> UpperBoundConvexTpS3;
typedef  UpperBoundLin<UpperBoundIndepS3,NodeTpS3>  UpperBoundIndepTpS3 ;
typedef  UpperBoundLin<UpperBoundConvexS3,NodeAA>   UpperBoundConvexAA  ;
typedef  UpperBoundLin<UpperBoundIndepS3,NodeAA>    UpperBoundIndepAA   ;
}
#include <tdp/bb/upper_bound_Lin_impl.hpp>
