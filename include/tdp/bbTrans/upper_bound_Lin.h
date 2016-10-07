/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "bbTrans/node_TpS3.h"
#include "bbTrans/node_AA.h"
#include "bbTrans/numeric_helpers.h"
#include "bbTrans/bound.h"
#include "bbTrans/upper_bound_indep_S3.h"
#include "bbTrans/upper_bound_convex_S3.h"

namespace bb {

template<class UpperBound, class NodeLin>
class UpperBoundLin : public Bound<NodeLin> {
 public:
  UpperBoundLin(UpperBound& boundS3);
  virtual ~UpperBoundLin() = default;
  virtual double Evaluate(const NodeLin& node);
  virtual double EvaluateAndSet(NodeLin& node);
 private:
  UpperBound& boundS3_;
};
typedef  UpperBoundLin<UpperBoundConvexS3,NodeTpS3> UpperBoundConvexTpS3;
typedef  UpperBoundLin<UpperBoundIndepS3,NodeTpS3>  UpperBoundIndepTpS3 ;
typedef  UpperBoundLin<UpperBoundConvexS3,NodeAA>   UpperBoundConvexAA  ;
typedef  UpperBoundLin<UpperBoundIndepS3,NodeAA>    UpperBoundIndepAA   ;
}
#include "bbTrans/upper_bound_Lin_impl.hpp"
