/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tdp/bb/node_S3.h>
#include <tdp/bb/node_AA.h>
#include <tdp/bb/node_TpS3.h>
#include <tdp/bb/numeric_helpers.h>
#include <tdp/bb/bound.h>
#include <tdp/bb/lower_bound_S3.h>

namespace tdp {

template<class NodeLin>
class LowerBoundLin : public Bound<NodeLin> {
 public:
  LowerBoundLin(LowerBoundS3& boundS3);
  virtual ~LowerBoundLin() = default;
  virtual float Evaluate(const NodeLin& node);
  virtual float EvaluateAndSet(NodeLin& node);
 private:
  LowerBoundS3& boundS3_;
};
typedef LowerBoundLin<NodeTpS3> LowerBoundTpS3 ;
typedef LowerBoundLin<NodeAA>   LowerBoundAA   ;
}
#include <tdp/bb/lower_bound_Lin_impl.hpp>
