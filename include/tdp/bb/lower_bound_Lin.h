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

template<typename T, class NodeLin>
class LowerBoundLin : public Bound<T,NodeLin> {
 public:
  LowerBoundLin(LowerBoundS3<T>& boundS3);
  virtual ~LowerBoundLin() = default;
  virtual T Evaluate(const NodeLin& node);
  virtual T EvaluateAndSet(NodeLin& node);
 private:
  LowerBoundS3<T>& boundS3_;
};
typedef LowerBoundLin<float,NodeTpS3<float>> LowerBoundTpS3f ;
typedef LowerBoundLin<float,NodeAA<float>>   LowerBoundAAf   ;

typedef LowerBoundLin<double,NodeTpS3<double>> LowerBoundTpS3d ;
typedef LowerBoundLin<double,NodeAA<double>>   LowerBoundAAd   ;
}
#include <tdp/bb/lower_bound_Lin_impl.hpp>
