/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <vector>
#include <list>
#include <sstream>
#include <string>

#include <tdp/manifold/S.h>
#include <tdp/bb/node_Lin.h>
#include <tdp/bb/node_R3.h>
#include <tdp/bb/node_S3.h>
#include <tdp/bb/box.h>
#include <tdp/bb/tetrahedron.h>

namespace tdp {

template<typename T>
class NodeTpS3 : public NodeLin<T> {
 public:
  NodeTpS3(const Box<T>& box, std::vector<uint32_t> ids);
  NodeTpS3(const NodeTpS3<T>& node);
  virtual ~NodeTpS3() = default;
  virtual std::vector<NodeTpS3<T>> Branch() const;

  std::string GetSpace() const { return "TpS3"; }
 protected:
  virtual Eigen::Quaternion<T> Project(const Eigen::Matrix<T,3,1>& c) const;
};

typedef NodeTpS3<float>  NodeTpS3f;
typedef NodeTpS3<double> NodeTpS3d;

template<typename T>
std::list<NodeTpS3<T>> TessellateTpS3();
}
