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
class NodeAA : public NodeLin<T> {
 public:
  NodeAA(const Box<T>& box, std::vector<uint32_t> ids);
  NodeAA(const NodeAA<T>& node);
  virtual ~NodeAA() = default;
  virtual std::vector<NodeAA<T>> Branch() const;

  std::string GetSpace() const { return "AA"; }
 protected:
  virtual Eigen::Quaternion<T> Project(const Eigen::Matrix<T,3,1>& c) const;
};

typedef NodeAA<float> NodeAAf;
typedef NodeAA<double> NodeAAd;

template<typename T>
std::list<NodeAA<T>> TessellateAA();
}
