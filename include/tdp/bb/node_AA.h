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

class NodeAA : public NodeLin {
 public:
  NodeAA(const Box& box, std::vector<uint32_t> ids);
  NodeAA(const NodeAA& node);
  virtual ~NodeAA() = default;
  virtual std::vector<NodeAA> Branch() const;

  std::string GetSpace() const { return "AA"; }
 protected:
  virtual Eigen::Quaternion<float> Project(const Eigen::Vector3f& c) const;
};
std::list<NodeAA> TessellateAA();
}
