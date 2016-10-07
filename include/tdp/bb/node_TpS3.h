/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <vector>
#include <list>
#include <sstream>
#include <string>

#include "manifold/S.h"
#include "bbTrans/node_Lin.h"
#include "bbTrans/node_R3.h"
#include "bbTrans/node_S3.h"
#include "bbTrans/box.h"
#include "bbTrans/tetrahedron.h"

namespace bb {

class NodeTpS3 : public NodeLin {
 public:
  NodeTpS3(const Box& box, std::vector<uint32_t> ids);
  NodeTpS3(const NodeTpS3& node);
  virtual ~NodeTpS3() = default;
  virtual std::vector<NodeTpS3> Branch() const;

  std::string GetSpace() const { return "TpS3"; }
 protected:
  virtual Eigen::Quaterniond Project(const Eigen::Vector3d& c) const;
};

std::list<NodeTpS3> TessellateTpS3();
}
