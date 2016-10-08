/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <vector>
#include <list>
#include <sstream>
#include <string>
#include <tdp/bb/node.h>
#include <tdp/bb/tetrahedron.h>
#include <tdp/bb/s3_tessellation.h>

namespace tdp {

class NodeS3 : public BaseNode {
 public:
  NodeS3(const Tetrahedron4D& tetrahedron, std::vector<uint32_t> ids);
  NodeS3(const NodeS3& node);
  virtual ~NodeS3() = default;
  virtual std::vector<NodeS3> Branch() const;
  const Tetrahedron4D& GetTetrahedron() const { return tetrahedron_;}
  void SetLbArgument(const Eigen::Quaternion<float>& q) {q_lb_ = q;}
  Eigen::Quaternion<float> GetLbArgument() const {return q_lb_;}
  virtual uint32_t GetBranchingFactor(uint32_t i) const { return i==0? 600 : 8;}
  virtual std::string ToString() const;
  virtual std::string Serialize() const;
  std::string GetSpace() const { return "S3"; }
 protected:
  Tetrahedron4D tetrahedron_;
  Eigen::Quaternion<float> q_lb_;
  virtual float GetVolume_() const { return tetrahedron_.GetVolume();}
};

std::list<NodeS3> GenerateNotesThatTessellateS3();
}
