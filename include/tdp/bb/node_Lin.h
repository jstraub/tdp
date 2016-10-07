/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <vector>
#include <list>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <manifold/S.h>
#include <tdp/bb/node_R3.h>
#include <tdp/bb/node_S3.h>
#include <tdp/bb/box.h>
#include <tdp/bb/tetrahedron.h>

namespace tdp {

class NodeLin : public BaseNode {
 public:
  NodeLin(const Box& box, std::vector<uint32_t> ids);
  NodeLin(const NodeLin& node);
  virtual ~NodeLin() = default;

  void SetLbArgument(const Eigen::Quaterniond& q) {q_lb_ = q;}
  Eigen::Quaterniond GetLbArgument() const {return q_lb_;}
  Eigen::Quaterniond GetCenter() const;
  virtual uint32_t GetBranchingFactor(uint32_t i) const { return 8;}
  virtual std::string ToString() const;
  virtual std::string Serialize() const;
  std::string GetSpace() const { return "Lin"; }
  NodeS3 GetNodeS3() const;
  const std::vector<Eigen::Quaterniond>& GetQuaternions() const { return qs_; }
  std::vector<Eigen::Quaterniond>& GetQuaternions() { return qs_; }
 protected:
  NodeR3 nodeLin_;
  std::vector<Eigen::Quaterniond> qs_;
  Eigen::Quaterniond q_lb_;
  virtual double GetVolume_() const;
  virtual void Linearize(const Box& box);
  virtual Eigen::Quaterniond Project(const Eigen::Vector3d& c) const = 0;
  Tetrahedron4D TetraFromBox(const Box& box, uint32_t i,
      uint32_t j, uint32_t k, uint32_t l) const;
  static Eigen::Vector4d QuaternionToVec(const Eigen::Quaterniond& q);
};
}
