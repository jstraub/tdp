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

#include <tdp/manifold/S.h>
#include <tdp/bb/node_R3.h>
#include <tdp/bb/node_S3.h>
#include <tdp/bb/box.h>
#include <tdp/bb/tetrahedron.h>

namespace tdp {

template<typename T>
class NodeLin : public BaseNode<T> {
 public:
  NodeLin(const Box<T>& box, std::vector<uint32_t> ids);
  NodeLin(const NodeLin<T>& node);
  virtual ~NodeLin() = default;

  void SetLbArgument(const Eigen::Quaternion<T>& q) {q_lb_ = q;}
  Eigen::Quaternion<T> GetLbArgument() const {return q_lb_;}
  Eigen::Quaternion<T> GetCenter() const;
  virtual uint32_t GetBranchingFactor(uint32_t i) const { return 8;}
  virtual std::string ToString() const;
  virtual std::string Serialize() const;
  std::string GetSpace() const { return "Lin"; }
  NodeS3<T> GetNodeS3() const;
  const std::vector<Eigen::Quaternion<T>>& GetQuaternions() const { return qs_; }
  std::vector<Eigen::Quaternion<T>>& GetQuaternions() { return qs_; }
 protected:
  NodeR3<T> nodeLin_;
  std::vector<Eigen::Quaternion<T>> qs_;
  Eigen::Quaternion<T> q_lb_;
  virtual T GetVolume_() const;
  virtual void Linearize(const Box<T>& box);
  virtual Eigen::Quaternion<T> Project(const Eigen::Matrix<T,3,1>& c) const = 0;
  Tetrahedron4D<T> TetraFromBox(const Box<T>& box, uint32_t i,
      uint32_t j, uint32_t k, uint32_t l) const;
  static Eigen::Matrix<T,4,1> QuaternionToVec(const Eigen::Quaternion<T>& q);
};
}
