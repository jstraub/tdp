/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <tdp/bb/node.h>
#include <tdp/bb/box.h>

namespace tdp {

template<typename T>
class NodeR3 : public BaseNode<T> {
 public:
  NodeR3(const Box<T>& box, std::vector<uint32_t> ids);
  NodeR3(const NodeR3<T>& node);
  virtual ~NodeR3() = default;
  virtual std::vector<NodeR3<T>> Branch() const;
  const Box<T>& GetBox() const { return box_;}
  const Eigen::Matrix<T,3,1>& GetLbArgument() const {return t_lb_;}
  void SetLbArgument(const Eigen::Matrix<T,3,1>& t) {t_lb_ = t;}
  virtual uint32_t GetBranchingFactor(uint32_t i) const { return 8;}
  virtual std::string ToString() const;
  virtual std::string Serialize() const;
  std::string GetSpace() const { return "R3"; }
 protected:
  Box<T> box_;
  Eigen::Matrix<T,3,1> t_lb_;
  virtual T GetVolume_() const { return box_.GetVolume();}
};

typedef NodeR3<float>  NodeR3f;
typedef NodeR3<double> NodeR3d;

template<typename T>
std::list<NodeR3<T>> GenerateNotesThatTessellateR3(const Eigen::Matrix<T,3,1>&
    min, const Eigen::Matrix<T,3,1>& max, T max_side_len); 

}
