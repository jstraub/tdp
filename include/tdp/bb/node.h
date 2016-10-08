/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <vector>
#include <string>
#include <list>
#include <map>
#include <iostream>

namespace tdp {

class BaseNode {
 public:
  BaseNode(::std::vector<uint32_t> ids);
  BaseNode(::std::vector<uint32_t> ids, float lb, float ub);
  BaseNode(const BaseNode& node);
  virtual ~BaseNode() = default;
//  virtual std::vector<std::unique_ptr<BaseNode>> Branch() const = 0;
  uint32_t GetLevel() const {return ids_.size()-1;}
  ::std::vector<uint32_t> GetIds() const {return ids_;}
  float GetUB() const { return ub_;}
  float GetLB() const { return lb_;}
  void SetUB(float ub) { ub_ = ub;}
  void SetLB(float lb) { lb_ = lb;}
  float GetBoundGap() const {return ub_-lb_;}
  uint64_t GetIdAtLevel(uint32_t lvl) const;
  virtual uint32_t GetBranchingFactor(uint32_t i) const = 0;
  virtual std::string ToString() const = 0;
  virtual float GetVolume();
 protected:
  ::std::vector<uint32_t> ids_;
  float lb_;
  float ub_;
  float V_;
  virtual float GetVolume_() const = 0;
};

// For use with std::forward_list::remove_if
template <class Node>
class IsPrunableNode {
 public:
  IsPrunableNode(float lb) : lb_(lb) {}
  bool operator() (const Node& node) {return node.GetUB() < lb_;}
 private:
  float lb_;
};

template <class Node>
struct LessThanNodeUB {
  bool operator() (const Node& node_a, const Node& node_b) 
  {return node_a.GetUB() < node_b.GetUB();}
};

template <class Node>
struct LessThanNodeLB {
  bool operator() (const Node& node_a, const Node& node_b) 
  {return node_a.GetLB() < node_b.GetLB();}
};

//template <class Node>
//struct LessThanNodeLBAndTighter {
//  bool operator() (const Node& node_a, const Node& node_b) 
//  {return (node_a.GetLB() < node_b.GetLB()) 
//    && (node_a.GetUB() - node_a.GetLB() > node_b.GetUB() - node_b.GetLB());}
//};

template<class Node>
std::vector<uint32_t> CountBranchesInTree(const std::list<Node>& nodes);

}
#include <tdp/bb/node_impl.h>
