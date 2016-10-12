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

template<typename T>
class BaseNode {
 public:
  BaseNode(::std::vector<uint32_t> ids);
  BaseNode(::std::vector<uint32_t> ids, T lb, T ub);
  BaseNode(const BaseNode& node);
  virtual ~BaseNode() = default;
//  virtual std::vector<std::unique_ptr<BaseNode>> Branch() const = 0;
  uint32_t GetLevel() const {return ids_.size()-1;}
  ::std::vector<uint32_t> GetIds() const {return ids_;}
  T GetUB() const { return ub_;}
  T GetLB() const { return lb_;}
  void SetUB(T ub) { ub_ = ub;}
  void SetLB(T lb) { lb_ = lb;}
  T GetBoundGap() const {return ub_-lb_;}
  uint64_t GetIdAtLevel(uint32_t lvl) const;
  virtual uint32_t GetBranchingFactor(uint32_t i) const = 0;
  virtual std::string ToString() const = 0;
  virtual T GetVolume();
 protected:
  ::std::vector<uint32_t> ids_;
  T lb_;
  T ub_;
  T V_;
  virtual T GetVolume_() const = 0;
};

// For use with std::forward_list::remove_if
template <typename T, class Node>
class IsPrunableNode {
 public:
  IsPrunableNode(T lb) : lb_(lb) {}
  bool operator() (const Node& node) {return node.GetUB() < lb_;}
 private:
  T lb_;
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
