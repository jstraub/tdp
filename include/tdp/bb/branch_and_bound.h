/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <memory>
#include <vector>
#include <list>
#include <fstream>
#include <tdp/bb/node.h>
#include <tdp/bb/bound.h>
#include <tdp/utils/timer.hpp>

namespace tdp {

template <typename T, class Node>
class BranchAndBound {
 public:
  BranchAndBound(Bound<T,Node>& lower_bound, Bound<T,Node>& upper_bound);
  ~BranchAndBound() = default;
  Node Compute(std::list<Node>& nodes, T eps, uint32_t max_lvl,
      uint32_t max_it);
 private:
  Bound<T,Node>& lower_bound_;
  Bound<T,Node>& upper_bound_;
  uint32_t BoundAndPrune(std::list<Node>& nodes, T& lb, T&
      ub, T eps);

  typename std::list<Node>::iterator FindBestNodeToExplore(
      ::std::list<Node>& nodes, T eps);
  typename std::list<Node>::iterator FindBestNode(
      ::std::list<Node>& nodes, T eps);

  void WriteStats(std::ofstream& out, std::list<Node>& nodes, T
      lb, T ub, T dt, typename std::list<Node>::iterator& node_star);
  void WriteNodes(std::ofstream& out, std::list<Node>& nodes, T
      lb, T ub);
};

}
#include <tdp/bb/branch_and_bound_impl.h>
