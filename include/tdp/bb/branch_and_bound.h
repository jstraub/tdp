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
#include <jsCore/timer.hpp>

namespace tdp {

template <class Node>
class BranchAndBound {
 public:
  BranchAndBound(Bound<Node>& lower_bound, Bound<Node>& upper_bound);
  ~BranchAndBound() = default;
  Node Compute(std::list<Node>& nodes, double eps, uint32_t max_lvl,
      uint32_t max_it);
 private:
  Bound<Node>& lower_bound_;
  Bound<Node>& upper_bound_;
  uint32_t BoundAndPrune(std::list<Node>& nodes, double& lb, double&
      ub, double eps);

  typename std::list<Node>::iterator FindBestNodeToExplore(std::list<Node>& nodes, double eps);
  typename std::list<Node>::iterator FindBestNode(std::list<Node>& nodes, double eps);

  void WriteStats(std::ofstream& out, std::list<Node>& nodes, double
      lb, double ub, double dt, typename std::list<Node>::iterator& node_star);
  void WriteNodes(std::ofstream& out, std::list<Node>& nodes, double
      lb, double ub);
};
}
#include <tdp/bb/branch_and_bound_impl.h>
