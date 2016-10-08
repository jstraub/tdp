/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/node.h>

namespace tdp {

BaseNode::BaseNode(std::vector<uint32_t> ids) :
  ids_(ids), lb_(-1e12), ub_(1e12), V_(-1.)
{}

BaseNode::BaseNode(std::vector<uint32_t> ids, float lb,
    float ub) : ids_(ids), lb_(lb), ub_(ub), V_(-1.) {
}

BaseNode::BaseNode(const BaseNode& node) : 
  ids_(node.GetIds()), lb_(node.GetLB()), ub_(node.GetUB()), V_(-1.) {
}

uint64_t BaseNode::GetIdAtLevel(uint32_t lvl) const {
  uint64_t id = 0;
  uint32_t factor = 1;
  for (uint32_t i=0; i < std::min(ids_.size(), size_t(lvl)); ++i) {
    id += ids_[i]*factor;
    factor *= GetBranchingFactor(i);
  } 
  return id;
}

float BaseNode::GetVolume() {
  if (V_ < 0.) V_ = GetVolume_();
  return V_;
}

}
