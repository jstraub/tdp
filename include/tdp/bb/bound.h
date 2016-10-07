/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <tdp/bb/node.h>

namespace tdp {

template <class Node>
class Bound {
 public:
  Bound() : verbose_(false) {};
  virtual ~Bound() = default;
  virtual double Evaluate(const Node& node) { return 0;}
  virtual double EvaluateAndSet(Node& node) { return 0;};
  virtual void ToggleVerbose() {verbose_ = verbose_?false:true;}
 protected:
  bool verbose_;
};
}
