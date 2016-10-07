/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace tdp {

template<class UpperBound, class NodeLin>
UpperBoundLin<UpperBound,NodeLin>::UpperBoundLin(UpperBound& boundS3) 
  : boundS3_(boundS3)
{ }

template<class UpperBound, class NodeLin>
double UpperBoundLin<UpperBound,NodeLin>::Evaluate(const NodeLin& node) {
//  double ub = -1e99;
  return boundS3_.EvaluateRotationSet(node.GetQuaternions());
//    std::cout << ub_i << " ";
//    if (ub_i > ub)
//      ub = ub_i;
//  }
//  std::cout << ub << std::endl;
//  return ubs.maxCoeff();
}

template<class UpperBound, class NodeLin>
double UpperBoundLin<UpperBound,NodeLin>::EvaluateAndSet(NodeLin& node) {
  double ub = Evaluate(node);
  node.SetUB(ub);
  return ub;
}

}
