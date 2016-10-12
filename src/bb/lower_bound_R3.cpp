/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/lower_bound_R3.h>

namespace tdp {

template <typename T>
LowerBoundR3<T>::LowerBoundR3(const
    std::vector<Normal<T,3>>& gmmA, const std::vector<Normal<T,3>>& gmmB, 
    const Eigen::Quaternion<T>& q) {
  ComputeGmmT(gmmA, gmmB, gmmT_, q);
}

template <typename T>
T LowerBoundR3<T>::Evaluate(const NodeR3<T>& node) {
  Eigen::Matrix<T,3,9> xs;
  Eigen::Matrix<T,9,1> lbs;
  Evaluate(node, xs, lbs);
  return lbs(0); // at center
//  return lbs.maxCoeff();
}

template <typename T>
T LowerBoundR3<T>::EvaluateAndSet(NodeR3<T>& node) {
  Eigen::Matrix<T,3,9> xs;
  Eigen::Matrix<T,9,1> lbs;
  Evaluate(node, xs, lbs);
  uint32_t id_max = 0;
//  T lb = lbs.maxCoeff(&id_max);
  T lb = lbs(id_max);
  node.SetLB(lb);
  node.SetLbArgument(xs.col(id_max));
  return lb;
}

template <typename T>
void LowerBoundR3<T>::Evaluate(const NodeR3<T>& node,
    Eigen::Matrix<T,3,9>& xs, Eigen::Matrix<T,9,1>& lbs) {
  xs.col(0) = node.GetBox().GetCenter();
  Eigen::Matrix<T,3,1> c;
  for (uint32_t i=0; i<8; ++i) {
    node.GetBox().GetCorner(i,c);
    xs.col(i+1) = c;
  }
  lbs = Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(9);
  Eigen::Matrix<T,Eigen::Dynamic,1> lbelem(gmmT_.size());
  for (uint32_t i=0; i<9; ++i) {
    for (uint32_t j=0; j< gmmT_.size(); ++j) {
      lbelem(j) = log(gmmT_[j].GetPi()) + gmmT_[j].logPdf(xs.col(i));
      //      lbs(i) += gT.GetPi() * gT.pdf(xs.col(i));
    }
    lbs(i) = SumExp(lbelem);
  }
}

template <typename T>
void ComputeGmmT( const std::vector<Normal<T,3>>& gmmA, 
    const std::vector<Normal<T,3>>& gmmB, 
    std::vector<Normal<T,3>>& gmmT, 
    const Eigen::Quaternion<T>& q) {
  gmmT.reserve(gmmA.size() * gmmB.size());
  Eigen::Matrix<T,3,3> R = q.toRotationMatrix();
  for (auto& gA : gmmA) 
    for (auto& gB : gmmB) {
      gmmT.push_back(
          Normal<T,3>(gB.GetMu() - R*gA.GetMu(),
            R*gA.GetSigma()*R.transpose() + gB.GetSigma(), 
            gB.GetPi()*gA.GetPi()));
    }
}

template class LowerBoundR3<float>;
template class LowerBoundR3<double>;

}
