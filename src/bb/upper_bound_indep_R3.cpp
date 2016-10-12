/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/upper_bound_indep_R3.h>

namespace tdp {

template <typename T>
UpperBoundIndepR3<T>::UpperBoundIndepR3(const
    std::vector<Normal<T,3>>& gmmA, const std::vector<Normal<T,3>>& gmmB, 
    const Eigen::Quaternion<T>& q) {
  ComputeGmmT(gmmA, gmmB, gmmT_, q);
}

template <typename T>
T UpperBoundIndepR3<T>::Evaluate(const NodeR3<T>& node) {
  T ub = 0.;
  for (auto& gT : gmmT_) {
    Eigen::Matrix<T,3,1> t = FindMinTranslationInNode(gT.GetOmega(),
        gT.GetXi(), node);
    ub += gT.GetPi() * gT.pdf(t);
  }
  return ub;
}

template <typename T>
T UpperBoundIndepR3<T>::EvaluateAndSet(NodeR3<T>& node) {
  T ub = Evaluate(node);
  node.SetUB(ub);
  return ub;
}

template <typename T>
Eigen::Matrix<T,3,1> FindMinTranslationInNode(const Eigen::Matrix3f& A, 
    const Eigen::Matrix<T,3,1>& b, const NodeR3<T>& node) {
  // Check if the unconstraint maximum lies inside the node.
  // This maximum is the mean of the Gaussian with Information matrix A
  // and Information vector b.
//  Eigen::ColPivHouseholderQR<Eigen::Matrix3f> qr(A);
  Eigen::FullPivLU<Eigen::Matrix3f> lu(A);
  Eigen::Matrix<T,3,1> t;
  if (lu.rank() == 3) {
    t = lu.solve(b);
    if (node.GetBox().IsInside(t)) 
      return t;
  }
  std::vector<Eigen::Matrix<T,3,1>> ts;
  ts.reserve(26);
  // Check side planes of box.
  for (uint32_t i=0; i<6; ++i) {
    Eigen::Matrix<T,3,1> p0;
    Eigen::Matrix<T, 3,2> E;
    node.GetBox().GetSide(i, p0, E);
    Eigen::FullPivLU<Eigen::Matrix<T,3,2>> lu(A*E);
    if (lu.rank() == 2) {
      Eigen::Matrix<T,2,1> alpha = lu.solve((b-A*p0));
      if ((alpha.array() >= 0.).all() && (alpha.array() <= 1.).all()) {
        ts.push_back(p0+E*alpha);
      }
    }
  }
  // Check edges of box.
  for (uint32_t i=0; i<12; ++i) {
    Eigen::Matrix<T,3,1> e0, d;
    node.GetBox().GetEdge(i, e0, d);
    T alpha = (d.transpose()*b - d.transpose()*A*e0)(0)/(d.transpose()*A*d)(0);
    if (0. <= alpha && alpha <= 1.) {
      ts.push_back(e0+alpha*d);
    }
  }
  // Check corners of box.
  for (uint32_t i=0; i<8; ++i) {
    Eigen::Matrix<T,3,1> c;
    node.GetBox().GetCorner(i, c);
    ts.push_back(c);
  }
  Eigen::VectorXf vals(ts.size());
  for (uint32_t i=0; i<ts.size(); ++i) {
    vals(i) = (ts[i].transpose()*A*ts[i] - 2.*ts[i].transpose()*b)(0);
  }
  uint32_t id_min = 0;
  vals.minCoeff(&id_min);
  return ts[id_min];
}

}
