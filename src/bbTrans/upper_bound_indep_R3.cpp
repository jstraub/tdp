/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include "bbTrans/upper_bound_indep_R3.h"

namespace bb {

UpperBoundIndepR3::UpperBoundIndepR3(const
    std::vector<Normal<3>>& gmmA, const std::vector<Normal<3>>& gmmB, 
    const Eigen::Quaterniond& q) {
  ComputeGmmT(gmmA, gmmB, gmmT_, q);
}

double UpperBoundIndepR3::Evaluate(const NodeR3& node) {
  double ub = 0.;
  for (auto& gT : gmmT_) {
    Eigen::Vector3d t = FindMinTranslationInNode(gT.GetOmega(),
        gT.GetXi(), node);
    ub += gT.GetPi() * gT.pdf(t);
  }
  return ub;
}

double UpperBoundIndepR3::EvaluateAndSet(NodeR3& node) {
  double ub = Evaluate(node);
  node.SetUB(ub);
  return ub;
}

Eigen::Vector3d FindMinTranslationInNode(const Eigen::Matrix3d& A, 
    const Eigen::Vector3d& b, const NodeR3& node) {
  // Check if the unconstraint maximum lies inside the node.
  // This maximum is the mean of the Gaussian with Information matrix A
  // and Information vector b.
//  Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(A);
  Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
  Eigen::Vector3d t;
  if (lu.rank() == 3) {
    t = lu.solve(b);
    if (node.GetBox().IsInside(t)) 
      return t;
  }
  std::vector<Eigen::Vector3d> ts;
  ts.reserve(26);
  // Check side planes of box.
  for (uint32_t i=0; i<6; ++i) {
    Eigen::Vector3d p0;
    Eigen::Matrix<double, 3,2> E;
    node.GetBox().GetSide(i, p0, E);
    Eigen::FullPivLU<Eigen::Matrix<double,3,2>> lu(A*E);
    if (lu.rank() == 2) {
      Eigen::Vector2d alpha = lu.solve((b-A*p0));
      if ((alpha.array() >= 0.).all() && (alpha.array() <= 1.).all()) {
        ts.push_back(p0+E*alpha);
      }
    }
  }
  // Check edges of box.
  for (uint32_t i=0; i<12; ++i) {
    Eigen::Vector3d e0, d;
    node.GetBox().GetEdge(i, e0, d);
    double alpha = (d.transpose()*b - d.transpose()*A*e0)(0)/(d.transpose()*A*d)(0);
    if (0. <= alpha && alpha <= 1.) {
      ts.push_back(e0+alpha*d);
    }
  }
  // Check corners of box.
  for (uint32_t i=0; i<8; ++i) {
    Eigen::Vector3d c;
    node.GetBox().GetCorner(i, c);
    ts.push_back(c);
  }
  Eigen::VectorXd vals(ts.size());
  for (uint32_t i=0; i<ts.size(); ++i) {
    vals(i) = (ts[i].transpose()*A*ts[i] - 2.*ts[i].transpose()*b)(0);
  }
  uint32_t id_min = 0;
  vals.minCoeff(&id_min);
  return ts[id_min];
}

}
