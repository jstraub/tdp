/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/upper_bound_indep_R3.h>

namespace tdp {

UpperBoundIndepR3::UpperBoundIndepR3(const
    std::vector<Normal3f>& gmmA, const std::vector<Normal3f>& gmmB, 
    const Eigen::Quaternion<float>& q) {
  ComputeGmmT(gmmA, gmmB, gmmT_, q);
}

float UpperBoundIndepR3::Evaluate(const NodeR3& node) {
  float ub = 0.;
  for (auto& gT : gmmT_) {
    Eigen::Vector3f t = FindMinTranslationInNode(gT.GetOmega(),
        gT.GetXi(), node);
    ub += gT.GetPi() * gT.pdf(t);
  }
  return ub;
}

float UpperBoundIndepR3::EvaluateAndSet(NodeR3& node) {
  float ub = Evaluate(node);
  node.SetUB(ub);
  return ub;
}

Eigen::Vector3f FindMinTranslationInNode(const Eigen::Matrix3f& A, 
    const Eigen::Vector3f& b, const NodeR3& node) {
  // Check if the unconstraint maximum lies inside the node.
  // This maximum is the mean of the Gaussian with Information matrix A
  // and Information vector b.
//  Eigen::ColPivHouseholderQR<Eigen::Matrix3f> qr(A);
  Eigen::FullPivLU<Eigen::Matrix3f> lu(A);
  Eigen::Vector3f t;
  if (lu.rank() == 3) {
    t = lu.solve(b);
    if (node.GetBox().IsInside(t)) 
      return t;
  }
  std::vector<Eigen::Vector3f> ts;
  ts.reserve(26);
  // Check side planes of box.
  for (uint32_t i=0; i<6; ++i) {
    Eigen::Vector3f p0;
    Eigen::Matrix<float, 3,2> E;
    node.GetBox().GetSide(i, p0, E);
    Eigen::FullPivLU<Eigen::Matrix<float,3,2>> lu(A*E);
    if (lu.rank() == 2) {
      Eigen::Vector2f alpha = lu.solve((b-A*p0));
      if ((alpha.array() >= 0.).all() && (alpha.array() <= 1.).all()) {
        ts.push_back(p0+E*alpha);
      }
    }
  }
  // Check edges of box.
  for (uint32_t i=0; i<12; ++i) {
    Eigen::Vector3f e0, d;
    node.GetBox().GetEdge(i, e0, d);
    float alpha = (d.transpose()*b - d.transpose()*A*e0)(0)/(d.transpose()*A*d)(0);
    if (0. <= alpha && alpha <= 1.) {
      ts.push_back(e0+alpha*d);
    }
  }
  // Check corners of box.
  for (uint32_t i=0; i<8; ++i) {
    Eigen::Vector3f c;
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
