/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/lower_bound_R3.h>

namespace tdp {

LowerBoundR3::LowerBoundR3(const
    std::vector<Normal3f>& gmmA, const std::vector<Normal3f>& gmmB, 
    const Eigen::Quaternion<float>& q) {
  ComputeGmmT(gmmA, gmmB, gmmT_, q);
}

float LowerBoundR3::Evaluate(const NodeR3& node) {
  Eigen::Matrix<float,3,9> xs;
  Eigen::Matrix<float,9,1> lbs;
  Evaluate(node, xs, lbs);
  return lbs(0); // at center
//  return lbs.maxCoeff();
}

float LowerBoundR3::EvaluateAndSet(NodeR3& node) {
  Eigen::Matrix<float,3,9> xs;
  Eigen::Matrix<float,9,1> lbs;
  Evaluate(node, xs, lbs);
  uint32_t id_max = 0;
//  float lb = lbs.maxCoeff(&id_max);
  float lb = lbs(id_max);
  node.SetLB(lb);
  node.SetLbArgument(xs.col(id_max));
  return lb;
}

void LowerBoundR3::Evaluate(const NodeR3& node,
    Eigen::Matrix<float,3,9>& xs, Eigen::Matrix<float,9,1>& lbs) {
  xs.col(0) = node.GetBox().GetCenter();
  Eigen::Vector3f c;
  for (uint32_t i=0; i<8; ++i) {
    node.GetBox().GetCorner(i,c);
    xs.col(i+1) = c;
  }
  lbs = Eigen::VectorXf::Zero(9);
  Eigen::VectorXf lbelem(gmmT_.size());
  for (uint32_t i=0; i<9; ++i) {
    for (uint32_t j=0; j< gmmT_.size(); ++j) {
      lbelem(j) = log(gmmT_[j].GetPi()) + gmmT_[j].logPdf(xs.col(i));
      //      lbs(i) += gT.GetPi() * gT.pdf(xs.col(i));
    }
    lbs(i) = SumExp(lbelem);
  }
}

void ComputeGmmT( const std::vector<Normal3f>& gmmA, const
    std::vector<Normal3f>& gmmB, std::vector<Normal3f>& gmmT, const
    Eigen::Quaternion<float>& q) {
  gmmT.reserve(gmmA.size() * gmmB.size());
  Eigen::Matrix3f R = q.toRotationMatrix();
  for (auto& gA : gmmA) 
    for (auto& gB : gmmB) {
      gmmT.push_back(
          Normal3f(gB.GetMu() - R*gA.GetMu(),
            R*gA.GetSigma()*R.transpose() + gB.GetSigma(), 
            gB.GetPi()*gA.GetPi()));
    }
}

}
