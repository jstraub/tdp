/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/bb/lower_bound_S3.h>

namespace tdp {

LowerBoundS3::LowerBoundS3(const vMFMM<3>& vmf_mm_A, const vMFMM<3>&
    vmf_mm_B) 
  : vmf_mm_A_(vmf_mm_A), vmf_mm_B_(vmf_mm_B)
{}

double LowerBoundS3::Evaluate(const NodeS3& node) {
  // at Center only
  std::vector<Eigen::Quaterniond> qs(1);
  qs[0] = node.GetTetrahedron().GetCenterQuaternion();
//  for (uint32_t i=0; i<4; ++i)
//    qs[i+1] = node.GetTetrahedron().GetVertexQuaternion(i);
  Eigen::VectorXd lbs(1);
  EvaluateRotationSet(qs, lbs);
  return lbs(0);
}

double LowerBoundS3::EvaluateAndSet(NodeS3& node) {
  // at Center only
  std::vector<Eigen::Quaterniond> qs(1);
  qs[0] = node.GetTetrahedron().GetCenterQuaternion();
//  for (uint32_t i=0; i<4; ++i)
//    qs[i+1] = node.GetTetrahedron().GetVertexQuaternion(i);
  Eigen::VectorXd lbs(1);
  EvaluateRotationSet(qs, lbs);
  uint32_t id_max = 0;
  double lb = lbs(0); // at Center only
//  double lb = lbs.maxCoeff(&id_max);
  node.SetLB(lb);
  node.SetLbArgument(qs[id_max]);
  return lb;
}

void LowerBoundS3::EvaluateRotationSet(const
    std::vector<Eigen::Quaterniond>& qs, Eigen::VectorXd& lbs) const {
  lbs = Eigen::VectorXd::Zero(qs.size());
  for (uint32_t i=0; i<qs.size(); ++i) {
    Eigen::VectorXd lbElem(vmf_mm_A_.GetK()*vmf_mm_B_.GetK());
    for (std::size_t j=0; j < vmf_mm_A_.GetK(); ++j) {
      for (std::size_t k=0; k < vmf_mm_B_.GetK(); ++k) {
        lbElem(j*vmf_mm_B_.GetK() + k) =
          ComputeLogvMFtovMFcost<3>(vmf_mm_A_.Get(j), vmf_mm_B_.Get(k),
              qs[i]._transformVector(vmf_mm_B_.Get(k).GetMu()));
      }
    }
    lbs(i) = SumExp(lbElem);
    if (this->verbose_)
      std::cout << lbElem.transpose() <<  " " << lbs(i) << std::endl;
  }
}

//void LowerBoundS3::Evaluate(const NodeS3& node,
//  std::vector<Eigen::Quaterniond>& qs, Eigen::Matrix<double,5,1>& lbs) {
//  qs[0] = node.GetTetrahedron().GetCenterQuaternion();
//  for (uint32_t i=0; i<4; ++i)
//    qs[i+1] = node.GetTetrahedron().GetVertexQuaternion(i);
//  for (uint32_t i=0; i<5; ++i) {
//    Eigen::VectorXd lbElem(vmf_mm_A_.GetK()*vmf_mm_B_.GetK());
////    std::cout << qs[i].vec().transpose() << " " << qs[i].w() << std::endl;
//    for (std::size_t j=0; j < vmf_mm_A_.GetK(); ++j) {
////      std::cout << "vMF " << vmf_mm_A_.Get(j).GetMu().transpose()
////        << " " << vmf_mm_A_.Get(j).GetTau() << " " << vmf_mm_A_.Get(j).GetPi()
////        << std::endl;
//      for (std::size_t k=0; k < vmf_mm_B_.GetK(); ++k) {
////        std::cout << "vMF " << vmf_mm_B_.Get(k).GetMu().transpose()
////          << " " << vmf_mm_B_.Get(k).GetTau() << " " << vmf_mm_B_.Get(k).GetPi()
////          << std::endl;
//        lbElem(j*vmf_mm_B_.GetK() + k) =
//          ComputeLogvMFtovMFcost<3>(vmf_mm_A_.Get(j), vmf_mm_B_.Get(k),
//              qs[i]._transformVector(vmf_mm_B_.Get(k).GetMu()));
////              qs[i].toRotationMatrix()*vmf_mm_B_.Get(k).GetMu());
////        std::cout << lbElem(j*vmf_mm_B_.GetK() + k) << std::endl;
////        std::cout << "rotated muB " << qs[i]._transformVector(vmf_mm_B_.Get(k).GetMu()).transpose()
////          << std::endl;
//      }
//    }
//
//    lbs(i) = SumExp(lbElem);
//    if (this->verbose_)
//      std::cout << lbElem.transpose() <<  " " << lbs(i) << std::endl;
//  }
//}

}
