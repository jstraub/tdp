/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/bb/lower_bound_S3.h>

namespace tdp {

template <typename T>
LowerBoundS3<T>::LowerBoundS3(
    const std::vector<vMF<T,3>>& vmf_mm_A, 
    const std::vector<vMF<T,3>>& vmf_mm_B)
  : vmf_mm_A_(vmf_mm_A), vmf_mm_B_(vmf_mm_B)
{}

template <typename T>
T LowerBoundS3<T>::Evaluate(const NodeS3<T>& node) {
  // at Center only
  std::vector<Eigen::Quaternion<T>> qs(1);
  qs[0] = node.GetTetrahedron().GetCenterQuaternion();
//  for (uint32_t i=0; i<4; ++i)
//    qs[i+1] = node.GetTetrahedron().GetVertexQuaternion(i);
  Eigen::Matrix<T,Eigen::Dynamic,1> lbs(1);
  EvaluateRotationSet(qs, lbs);
  return lbs(0);
}

template <typename T>
T LowerBoundS3<T>::EvaluateAndSet(NodeS3<T>& node) {
  // at Center only
  std::vector<Eigen::Quaternion<T>> qs(1);
  qs[0] = node.GetTetrahedron().GetCenterQuaternion();
//  for (uint32_t i=0; i<4; ++i)
//    qs[i+1] = node.GetTetrahedron().GetVertexQuaternion(i);
  Eigen::Matrix<T,Eigen::Dynamic,1> lbs(1);
  EvaluateRotationSet(qs, lbs);
  uint32_t id_max = 0;
  T lb = lbs(0); // at Center only
//  T lb = lbs.maxCoeff(&id_max);
  node.SetLB(lb);
  node.SetLbArgument(qs[id_max]);
  return lb;
}

template <typename T>
void LowerBoundS3<T>::EvaluateRotationSet(const
    std::vector<Eigen::Quaternion<T>>& qs, 
    Eigen::Matrix<T,Eigen::Dynamic,1>& lbs) const {
  lbs = Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qs.size());
  for (uint32_t i=0; i<qs.size(); ++i) {
    Eigen::Matrix<T,Eigen::Dynamic,1> lbElem(vmf_mm_A_.size()*vmf_mm_B_.size());
    for (std::size_t j=0; j < vmf_mm_A_.size(); ++j) {
      for (std::size_t k=0; k < vmf_mm_B_.size(); ++k) {
        lbElem(j*vmf_mm_B_.size() + k) =
          ComputeLogvMFtovMFcost<T,3>(vmf_mm_A_[j], vmf_mm_B_[k],
              qs[i]._transformVector(vmf_mm_B_[k].GetMu()));
      }
    }
    lbs(i) = SumExp(lbElem);
    if (this->verbose_)
      std::cout << lbElem.transpose() <<  " " << lbs(i) << std::endl;
  }
}

template class LowerBoundS3<float>;
template class LowerBoundS3<double>;

//void LowerBoundS3<T>::Evaluate(const NodeS3<T>& node,
//  std::vector<Eigen::Quaternion<T>>& qs, Eigen::Matrix<T,5,1>& lbs) {
//  qs[0] = node.GetTetrahedron().GetCenterQuaternion();
//  for (uint32_t i=0; i<4; ++i)
//    qs[i+1] = node.GetTetrahedron().GetVertexQuaternion(i);
//  for (uint32_t i=0; i<5; ++i) {
//    Eigen::Matrix<T,Eigen::Dynamic,1> lbElem(vmf_mm_A_.size()*vmf_mm_B_.size());
////    std::cout << qs[i].vec().transpose() << " " << qs[i].w() << std::endl;
//    for (std::size_t j=0; j < vmf_mm_A_.size(); ++j) {
////      std::cout << "vMF " << vmf_mm_A_[j].GetMu().transpose()
////        << " " << vmf_mm_A_[j].GetTau() << " " << vmf_mm_A_[j].GetPi()
////        << std::endl;
//      for (std::size_t k=0; k < vmf_mm_B_.size(); ++k) {
////        std::cout << "vMF " << vmf_mm_B_[k].GetMu().transpose()
////          << " " << vmf_mm_B_[k].GetTau() << " " << vmf_mm_B_[k].GetPi()
////          << std::endl;
//        lbElem(j*vmf_mm_B_.size() + k) =
//          ComputeLogvMFtovMFcost<3>(vmf_mm_A_[j], vmf_mm_B_[k],
//              qs[i]._transformVector(vmf_mm_B_[k].GetMu()));
////              qs[i].toRotationMatrix()*vmf_mm_B_[k].GetMu());
////        std::cout << lbElem(j*vmf_mm_B_.size() + k) << std::endl;
////        std::cout << "rotated muB " << qs[i]._transformVector(vmf_mm_B_[k].GetMu()).transpose()
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
