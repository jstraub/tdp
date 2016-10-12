/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/upper_bound_convex_R3.h>

namespace tdp {

template <typename T>
UpperBoundConvexR3<T>::UpperBoundConvexR3(
    const std::vector<Normal<T,3>>& gmmA, 
    const std::vector<Normal<T,3>>& gmmB, 
    const Eigen::Quaternion<T>& q) {
  ComputeGmmT(gmmA, gmmB, gmmT_, q);
}

template <typename T>
T UpperBoundConvexR3<T>::Evaluate(const NodeR3<T>& node) {
  Eigen::Matrix<T,Eigen::Dynamic,1> Aelem = 
    Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(gmmT_.size());
  Eigen::Matrix<T,Eigen::Dynamic,1> belem = 
    Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(gmmT_.size());
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> celem = 
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Zero(gmmT_.size(),2);
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> celemSign = 
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Zero(gmmT_.size(),2);

  for (uint32_t i=0; i<gmmT_.size(); ++i) {
    auto& gT = gmmT_[i];
    Eigen::Matrix<T,3,1> tU = FindMinTranslationInNode(gT.GetOmega(),
        gT.GetXi(), node);
    Eigen::Matrix<T,3,1> tL = FindMaxTranslationInNode(gT.GetOmega(),
        gT.GetXi(), node);
    T L = -0.5*(tL-gT.GetMu()).transpose() *
      gT.GetSigmaLDLT().solve(tL-gT.GetMu());
    T U = -0.5*(tU-gT.GetMu()).transpose() *
      gT.GetSigmaLDLT().solve(tU-gT.GetMu());
    T g = log(1.-exp(L-U)) + U - log(U-L);
    T h = log(U*exp(L-U)-L) + U - log(U-L);
    T D = log(gT.GetPi()) - 1.5*log(2.*M_PI) - 0.5*gT.GetLogDetSigma();
//    A -= 0.5*D*g*gT.GetOmega();
//    b += D*g*gT.GetXi();
//    c += D*(h-0.5*g*(gT.GetMu().transpose()*gT.GetXi())(0));
    Aelem(i) = D + g - log(2.);
    belem(i) = D + g;
    celem.row(i) << D+h, D-log(2)+g;
    celemSign.row(i) << 1., -(gT.GetMu().transpose()*gT.GetXi())(0);
//    std::cout << "-- L<U: " << L << " " << U << std::endl;
//    std::cout << "-- g,h: " << g << " " << h << std::endl;
  }

  Eigen::Matrix3f A;
  for (uint32_t j=0; j<3; ++j)
    for (uint32_t k=0; k<3; ++k) {
      Eigen::Matrix<T,Eigen::Dynamic,1> A_jk_elem(gmmT_.size());
      for (uint32_t i=0; i<gmmT_.size(); ++i)
        A_jk_elem(i) = -gmmT_[i].GetOmega()(j,k);
      A(j,k) = (A_jk_elem.array()*(Aelem.array() -
            Aelem.maxCoeff()).array().exp()).sum() *
        exp(Aelem.maxCoeff());
    }
  Eigen::Matrix<T,3,1> b;
  for (uint32_t j=0; j<3; ++j) {
      Eigen::Matrix<T,Eigen::Dynamic,1> b_j_elem(gmmT_.size());
      for (uint32_t i=0; i<gmmT_.size(); ++i)
        b_j_elem(i) = gmmT_[i].GetXi()(j);
      b(j) = (b_j_elem.array()*(belem.array() -
            belem.maxCoeff()).array().exp()).sum() *
        exp(belem.maxCoeff());
  }
  T c = (celemSign.array()*(celem.array() -
        celem.maxCoeff()).exp()).sum()*exp(celem.maxCoeff());

  Eigen::Matrix<T,3,1> t = FindMinTranslationInNode(-A, 0.5*b, node);
  T ub = (t.transpose()*A*t)(0) + (b.transpose()*t)(0) + c;

  if (this->verbose_) {
    std::cout << "# GMM " << gmmT_.size() << std::endl;
    std::cout << Aelem.transpose() << std::endl
      << belem.transpose() << std::endl
      << celem.transpose() << std::endl;

    std::cout << "- A: \n" << A << std::endl;
    std::cout << "- b: " << b.transpose() << std::endl
      << "t* " << t.transpose() << std::endl;
    std::cout << "- c: " << c << std::endl;
    std::cout << " UB: " << ub << std::endl;
  }
//  std::cout << "- tAt: " <<  (t.transpose()*A*t)(0) 
//    << " bt: " << (b.transpose()*t)(0) << " c: " << c 
//    << " UB: " << ub
//    << std::endl;
  return ub;
}

//T UpperBoundConvexR3<T>::Evaluate(const NodeR3<T>& node) {
//  Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
//  Eigen::Matrix<T,3,1> b = Eigen::Matrix<T,3,1>::Zero();
//  T c = 0.;
//  for (auto& gT : gmmT_) {
//    Eigen::Matrix<T,3,1> tU = FindMinTranslationInNode(gT.GetOmega(),
//        gT.GetXi(), node);
//    Eigen::Matrix<T,3,1> tL = FindMaxTranslationInNode(gT.GetOmega(),
//        gT.GetXi(), node);
//    T L = -0.5*(tL-gT.GetMu()).transpose() *
//      gT.GetSigmaLDLT().solve(tL-gT.GetMu());
//    T U = -0.5*(tU-gT.GetMu()).transpose() *
//      gT.GetSigmaLDLT().solve(tU-gT.GetMu());
//    T g = (1.-exp(L-U))*exp(U)/(U-L);
//    T h = (U*exp(L-U)-L)*exp(U)/(U-L);
//    T D = gT.GetPi() / sqrt((2.*M_PI)*(2.*M_PI)*(2.*M_PI) *
//        exp(gT.GetLogDetSigma()));
//    A -= 0.5*D*g*gT.GetOmega();
//    b += D*g*gT.GetXi();
//    c += D*(h-0.5*g*(gT.GetMu().transpose()*gT.GetXi())(0));
////    std::cout << "-- L<U: " << L << " " << U << std::endl;
////    std::cout << "-- g,h: " << g << " " << h << std::endl;
//  }
//
////  std::cout << "- A: \n" << A << std::endl;
////  std::cout << "- b: " << b.transpose() << std::endl;
////  std::cout << "- c: " << c << std::endl;
//  Eigen::Matrix<T,3,1> t = FindMinTranslationInNode(-A, 0.5*b, node);
////  std::cout << "- t: " << t.transpose() << std::endl;
//  T ub = (t.transpose()*A*t)(0) + (b.transpose()*t)(0) + c;
////  std::cout << "- tAt: " <<  (t.transpose()*A*t)(0) 
////    << " bt: " << (b.transpose()*t)(0) << " c: " << c 
////    << " UB: " << ub
////    << std::endl;
//  return ub;
//}

template <typename T>
T UpperBoundConvexR3<T>::EvaluateAndSet(NodeR3<T>& node) {
  T ub = Evaluate(node);
  node.SetUB(ub);
  return ub;
}

template <typename T>
Eigen::Matrix<T,3,1> FindMaxTranslationInNode(const Eigen::Matrix3f& A, 
    const Eigen::Matrix<T,3,1>& b, const NodeR3<T>& node) {
  // Check corners of box.
  Eigen::Matrix<T,8,1> vals;
  for (uint32_t i=0; i<8; ++i) {
    Eigen::Matrix<T,3,1> t;
    node.GetBox().GetCorner(i, t);
    vals(i) = (t.transpose()*A*t - 2.*t.transpose()*b)(0);
  }
  uint32_t id_max = 0;
  vals.maxCoeff(&id_max);
  Eigen::Matrix<T,3,1> t;
  node.GetBox().GetCorner(id_max, t);
  return t;
}

}
