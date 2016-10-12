/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/bb/upper_bound_convex_S3.h>

namespace tdp {

template <typename T>
UpperBoundConvexS3<T>::UpperBoundConvexS3(
      const std::vector<vMF<T,3>>& vmf_mm_A, 
      const std::vector<vMF<T,3>>& vmf_mm_B)
  : vmf_mm_A_(vmf_mm_A), vmf_mm_B_(vmf_mm_B)
{}

template <typename T>
Eigen::Matrix<T,4,4> UpperBoundConvexS3<T>::BuildM(const
    Eigen::Matrix<T,3,1>& u, const Eigen::Matrix<T,3,1>& v) {
  const T ui = u(0);
  const T uj = u(1);
  const T uk = u(2);
  const T vi = v(0);
  const T vj = v(1);
  const T vk = v(2);
  Eigen::Matrix<T,4,4> M;
  M << u.transpose()*v, uk*vj-uj*vk,       ui*vk-uk*vi,       uj*vi-ui*vj, 
       uk*vj-uj*vk,     ui*vi-uj*vj-uk*vk, uj*vi+ui*vj,       ui*vk+uk*vi,
       ui*vk-uk*vi,     uj*vi+ui*vj,       uj*vj-ui*vi-uk*vk, uj*vk+uk*vj,
       uj*vi-ui*vj,     ui*vk+uk*vi,       uj*vk+uk*vj,       uk*vk-ui*vi-uj*vj;
  return M;
}

template <typename T>
T UpperBoundConvexS3<T>::Evaluate(const NodeS3<T>& node) {
  std::vector<Eigen::Quaternion<T>> qs(4);
  for (uint32_t i=0; i<4; ++i)
    qs[i] = node.GetTetrahedron().GetVertexQuaternion(i);
  return EvaluateRotationSet(qs);
}

template <typename T>
T UpperBoundConvexS3<T>::EvaluateAndSet(NodeS3<T>& node) {
  T ub = Evaluate(node);
  node.SetUB(ub);
  return ub;
}

template <typename T>
T UpperBoundConvexS3<T>::EvaluateRotationSet(const
    std::vector<Eigen::Quaternion<T>>& qs) const {

  Eigen::Matrix<T,4,Eigen::Dynamic> Q(4,qs.size());
  for (uint32_t i=0; i<qs.size(); ++i) {
    Q(0,i) = qs[i].w();
    Q.template block<3,1>(1,i) = qs[i].vec();
//    std::cout << Q.col(i).norm() << " ";
  }
//  std::cout << std::endl << Q << std::endl;

  std::vector<Eigen::Matrix<T,4,4>> Melem(vmf_mm_A_.size()*vmf_mm_B_.size());
  Eigen::Matrix<T,Eigen::Dynamic,1> Aelem(vmf_mm_A_.size()*vmf_mm_B_.size());
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> Belem(vmf_mm_A_.size()*vmf_mm_B_.size(),4);
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> BelemSign(vmf_mm_A_.size()*vmf_mm_B_.size(),4);

  for (std::size_t j=0; j < vmf_mm_A_.size(); ++j)
    for (std::size_t k=0; k < vmf_mm_B_.size(); ++k) {
      const vMF<T,3>& vmf_A = vmf_mm_A_[j];
      const vMF<T,3>& vmf_B = vmf_mm_B_[k];
      Eigen::Matrix<T,3,1> p_U = ClosestPointInRotationSet(vmf_A,
          vmf_B, qs, false, this->verbose_);
      Eigen::Matrix<T,3,1> p_L = FurthestPointInRotationSet(vmf_A,
          vmf_B, qs, this->verbose_);
  //    std::cout << p_U.transpose() << " and " << p_L.transpose() << std::endl;
      T U = (vmf_A.GetTau()*vmf_A.GetMu() +
          vmf_B.GetTau()*p_U).norm();
      T L = (vmf_A.GetTau()*vmf_A.GetMu() +
          vmf_B.GetTau()*p_L).norm();
      T LfU = 0.;
      T UfL = 0.;
      T fUfLoU2L2 = 0.;
      T L2fUU2fLoU2L2 = 0.;
      if (this->verbose_)
        std::cout << "-- U " << U << " L " << L << std::endl;
      if (fabs(U-L) < 1.e-6) {
        if (U > 50.) {
          fUfLoU2L2 = log(U-1.) + 2.*U - log(2.) - 3.*log(U) - U;
          LfU = log(U) + 2.*U - log(2.) - log(U) - U;
        } else {
          LfU = log(3+U+U*exp(2.*U)) - log(2.) - log(U) - U;
          fUfLoU2L2 = log(1. + U + (U-1.) * exp(2.*U)) - log(2.) - 3.*log(U) - U;
        }
        UfL = log(3.) + 2.*U - log(2.) - log(U) - U;
      } else {
        T f_U = ComputeLog2SinhOverZ(U);
        T f_L = ComputeLog2SinhOverZ(L);
        if (this->verbose_)
          std::cout << "f_U " << f_U << " f_L " << f_L << std::endl;
        fUfLoU2L2 = - log(U - L) - log(U + L);
//        if (f_U > f_L) {
        fUfLoU2L2 += log(1. - exp(f_L-f_U)) + f_U;
        if (this->verbose_)
          std::cout << "f_L - f_U " << f_L-f_U << " exp(.) " << exp(f_L-f_U)
            << std::endl;
//        } else {
//          fUfLoU2L2 += log(exp(f_U-f_L) - 1.) + f_L;
//          if (this->verbose_)
//            std::cout << "f_U - f_L " << f_U-f_L << " exp(.) " << exp(f_U-f_L)
//                << std::endl;
//        }
        L2fUU2fLoU2L2 = -log(U - L) -log(U + L);
        LfU = 2.*log(L)+f_U + L2fUU2fLoU2L2;
        UfL = 2.*log(U)+f_L + L2fUU2fLoU2L2;
        if (this->verbose_)
          std::cout << "LfU " << LfU << " UfL " << UfL << std::endl;
      }
      uint32_t K = vmf_mm_B_.size();
      Melem[j*K+k] = BuildM(vmf_A.GetMu(), vmf_B.GetMu());
//      std::cout << vmf_A.GetMu().transpose() << " and " << vmf_B.GetMu().transpose() << std::endl;
//      std::cout << j << " k " << k << std::endl << Melem[j*k+k] << std::endl;
      T D = log(2.*M_PI) + log(vmf_A.GetPi()) + log(vmf_B.GetPi())
        + vmf_A.GetLogZ() + vmf_B.GetLogZ();
      Aelem(j*K+k) = log(2.) + log(vmf_A.GetTau()) + log(vmf_B.GetTau())
        + D + fUfLoU2L2;
      Eigen::Matrix<T,4,1> b;
      b << 2.*log(vmf_A.GetTau()) + fUfLoU2L2,
        2.*log(vmf_B.GetTau())+fUfLoU2L2, LfU, UfL;
      Belem.row(j*K+k) = (b.array()+D).matrix();
      BelemSign.row(j*K+k) << 1.,1.,-1.,1.;
    }
  Eigen::Matrix<T,4,4> A;
  for (uint32_t j=0; j<4; ++j)
    for (uint32_t k=0; k<4; ++k) {
      Eigen::Matrix<T,Eigen::Dynamic,1> M_jk_elem(Melem.size());
      for (uint32_t i=0; i<Melem.size(); ++i)
        M_jk_elem(i) = Melem[i](j,k);
      A(j,k) = (M_jk_elem.array()*(Aelem.array() -
            Aelem.maxCoeff()).array().exp()).sum() *
        exp(Aelem.maxCoeff());
      if (this->verbose_)
        std::cout << j << " " << k << " " << M_jk_elem.transpose() 
          << " = " << A(j,k) << std::endl;
    }
  T B = (BelemSign.array()*(Belem.array() -
        Belem.maxCoeff()).exp()).sum() * exp(Belem.maxCoeff());
  if (this->verbose_) {
    std::cout << "Aelem " << Aelem.transpose() << std::endl;
    std::cout << "A " <<  std::endl;
    std::cout << A << std::endl;
    std::cout << "BelemSign " << BelemSign << std::endl;
    std::cout << "Belem " << Belem << std::endl;
    std::cout << "B " << B << std::endl;
    T Bb = 0.;
    for (uint32_t l=0; l<Belem.rows(); ++l) {
      T  dB =  (BelemSign.row(l).array()*(Belem.row(l).array() -
        Belem.row(l).maxCoeff()).exp()).sum() * exp(Belem.row(l).maxCoeff());
      std::cout << dB << " ";
      Bb += dB;
    }
    std::cout << " = " << Bb << std::endl;
  }
  T lambda_max = FindMaximumQAQ<T>(A, Q, this->verbose_);
  if (this->verbose_) {
    std::cout << "B " << B << " lambda_max " << lambda_max << std::endl;
  }
  return B + lambda_max;
}

template class UpperBoundConvexS3<float>;
template class UpperBoundConvexS3<double>;

template <typename T>
T FindMaximumQAQ(const Eigen::Matrix<T,4,4>& A, const
  Eigen::Matrix<T,4,Eigen::Dynamic> Q, bool verbose) {
  assert(Q.cols() >= 4);
  std::vector<T> lambdas;
  // Only one q: 
  for (uint32_t i=0; i<Q.cols(); ++i) {
    lambdas.push_back(Q.col(i).transpose() * A * Q.col(i));
    if(verbose) std::cout<<"lambda 1x1 " << lambdas[i] << std::endl;
  }
  // Full problem:
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> A_ = Q.transpose() * A * Q;
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> B_ = Q.transpose() * Q;

//  std::cout << "full A and B: " << std::endl
//    << A_ << std::endl << B_ << std::endl;
  ComputeLambdasOfSubset<T,2>(A_,B_,Q,verbose,lambdas);
  ComputeLambdasOfSubset<T,3>(A_,B_,Q,verbose,lambdas);
  ComputeLambdasOfSubset<T,4>(A_,B_,Q,verbose,lambdas);
  if (Q.cols() > 4)
    ComputeLambdasOfSubset<T,5>(A_,B_,Q,verbose,lambdas);
  if (Q.cols() > 5)
    ComputeLambdasOfSubset<T,6>(A_,B_,Q,verbose,lambdas);
  if (Q.cols() > 6)
    ComputeLambdasOfSubset<T,7>(A_,B_,Q,verbose,lambdas);
  if (Q.cols() > 7)
    ComputeLambdasOfSubset<T,8>(A_,B_,Q,verbose,lambdas);
  if (Q.cols() > 8)
    ComputeLambdasOfSubset<T,9>(A_,B_,Q,verbose,lambdas);
  if (Q.cols() > 9)
    std::cout << "ERROR: FindMaximumQAQ does not compute all lambdas;"
      << " you have to many rotations in the set." << std::endl;
//  std::cout << "lambda: " ;
//  for (auto l : lambdas)
//    std::cout << l << " ";
//  std::cout << std::endl;
  return *std::max_element(lambdas.begin(), lambdas.end());
}

template <typename T>
T FindMaximumQAQ(const Eigen::Matrix<T,4,4>& A, const Tetrahedron4D<T>&
    tetrahedron, bool verbose) {
  std::vector<T> lambdas;
  Eigen::Matrix<T,4,Eigen::Dynamic> Q(4,4);
  for (uint32_t i=0; i<4; ++i)
    Q.col(i) = tetrahedron.GetVertex(i);
  return FindMaximumQAQ<T>(A,Q,verbose);
}

}
