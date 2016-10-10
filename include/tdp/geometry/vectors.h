#pragma once
#include <Eigen/Dense>
#include <tdp/eigen/dense.h>

namespace tdp {

/// compute dot product at B 
template<typename DerivedA, typename DerivedB, typename DerivedC>
float DotABC(const Eigen::MatrixBase<DerivedA>& a,
    const Eigen::MatrixBase<DerivedB>& b,
    const Eigen::MatrixBase<DerivedC>& c) {
//  typedef typename Eigen::internal::plain_row_type<DerivedB>::type RowVectorType;
//  RowVectorType dirab = a-b;
//  RowVectorType dircb = c-b;
//  return (dirab.dot(dircb)/(dirab.norm()*dircb.norm()));
  return std::min(1.f,std::max(-1.f, 
        static_cast<float>(((a-b).dot(c-b)/((a-b).norm()*(c-b).norm())))));
}

//template<typename DerivedA, typename DerivedB>
//Eigen::MatrixBase<DerivedB> ProjectAontoB(const Eigen::MatrixBase<DerivedA>& a,
//    const Eigen::MatrixBase<DerivedB>& b) {
//  return (a.dot(b)/b.dot(b))*b;
//}
//
//template<typename DerivedA, typename DerivedB>
//Eigen::MatrixBase<DerivedB> RejectAfromB(const Eigen::MatrixBase<DerivedA>& a, 
//    const Eigen::MatrixBase<DerivedB>& b) {
//  return a - ProjectAontoB(a,b);
//}

template<typename DerivedA, typename DerivedB, typename DerivedC>
void ProjectAontoB(const Eigen::MatrixBase<DerivedA>& a,
    const Eigen::MatrixBase<DerivedB>& b,
    Eigen::MatrixBase<DerivedC>& c) {
  c = (a.dot(b)/b.dot(b))*b;
}

template<typename DerivedA, typename DerivedB, typename DerivedC>
void RejectAfromB(const Eigen::MatrixBase<DerivedA>& a, 
    const Eigen::MatrixBase<DerivedB>& b,
    Eigen::MatrixBase<DerivedC>& c) {
  ProjectAontoB(a,b,c);
  c = a - c;
}

template<typename DerivedA, typename DerivedB>
float LengthOfAonB(const Eigen::MatrixBase<DerivedA>& a,
    const Eigen::MatrixBase<DerivedB>& b) {
  return a.dot(b);
}

//template<typename DerivedA, typename DerivedB>
//float LengthOfAorthoToB(const Eigen::MatrixBase<DerivedA>& a,
//    const Eigen::MatrixBase<DerivedB>& b) {
//  Eigen::MatrixBase<DerivedA> c;
//  RejectAfromB(a,b,c);
//  return c.norm();
//}

}
