#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/manifold.h>
#include <tdp/manifold/SO3.h>

namespace tdp {

template<typename T>
class SE3 : Manifold<T,6> {
 friend class SO3<T>;
 public:
  SE3();
  SE3(const Eigen::Matrix<T,4,4>& Tmat);
  SE3(const SE3<T>& other);
  ~SE3() {};

  const Eigen::Matrix<T,4,4>& matrix() const { return T_;};
  Eigen::Matrix<T,4,4>& matrix() { return T_;};

  SE3<T> Inverse() const ;
  SE3<T> Exp (const Eigen::Matrix<T,6,1>& w)const ;
  Eigen::Matrix<T,6,1> Log (const SE3<T>& other)const ;

  Eigen::Matrix<T,3,1> vee() const;

  Eigen::Matrix<T,6,1> operator-(const SE3<T>& other);
  SE3<T>& operator+=(const SE3<T>& other);
  const SE3<T> operator+(const SE3<T>& other) const;

  SE3<T>& operator+=(const Eigen::Matrix<T,6,1>& w);
  const SE3<T> operator+(const Eigen::Matrix<T,6,1>& w);

//  /// Generator matrices of SE3
//  static Eigen::Matrix<T,3,3> G1();
//  static Eigen::Matrix<T,3,3> G2();
//  static Eigen::Matrix<T,3,3> G3();
//  static Eigen::Matrix<T,3,3> G(uint32_t i);

 private:
  Eigen::Matrix<T,4,4> T_;

  static Eigen::Matrix<T,4,4> Exp_(const Eigen::Matrix<T,6,1>& w);
  static Eigen::Matrix<T,6,1> Log_(const Eigen::Matrix<T,4,4>& Tmat);
  
};

typedef SE3<double> SE3d;
typedef SE3<float> SE3f;

template<typename T>
std::ostream& operator<<(std::ostream& out, const SE3<T>& se3);

}

#include <tdp/manifold/SE3_impl.hpp>
