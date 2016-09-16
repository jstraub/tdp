#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/config.h>
#include <tdp/manifold/manifold.h>
#include <tdp/manifold/SO3.h>

namespace tdp {

template<typename T>
class SE3 : Manifold<T,6> {
 //friend class tdp::SO3<T>;
 public:
  TDP_HOST_DEVICE
  SE3();

  TDP_HOST_DEVICE
  SE3(const Eigen::Matrix<T,4,4>& Tmat);

  TDP_HOST_DEVICE
  SE3(const Eigen::Matrix<T,3,3>& Rmat, const Eigen::Matrix<T,3,1>& tmat);

  TDP_HOST_DEVICE
  SE3(const SE3<T>& other);

  TDP_HOST_DEVICE
  ~SE3() {};

  TDP_HOST_DEVICE
  const Eigen::Matrix<T,4,4>& matrix() const { return T_;};

  TDP_HOST_DEVICE
  Eigen::Matrix<T,4,4>& matrix() { return T_;};

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,4> matrix3x4() const { return T_.topRows(3);};

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> translation() const { return T_.topRightCorner(3,1);};

  TDP_HOST_DEVICE
  SO3<T> rotation() const { return SO3<T>(T_.topLeftCorner(3,3));};

  TDP_HOST_DEVICE
  SE3<T> Inverse() const ;

  TDP_HOST_DEVICE
  SE3<T> Exp(const Eigen::Matrix<T,6,1>& w) const ;

  TDP_HOST_DEVICE
  Eigen::Matrix<T,6,1> Log(const SE3<T>& other) const ;

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> vee() const;

  Eigen::Matrix<T,6,1> operator-(const SE3<T>& other);
  SE3<T>& operator+=(const SE3<T>& other);
  const SE3<T> operator+(const SE3<T>& other) const;

  SE3<T>& operator*=(const SE3<T>& other);
  const SE3<T> operator*(const SE3<T>& other) const;

  SE3<T>& operator+=(const Eigen::Matrix<T,6,1>& w);
  const SE3<T> operator+(const Eigen::Matrix<T,6,1>& w);

  // transform 3D data point
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> operator*(const Eigen::Matrix<T,3,1>& x) const;

//  /// Generator matrices of SE3
//  static Eigen::Matrix<T,3,3> G1();
//  static Eigen::Matrix<T,3,3> G2();
//  static Eigen::Matrix<T,3,3> G3();
//  static Eigen::Matrix<T,3,3> G(uint32_t i);

  static Eigen::Matrix<T,4,4> Exp_(const Eigen::Matrix<T,6,1>& w);
  static Eigen::Matrix<T,6,1> Log_(const Eigen::Matrix<T,4,4>& Tmat);

 private:
  Eigen::Matrix<T,4,4> T_;
};

typedef SE3<double> SE3d;
typedef SE3<float> SE3f;

template<typename T>
std::ostream& operator<<(std::ostream& out, const SE3<T>& se3);

}

#include <tdp/manifold/SE3_impl.hpp>
