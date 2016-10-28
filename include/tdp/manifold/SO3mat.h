#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/config.h>
#include <tdp/manifold/manifold.h>
#include <tdp/manifold/rotation.h>

namespace tdp {

template<typename T>
class SO3mat : Manifold<T,3> {
 public:
  TDP_HOST_DEVICE
  SO3mat();

  TDP_HOST_DEVICE
  SO3mat(const Eigen::Matrix<T,3,3>& R);

  TDP_HOST_DEVICE
  SO3mat(const SO3mat<T>& other);

  TDP_HOST_DEVICE
  SO3mat(const Eigen::Quaternion<T>& q);

  TDP_HOST_DEVICE
  ~SO3mat() {}

  TDP_HOST_DEVICE
  const Eigen::Matrix<T,3,3>& matrix() const { return R_;}
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,3>& matrix() { return R_;}

  Eigen::Quaternion<T> quat() { return Eigen::Quaternion<T>(R_); }

  TDP_HOST_DEVICE
  SO3mat<T> Inverse() const ;
  TDP_HOST_DEVICE
  SO3mat<T> Exp (const Eigen::Matrix<T,3,1>& w)const ;
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> Log (const SO3mat<T>& other)const ;

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> vee() const;

  Eigen::Matrix<T,3,1> operator-(const SO3mat<T>& other);
  SO3mat<T>& operator+=(const SO3mat<T>& other);
  const SO3mat<T> operator+(const SO3mat<T>& other) const;

  SO3mat<T>& operator*=(const SO3mat<T>& other);
  const SO3mat<T> operator*(const SO3mat<T>& other) const;

  SO3mat<T>& operator+=(const Eigen::Matrix<T,3,1>& w);
  const SO3mat<T> operator+(const Eigen::Matrix<T,3,1>& w);

  // transform 3D data point
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> operator*(const Eigen::Matrix<T,3,1>& x) const;

  TDP_HOST_DEVICE
  static Eigen::Matrix<T,3,3> invVee(const Eigen::Matrix<T,3,1>& w);
  TDP_HOST_DEVICE
  static Eigen::Matrix<T,3,1> vee(const Eigen::Matrix<T,3,3>& W);
  TDP_HOST_DEVICE
  static Eigen::Matrix<T,3,3> skew(const Eigen::Matrix<T,3,3>& W);

  /// Generator matrices of SO3
  static Eigen::Matrix<T,3,3> G1();
  static Eigen::Matrix<T,3,3> G2();
  static Eigen::Matrix<T,3,3> G3();
  static Eigen::Matrix<T,3,3> G(uint32_t i);

  static Eigen::Matrix<T,3,3> Exp_(const Eigen::Matrix<T,3,1>& w);
  static Eigen::Matrix<T,3,1> Log_(const Eigen::Matrix<T,3,3>& R);

  static SO3mat<T> Random();

  /// Elementary rotation matrices about the x,y,z axis by angle alpha
  /// [rad]
  static SO3mat<T> Rx(T alpha);
  static SO3mat<T> Ry(T alpha);
  static SO3mat<T> Rz(T alpha);

  /// Rotation matrix from roll pitch yaw (rpy) angles;
  static SO3mat<T> R_rpy(Eigen::Matrix<T,3,1> rpy) {
    return Rz(rpy(0))+Ry(rpy(1))+Rz(rpy(2)); 
  }

 private:
  Eigen::Matrix<T,3,3> R_;
};

typedef SO3mat<double> SO3matd;
typedef SO3mat<float> SO3matf;

template<typename T>
std::ostream& operator<<(std::ostream& out, const SO3mat<T>& so3) {
  out << so3.matrix();
  return out;
}



}
#include <tdp/manifold/SO3mat_impl.hpp>

