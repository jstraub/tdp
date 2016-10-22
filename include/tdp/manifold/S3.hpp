#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/config.h>
#include <tdp/manifold/manifold.h>

namespace tdp {

template<typename T, int Options=Eigen::ColMajor>
class S3 : Manifold<T,3> {
 public:
  TDP_HOST_DEVICE
  S3();
  TDP_HOST_DEVICE
  S3(const Eigen::Matrix<T,3,3>& R);
  TDP_HOST_DEVICE
  S3(const S3<T,Options>& other);
  TDP_HOST_DEVICE
  S3(const Eigen::Quaternion<T>& q);
  TDP_HOST_DEVICE
  S3(const Eigen::Matrix<T,3,1>& axis, T angle);
  TDP_HOST_DEVICE
  S3(const Eigen::Matrix<T,3,1>& axisAngle);

  TDP_HOST_DEVICE
  ~S3() {}

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,3,Options> matrix() const { 
    return q_.matrix();
  }
  TDP_HOST_DEVICE
  Eigen::Matrix<T,4,1,Options> vector() const { 
    return q_.coeffs();
  }

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1,Options> ToAxisAngle() const {
    Eigen::Matrix<T,3,1,Options> aa;
    T angle;
    ToAxisAngle(aa, angle);
    return aa*angle;
  }
  TDP_HOST_DEVICE
  void ToAxisAngle(Eigen::Matrix<T,3,1,Options>& axis, T& angle) const {
    axis = q_.vec().normalized();
    angle = 2.*acos(q_.w());
  }

  TDP_HOST_DEVICE
  S3<T,Options> Inverse() const ;
  TDP_HOST_DEVICE
  S3<T,Options> Exp (const Eigen::Matrix<T,3,1,Options>& w) const ;
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1,Options> Log (const S3<T,Options>& other) const;

  TDP_HOST_DEVICE
  S3<T,Options>& operator*=(const S3<T,Options>& other);
  TDP_HOST_DEVICE
  const S3<T,Options> operator*(const S3<T,Options>& other) const;

  /// transform 3D data point
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1,Options> operator*(
      const Eigen::Matrix<T,3,1,Options>& x) const;

  static Eigen::Matrix<T,3,3> Exp_(const Eigen::Matrix<T,3,1>& w);
  static Eigen::Matrix<T,3,1> Log_(const Eigen::Matrix<T,3,3>& R);

  static S3<T,Options> Random();

  /// Elementary rotation matrices about the x,y,z axis by angle alpha
  /// [rad]
  static S3<T,Options> Rx(T alpha);
  static S3<T,Options> Ry(T alpha);
  static S3<T,Options> Rz(T alpha);

  /// Rotation matrix from roll pitch yaw (rpy) angles;
  static S3<T,Options> R_rpy(Eigen::Matrix<T,3,1> rpy) {
    return Rz(rpy(0))*Ry(rpy(1))*Rz(rpy(2)); 
  }

 private:
  Eigen::Quaternion<T,Options> q_;
};

typedef S3<double> S3d;
typedef S3<float> S3f;
typedef S3<double,Eigen::DontAligne> S3dda;
typedef S3<float,Eigen::DontAligne> S3fda;

template<typename T>
std::ostream& operator<<(std::ostream& out, const S3<T,Options>& s3) {
  out << s3.vector().transpose() << std::endl;
  return out;
}



}
#include <tdp/manifold/S3_impl.hpp>


