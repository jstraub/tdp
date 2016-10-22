#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tdp/config.h>
#include <tdp/manifold/manifold.h>
#include <tdp/manifold/SO3mat.h>

namespace tdp {

template<typename T, int Options=Eigen::ColMajor>
class SO3 : Manifold<T,3> {
 public:
  TDP_HOST_DEVICE
  SO3();
  TDP_HOST_DEVICE
  SO3(const Eigen::Matrix<T,3,3>& R);
  TDP_HOST_DEVICE
  SO3(const SO3<T,Options>& other);
  TDP_HOST_DEVICE
  SO3(const Eigen::Quaternion<T,Options>& q);
  TDP_HOST_DEVICE
  SO3(const Eigen::Matrix<T,3,1>& axis, T angle);
  TDP_HOST_DEVICE
  SO3(const Eigen::Matrix<T,3,1>& axisAngle);

  TDP_HOST_DEVICE
  ~SO3() {}

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
  SO3<T,Options> Inverse() const ;
  TDP_HOST_DEVICE
  SO3<T,Options> Exp (const Eigen::Matrix<T,3,1>& w) const ;
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> Log (const SO3<T,Options>& other) const;

  TDP_HOST_DEVICE
  SO3<T,Options>& operator*=(const SO3<T,Options>& other);
  TDP_HOST_DEVICE
  const SO3<T,Options> operator*(const SO3<T,Options>& other) const;

  /// transform 3D data point
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> operator*(
      const Eigen::Matrix<T,3,1>& x) const;

  static SO3<T,Options> Exp_(const Eigen::Matrix<T,3,1>& w);
  static Eigen::Matrix<T,3,1> Log_(const SO3<T,Options>& R);

  static SO3<T,Options> Random();

  /// Elementary rotation matrices about the x,y,z axis by angle alpha
  /// [rad]
  static SO3<T,Options> Rx(T alpha);
  static SO3<T,Options> Ry(T alpha);
  static SO3<T,Options> Rz(T alpha);

  /// Rotation matrix from roll pitch yaw (rpy) angles;
  static SO3<T,Options> R_rpy(Eigen::Matrix<T,3,1> rpy) {
    return Rz(rpy(0))*Ry(rpy(1))*Rz(rpy(2)); 
  }

  static Eigen::Matrix<T,3,3> invVee(const Eigen::Matrix<T,3,1>& w) {
    return SO3mat<T>::invVee(w);
  }
  static Eigen::Matrix<T,3,1> vee(const Eigen::Matrix<T,3,3>& W) {
    return SO3mat<T>::vee(W);
  }
  static Eigen::Matrix<T,3,3> skew(const Eigen::Matrix<T,3,3>& W) {
    return SO3mat<T>::skew(W);
  }

 private:
  Eigen::Quaternion<T,Options> q_;
};


template<typename T, int Options>
std::ostream& operator<<(std::ostream& out, const SO3<T,Options>& s3) {
  out << s3.vector().transpose() << std::endl;
  return out;
}



}
#include <tdp/manifold/SO3_impl.hpp>

namespace tdp {

typedef SO3<double> SO3d;
typedef SO3<float> SO3f;
typedef SO3<double,Eigen::DontAlign> SO3dda;
typedef SO3<float,Eigen::DontAlign> SO3fda;
template <typename T> using SO3da = SO3<T,Eigen::DontAlign>;
}
