#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/config.h>
#include <tdp/manifold/manifold.h>

namespace tdp {

template<typename T>
class SO3 : Manifold<T,3> {
 public:
  TDP_HOST_DEVICE
  SO3();

  TDP_HOST_DEVICE
  SO3(const Eigen::Matrix<T,3,3>& R);

  TDP_HOST_DEVICE
  SO3(const SO3<T>& other);

  TDP_HOST_DEVICE
  ~SO3() {};

  TDP_HOST_DEVICE
  const Eigen::Matrix<T,3,3>& matrix() const { return R_;};
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,3>& matrix() { return R_;};

  TDP_HOST_DEVICE
  SO3<T> Inverse() const ;
  TDP_HOST_DEVICE
  SO3<T> Exp (const Eigen::Matrix<T,3,1>& w)const ;
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> Log (const SO3<T>& other)const ;

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> vee() const;

  Eigen::Matrix<T,3,1> operator-(const SO3<T>& other);
  SO3<T>& operator+=(const SO3<T>& other);
  const SO3<T> operator+(const SO3<T>& other) const;

  SO3<T>& operator+=(const Eigen::Matrix<T,3,1>& w);
  const SO3<T> operator+(const Eigen::Matrix<T,3,1>& w);

  static Eigen::Matrix<T,3,3> invVee(const Eigen::Matrix<T,3,1>& w);
  static Eigen::Matrix<T,3,1> vee(const Eigen::Matrix<T,3,3>& W);
  static Eigen::Matrix<T,3,3> skew(const Eigen::Matrix<T,3,3>& W);

  /// Generator matrices of SO3
  static Eigen::Matrix<T,3,3> G1();
  static Eigen::Matrix<T,3,3> G2();
  static Eigen::Matrix<T,3,3> G3();
  static Eigen::Matrix<T,3,3> G(uint32_t i);

  static Eigen::Matrix<T,3,3> Exp_(const Eigen::Matrix<T,3,1>& w);
  static Eigen::Matrix<T,3,1> Log_(const Eigen::Matrix<T,3,3>& R);

  static SO3<T> Random();

 private:
  Eigen::Matrix<T,3,3> R_;
};

typedef SO3<double> SO3d;
typedef SO3<float> SO3f;

template<typename T>
std::ostream& operator<<(std::ostream& out, const SO3<T>& so3) {
  out << so3.matrix();
  return out;
}

}
#include <tdp/manifold/SO3_impl.hpp>

