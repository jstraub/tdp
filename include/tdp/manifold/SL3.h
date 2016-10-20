#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <tdp/config.h>
#include <tdp/manifold/manifold.h>

namespace tdp {

/// Special Linear Group in 3D to represent homographies
template<typename T>
class SL3 : Manifold<T,8> {
 public:
  TDP_HOST_DEVICE
  SL3();
  TDP_HOST_DEVICE
  SL3(const Eigen::Matrix<T,3,3>& H);
  TDP_HOST_DEVICE
  SL3(const SL3<T>& other);

  TDP_HOST_DEVICE
  const Eigen::Matrix<T,3,3>& matrix() const { return H_;}
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,3>& matrix() { return H_;}

  static SL3<T> Random() { return Exp(Eigen::Matrix<T,8,1>::Random()); }

  static Eigen::Matrix<T,3,3> invVee(const Eigen::Matrix<T,8,1>& x);
  static Eigen::Matrix<T,3,3> Exp_(const Eigen::Matrix<T,8,1> x);
  static SL3<T> Exp(const Eigen::Matrix<T,8,1> x) { return SL3<T>(Exp_(x)); }

  /// Generator matrices of SL3
  static Eigen::Matrix<T,3,3> G1();
  static Eigen::Matrix<T,3,3> G2();
  static Eigen::Matrix<T,3,3> G3();
  static Eigen::Matrix<T,3,3> G4();
  static Eigen::Matrix<T,3,3> G5();
  static Eigen::Matrix<T,3,3> G6();
  static Eigen::Matrix<T,3,3> G7();
  static Eigen::Matrix<T,3,3> G8();
  static Eigen::Matrix<T,3,3> G(uint32_t i);

  SL3<T>& operator*=(const SL3<T>& other);
  const SL3<T> operator*(const SL3<T>& other) const;

 protected:
  Eigen::Matrix<T,3,3> H_;
};

typedef SL3<float> SL3f;
typedef SL3<double> SL3d;

}

#include <tdp/manifold/SL3_impl.hpp>

