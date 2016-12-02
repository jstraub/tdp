#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/config.h>
#include <tdp/manifold/manifold.h>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/SO3mat.h>

namespace tdp {

template<typename T, class SO3_T, int Options=Eigen::ColMajor>
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
  SE3(const SO3_T& R);

  TDP_HOST_DEVICE
  SE3(const SO3_T& R, const Eigen::Matrix<T,3,1>& t);

  TDP_HOST_DEVICE
  SE3(const SE3<T,SO3_T,Options>& other);


  TDP_HOST_DEVICE
  ~SE3() {};

  TDP_HOST_DEVICE
  Eigen::Matrix<T,4,4> matrix() const;

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,4> matrix3x4() const;

  TDP_HOST_DEVICE
  const Eigen::Matrix<T,3,1,Options>& translation() const { return t_;};
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1,Options>& translation() { return t_;};

  TDP_HOST_DEVICE
  const SO3_T& rotation() const { return R_;};
  TDP_HOST_DEVICE
  SO3_T& rotation() { return R_;};

  TDP_HOST_DEVICE
  SE3<T,SO3_T,Options> Inverse() const ;

  TDP_HOST_DEVICE
  SE3<T,SO3_T,Options> Exp(const Eigen::Matrix<T,6,1>& w) const ;

  TDP_HOST_DEVICE
  Eigen::Matrix<T,6,1> Log(const SE3<T,SO3_T,Options>& other) const ;

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> vee() const;

//  Eigen::Matrix<T,6,1> operator-(const SE3<T,SO3_T,Options>& other);
//  SE3<T,SO3_T,Options>& operator+=(const SE3<T,SO3_T,Options>& other);
//  const SE3<T,SO3_T,Options> operator+(const SE3<T,SO3_T,Options>& other) const;

  SE3<T,SO3_T,Options>& operator*=(const SE3<T,SO3_T,Options>& other);
  const SE3<T,SO3_T,Options> operator*(const SE3<T,SO3_T,Options>& other) const;

  SE3<T,SO3_T,Options>& operator+=(const Eigen::Matrix<T,6,1>& w);
  const SE3<T,SO3_T,Options> operator+(const Eigen::Matrix<T,6,1>& w);

  // transform 3D data point
  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> operator*(const Eigen::Matrix<T,3,1>& x) const;

//  /// Generator matrices of SE3
//  static Eigen::Matrix<T,3,3> G1();
//  static Eigen::Matrix<T,3,3> G2();
//  static Eigen::Matrix<T,3,3> G3();
//  static Eigen::Matrix<T,3,3> G(uint32_t i);

  static SE3<T,SO3_T,Options> Exp_(const Eigen::Matrix<T,6,1>& w);
  static Eigen::Matrix<T,6,1> Log_(const SE3<T,SO3_T,Options>& _T);

  static SE3<T,SO3_T,Options> Random() { 
    return SE3<T,SO3_T,Options>(SO3_T::Random(),
      Eigen::Matrix<T,3,1,Options>::Random());
  }
  static SE3<T,SO3_T,Options> Random(T maxAngle_rad, 
      const Eigen::Matrix<T,3,1,Options>& mean_t, T std_t);

 private:
  SO3_T R_;
  Eigen::Matrix<T,3,1,Options> t_;
};


template<typename T, class SO3_T, int Options>
std::ostream& operator<<(std::ostream& out, const SE3<T,SO3_T,Options>& se3);

}

#include <tdp/manifold/SE3_impl.hpp>

namespace tdp {

typedef SE3<double,SO3mat<double>,Eigen::DontAlign> SE3d;
typedef SE3<float ,SO3mat<float >,Eigen::DontAlign> SE3f;

typedef SE3<double,SO3mat<double>,Eigen::DontAlign> SE3dda;
typedef SE3<float ,SO3mat<float >,Eigen::DontAlign> SE3fda;

//template <typename T> using SE3da = SE3<T,Eigen::DontAlign>;

}
