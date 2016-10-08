/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <random>
#include <Eigen/Dense>
#include <tdp/manifold/SO3.h>

namespace tdp {

/// Class describing a data point on the sphere in D dimensions.
template<typename T, int D>
class S {
 public:
  S();
  S(const Eigen::Matrix<T,D,1>& x);
  S(const S<T,D>& other);
  ~S() {};

  /// Compute the Riemannian Exp map around this point p of x in TpS.
  /// Yields another point on the sphere.
  S<T,D> Exp(const Eigen::Ref<const Eigen::Matrix<T,D,1>>& x) const;

  /// Compute the Riemannian Log map around this point p of q and
  /// yields a data point x in TpS. Note that the returned vector is
  /// represented in the ambient Euclidean space. See Intrinsic().
  Eigen::Matrix<T,D,1> Log(const S<T,D>& q) const;

  /// Compute the intrisic representation of a vector in TpS which is
  /// D-1 dimensional.
  Eigen::Matrix<T,D-1,1> ToIntrinsic(
      const Eigen::Ref<const Eigen::Matrix<T,D,1>>& x) const;

  /// Compute the representation of a vector in TpS in the ambient
  /// space.
  Eigen::Matrix<T,D,1> ToAmbient(
      const Eigen::Ref<const Eigen::Matrix<T,D-1,1>>& xhat) const;

  /// Retraction that just orthogonally projects down to the sphere.
  /// A more efficient way of mapping TpS -> S than the Exp map.
  S<T,D> RetractOrtho(const Eigen::Ref<const Eigen::Matrix<T,D,1>>& x) const;

  /// Compute the dot product between two datapoints on the sphere.
  T dot(const S<T,D>& q) { return p_.dot(q.vector()); }
  T norm() { return p_.norm(); }

  /// Give access to the underlying vector.
  const Eigen::Matrix<T,D,1>& vector() const {return p_;}
  Eigen::Matrix<T,D,1>& vector() {return p_;}
  
  /// Compute the difference of two data points on S. The result will be
  /// in TotherS. This uses the Log map.
  Eigen::Matrix<T,D,1> operator-(const S<T,D>& other);

  /// Compute the rotation that rotates points in TpS to the north
  /// pole.
  Eigen::Matrix<T,D,D> north_R_TpS2() const;

  /// Compute the rotation that aligns a and b where a and b are on the
  /// sphere.
  static Eigen::Matrix<T,D,D> rotationFromAtoB(const
      Eigen::Matrix<T,D,1>& a, const Eigen::Matrix<T,D,1>& b, T
      percentage=1.0);

  /// Sample a random point on the sphere.
  static S<T,D> Random();

  constexpr static double MIN_DOT=-0.98;
  constexpr static double MAX_DOT=0.98;
 private:
  /// The data point.
  Eigen::Matrix<T,D,1> p_;
  
  /// Computes the inverse sinc(cos(dot)) in a stable way.
  static T invSincDot(T dot);
};

typedef S<double,2> S2d;
typedef S<double,3> S3d;
typedef S<double,4> S4d;

typedef S<float,2> S2f;
typedef S<float,3> S3f;
typedef S<float,4> S4f;

template<typename T, int D>
std::ostream& operator<<(std::ostream& out, const S<T,D>& q);

}

#include <tdp/manifold/S_impl.hpp>
