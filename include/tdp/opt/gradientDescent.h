#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/manifold.h>

namespace tdp {

template<typename T, uint32_t D, class M>
class GD {
 public:
  GD(bool verbose=false);
  virtual ~GD() {};

  virtual void Compute(const M& theta0, T thr, uint32_t itMax);

  virtual void ComputeJacobian(const M& theta, Eigen::Matrix<T,D,1>* J, T* f) = 0;

  const M& GetMinimum() {return theta_;}
 protected:
  T c_;
  T t_;
  M theta_;
  bool verbose_ = false;
  void LineSearch(Eigen::Matrix<T,D,1>* J, T* f);
};

template<typename T, uint32_t D, class M>
GD<T,D,M>::GD(bool verbose) : 
  c_(0.1), t_(0.3), verbose_(verbose)
{}

template<typename T, uint32_t D, class M>
void GD<T,D,M>::LineSearch(Eigen::Matrix<T,D,1>* J, T* f) {
  T delta = 1.;
  M thetaNew = theta_;
  ComputeJacobian(thetaNew, J, f);
  T fNew = *f;
  Eigen::Matrix<T,D,1> d = -(*J)/J->norm();
  if (verbose_)
    std::cout << "\tJ=" << J->transpose() << std::endl
      << "\td=" << d.transpose() << std::endl;
  T m = J->dot(d);
  while (*f-fNew < -c_*m*delta && delta > 1e-16) {
    delta *= t_;
    thetaNew = theta_ + delta*d;
    //std::cout << thetaNew << std::endl;
    ComputeJacobian(thetaNew, NULL, &fNew);
    if (verbose_)
      std::cout << *f-fNew << " <? " << -c_*m*delta 
        << "\tfNew=" << fNew << "\tdelta=" << delta << std::endl;
  }
  *J = delta*d;
  *f = fNew;
}

template<typename T, uint32_t D, class M>
void GD<T,D,M>::Compute(const M& theta0, T thr, uint32_t itMax) {
  theta_ = theta0;
  M thetaPrev = theta0;
  Eigen::Matrix<T,D,1> J = Eigen::Matrix<T,D,1>::Zero(); 
  T fPrev = 1e12;
  T f = 1e10;
//  T delta = 1e-2;
  uint32_t it=0;
  while((fPrev-f)/fabs(f) > thr && it < itMax) {
    fPrev = f;
    LineSearch(&J, &f);
//    ComputeJacobian(theta_, &J, &f);
    thetaPrev = theta_;
    theta_ += J;
    if (verbose_)
      std::cout << "@" << it << " f=" << f 
        << " df/f=" << (fPrev-f)/fabs(f) << std::endl;
    ++it;
  }
  if (f > fPrev) {
    theta_ = thetaPrev;
    f = fPrev;
  }
  std::cout << "@" << it << " f=" << f 
    << " df/f=" << (fPrev-f)/fabs(f) << std::endl;
}

}
