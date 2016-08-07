#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/manifold.h>

namespace tdp {

template<typename T, uint32_t D, class M>
class Newton {
 public:
  Newton();
  virtual ~Newton() {};

  virtual void Compute(const M& theta0, T thr, uint32_t itMax);

  virtual void ComputeJacobianAndHessian(const M& theta,
      Eigen::Matrix<T,D,D>* H, Eigen::Matrix<T,D,1>* J, T* f) = 0;

  const M& GetMinimum() {return theta_;}
 protected:
  T c_;
  T t_;
  M theta_;
  void LineSearch(Eigen::Matrix<T,D,1>* dT, T* f);
};

template<typename T, uint32_t D, class M>
Newton<T,D,M>::Newton() : 
  c_(0.1), t_(0.5)
{}

template<typename T, uint32_t D, class M>
void Newton<T,D,M>::LineSearch(Eigen::Matrix<T,D,1>* dT, T* f) {
  T delta = 1./t_;
  M thetaNew = theta_;
  Eigen::Matrix<T,D,1> J;
  Eigen::Matrix<T,D,D> H;
  ComputeJacobianAndHessian(thetaNew, &H, &J, f);
  T fNew = fabs(*f)*1e10; // inflate so that we enter the while loop at least once
  Eigen::Matrix<T,D,1> d = -H.ldlt().solve(J);
  d /= d.norm();
//  std::cout << "\tJ=" << J.transpose() << std::endl
//    << "\td=" << d.transpose() << std::endl;
//    << "\tH=" << std::endl << H << std::endl;
  T df = 0.; // no need to init this yet.
  uint32_t it = 0;
  while (*f-fNew < -df && delta > 1e-16) {
    delta *= t_;
    thetaNew = theta_+delta*d;
    ComputeJacobianAndHessian(thetaNew, NULL, NULL, &fNew);
    df = c_*delta*J.dot(d) + c_*delta*c_*delta*0.5*d.dot(H.ldlt().solve(d));
//    std::cout << *f-fNew << " <? " << -df 
//      << "\tfNew=" << fNew << "\tdelta=" << delta << std::endl;
    ++it;
  }
  std::cout << "\tlinesearch it=" << it << " df=" << *f-fNew 
    << " delta=" << delta << std::endl;
  *dT = delta*d;
  *f = fNew;
}

template<typename T, uint32_t D, class M>
void Newton<T,D,M>::Compute(const M& theta0, T thr, uint32_t itMax) {
  theta_ = theta0;
  M thetaPrev = theta0;
  Eigen::Matrix<T,D,1> dT = Eigen::Matrix<T,D,1>::Zero(); 
//  Eigen::Matrix<T,D,1> J = Eigen::Matrix<T,D,1>::Zero(); 
//  Eigen::Matrix<T,D,D> H = Eigen::Matrix<T,D,D>::Zero(); 
  T fPrev = 1e12;
  T f = 1e10;
//  T delta = 1e-2;
  uint32_t it=0;
  while((fPrev-f)/fabs(f) > thr && it < itMax) {
    fPrev = f;
    LineSearch(&dT, &f);
//    ComputeJacobianAndHessian(theta_, &H, &J, &f);
    thetaPrev = theta_;
//    theta_ += -0.005*H.ldlt().solve(J);
    theta_ += dT;
    std::cout << "@" << it << " f=" << f 
      << " df/f=" << (fPrev-f)/fabs(f) << std::endl;
    ++it;
  }
  if (f > fPrev) {
    theta_ = thetaPrev;
  }
}

}
