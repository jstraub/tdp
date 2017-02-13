/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <random>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#define LOG_2 0.69314718055994529
#define LOG_PI 1.1447298858494002
#define LOG_2PI 1.8378770664093453
#define LOG_4PI 2.5310242469692907

template<typename T> 
inline T logBesselI(T nu, T x)
{
  //TODO link against boost
  // for large values of x besselI \approx exp(x)/sqrt(2 PI x)
//  if(x>100.)  return x - 0.5*log(2.*M_PI*x);
//  return log(std::cyl_bessel_i(nu,x));
  return x - 0.5*LOG_2PI -0.5*log(x);
};

template<typename T> 
inline T logxOverSinhX(T x) {
  if (fabs(x) < 1e-9) 
    return 0.;
  else
    return log(x)-log(sinh(x));
}
template<typename T> 
inline T xOverSinhX(T x) {
  if (fabs(x) < 1e-9) 
    return 1.;
  else
    return x/sinh(x);
}
template<typename T> 
inline T xOverTanPiHalfX(T x) {
  if (fabs(x) < 1e-9) 
    return 2./M_PI;
  else
    return x/tan(x*M_PI*0.5);
}

template<typename T> 
inline T logSumExp(const Eigen::Matrix<T,Eigen::Dynamic,1>& logX) {
  T logMax = logX.maxCoeff();
  return log((logX.array()-logMax).exp().sum()) + logMax;
}

template <typename T, uint32_t D>
inline T MLEstimateTau(const Eigen::Matrix<T,3,1>& xSum, const
    Eigen::Matrix<T,3,1>& mu, T count) {
  // Need double precision to achive convergence; single is not enough.
  double tau = 1.0;
  double prevTau = 0.;
  double eps = 1e-8;
  double R = xSum.norm()/count;
  while (fabs(tau - prevTau) > eps) {
//    std::cout << "tau " << tau << " R " << R << std::endl;
    double inv_tanh_tau = 1./tanh(tau);
    double inv_tau = 1./tau;
    double f = -inv_tau + inv_tanh_tau - R;
    double df = inv_tau*inv_tau - inv_tanh_tau*inv_tanh_tau + 1.;
    prevTau = tau;
    tau -= f/df;
  }
  return tau;
};

template<typename T, int D>
class vMF 
{
public:
  vMF()
    : mu_(0,0,1), tau_(0.), unif_(0.,1.), gauss_(0.,1.)
  {}
  vMF(const Eigen::Matrix<T,D,1>& mu, T tau)
    : mu_(mu), tau_(tau), unif_(0.,1.), gauss_(0.,1.)
  {}
  vMF(const Eigen::Matrix<T,D,1>& tauMu)
    : mu_(tauMu.normalized()), tau_(tauMu.norm()), unif_(0.,1.), gauss_(0.,1.)
  {}
  vMF(const vMF<T,D>& vmf)
    : mu_(vmf.mu_), tau_(vmf.tau_), unif_(0.,1.), gauss_(0.,1.)
  {}

  T logPdf(const Eigen::Matrix<T,D,1>& x) const {
    const T d = static_cast<T>(D);
    if (tau_ < 1e-9) {
      // TODO insert general formula here
      return -LOG_4PI;
    } else {
      return (d/2. -1.)*log(tau_) - (d/2.)*LOG_2PI 
        - logBesselI<T>(d/2. -1.,tau_) + tau_*mu_.dot(x);
    }
  }

  /// Use uniform distribution on the sphere as a proposal distribution
  Eigen::Matrix<T,D,1> sample(std::mt19937& rnd) {
//    std::cout << "stating n sampling ------------ " << std::endl;
    // implemented using rejection sampling and proposals from a gaussian
    Eigen::Matrix<T,D,1> x;
    T pdf_g = -LOG_4PI;
    // bound via maximum over vMF at mu
    // TODO: dont know why I need to multiply by 2 here to get the
    // correct samples (as seen in concentration estimates)
    T M = 2.*tau_; 
    while(42) {
      // sample from zero mean Gaussian 
      for (uint32_t d=0; d<D; d++) x[d] = gauss_(rnd);
      x.normalize();
      // rejection sampling (in log domain)
      T u = log(unif_(rnd));
      T pdf_f = 2.*tau_*x.dot(mu_); //this->logPdf(x);
//      std::cout << pdf_f << " " << pdf_g << " " << M << " " << tau_ << std::endl;
      if(u < pdf_f-(M+pdf_g)) break;
    };
    return x;
  }


  Eigen::Matrix<T,D,1> mu_;
  T tau_;
private:
  std::uniform_real_distribution<T> unif_;
  std::normal_distribution<T> gauss_;
};


template<>
float vMF<float,3>::logPdf(const Eigen::Matrix<float,3,1>& x) const {
  if (tau_ < 1e-9) {
    return -LOG_4PI;
  } else {
    return -LOG_2PI + log(tau_) + tau_*(mu_.dot(x)-1.) - log(1.-exp(-2.*tau_));
//    return 0.5*LOG_PI - 0.5*LOG_2 + tau_*mu_.dot(x) + logxOverSinhX(tau_);
  }
}

template<>
Eigen::Matrix<float,3,1> vMF<float,3>::sample(std::mt19937& rnd) {
  //    std::cout << "stating n sampling ------------ " << std::endl;
  // https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
  // sample around (0,0,1)
  Eigen::Vector2f v(gauss_(rnd), gauss_(rnd));
  v.normalize();
  const float u = unif_(rnd);
  const float w = 1. + log(u+(1.-u)*exp(-2.*tau_))/tau_;
  const float a = sqrtf(1.-w*w);
  Eigen::Vector3f x(a*v(0), a*v(1), w);

  // rotate to mu
  Eigen::Vector3f axis = Eigen::Vector3f(0,0,1).cross(mu_);
  float angle = acos(mu_[2]);

  if (fabs(angle) <1e-9) 
    return x;

  Eigen::Quaternion<float> q(cos(angle*0.5), 
      sin(angle*0.5)*axis(0)/axis.norm(),
      sin(angle*0.5)*axis(1)/axis.norm(),
      sin(angle*0.5)*axis(2)/axis.norm());
  return q._transformVector(x);
}

