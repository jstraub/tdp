/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <random>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#include "vmf.hpp"

template<typename T>
class vMFprior {
 public:
  vMFprior(const Eigen::Matrix<T,3,1>& m0, T a, T b)
    : m0_(m0), b_(b), a_(a), unif_(0.,1.)
  {}

  T logMarginal(const Eigen::Matrix<T,3,1>& x) const {
    const T bTilde = (x + b_*m0_).norm();
    const T bOverTan = b_ < 1e-9 ? 2./M_PI : b_/tan(M_PI*0.5*b_);  
    const T sinc = bTilde < 1e-9 ? 1. : sin(bTilde*M_PI)/(bTilde*M_PI);
    const T sinus = sin(bTilde*0.5*M_PI);
    return log(bOverTan*0.125) + log(1.-sinc) - 2*log(sinus);
  }

  vMF<T,3> sample(std::mt19937& rnd) {
//    std::cout << "sample from vMF prior " << a_ << " " << b_ 
//      << " " << m0_.transpose() << std::endl;
    Eigen::Matrix<T,3,1> mu;
    T tau = 1.;
    vMF<T,3> vmf(m0_, tau*b_);
//    std::cout << "sampling from base" << std::endl;
    for (size_t it=0; it<10; ++it) {
      vmf.tau_ = tau*b_;
      mu = vmf.sample(rnd);
//      std::cout << "mu " << mu.transpose() << std::endl;
      const T dot = mu.dot(m0_); 
      tau = sampleConcentration(dot, rnd, 3, tau);
//      std::cout <<"@" << it << "tau " << tau << " mu " << mu.transpose() << std::endl;
    }
    return vMF<T,3>(mu, tau);
  }

  vMF<T,3> MAP() {
    Eigen::Matrix<T,3,1> xSum=b_*m0_;
    float tau = MLEstimateTau<float,3>(xSum, m0_, a_);
    return vMF<T,3>(m0_, tau);
  }

  vMFprior<T> posterior(const Eigen::Matrix<T,3,1>& xSum, const T count) const {
    T aN = a_+count;
    Eigen::Matrix<T,3,1> muN = xSum + b_*m0_;
    T bN = muN.norm();
    muN /= bN;
    return vMFprior<T>(muN, aN, bN);
  }

  vMFprior<T> posterior(const Eigen::Matrix<T,4,1>& ss) const {
    T aN = a_+ss(3);
    Eigen::Matrix<T,3,1> muN = ss.topRows(3) + b_*m0_;
    T bN = muN.norm();
    muN /= bN;
    return vMFprior<T>(muN, aN, bN);
  }

  Eigen::Matrix<T,3,1> m0_;
  T b_;
  T a_;
 private:

  T propToConcentrationLogPdf(const T tau, const T dot) const
  {
    if (tau < 1e-16) {
      return 0.; 
    } else {
      return a_*(log(tau) + LOG_2 - log(1.-exp(-2.*tau))) + tau*(b_*dot-a_); 
    }
  };

  T propToConcentrationLogPdfDeriv(const T tau, const T dot) const
  {
    // this is only for 3D case
    if (tau < 1e-16) {
      return b_*dot; 
    } else {
      return a_/tau - (2.*a_*exp(-2.*tau)/(1.-exp(-2.*tau))) + b_*dot -a_;
    }
  };
  T propToConcentrationLogPdfDerivDeriv(const T tau, const T dot) const
  {
    // this is only for 3D case
    if (tau < 1e-16) {
      return -a_/3.; 
    } else {
//      return -a_/(tau*tau) + (4.*a_*exp(2.*tau)/(1.-2.*exp(2.*tau)+exp(4.*tau)));
      return -a_/(tau*tau) + (4.*a_*exp(-2.*tau)/(1.-2.*exp(-2.*tau)+exp(-4.*tau)));
    }
  };

  T maximum(const T dot) {
    if (dot*b_ <= 0)
      return 0.;
    T tau = 1.;
    for (size_t it=0; it<100; ++it) {
      T f = propToConcentrationLogPdfDeriv(tau, dot);
      T df = propToConcentrationLogPdfDerivDeriv(tau, dot);
      tau -= f/df;
      if (fabs(f/df) < 1e-6)
        break;
    }
    return tau;
  };

  T intersect(const T c, const T dot, const T tau0) {
    T tau = tau0;
    for (size_t it=0; it<100; ++it) {
      T f = propToConcentrationLogPdf(tau, dot) - c;
      T df = propToConcentrationLogPdfDeriv(tau, dot);
      tau = std::max(0.f, tau-f/df);
//      std::cout << "   __ " <<it << ": " << f << " " << df 
//        << "\t f/df " << fabs(f/df) 
//        << "\t step to " << tau-f/df << ": "<< tau << std::endl;
      if (fabs(f/df) < 1e-6 || tau == 0.)
        break;
    }
    return tau;
  };

  /// slice sampler for concentration paramter tau
  T sampleConcentration(const T dot, std::mt19937& rnd, size_t maxIt, T tau0 = 0.3)
  {
    T tauMax = maximum(dot);
    T tau = tau0;
//    std::cout << " ----- max " << tauMax << " " << tau0 
//      << " dot " << dot << std::endl;
    T tauL = 0.;
    T tauR = tauMax;
    for(size_t t=0; t<maxIt; ++t)
    {
      const T f = propToConcentrationLogPdf(tau,dot);
      const T u = log(unif_(rnd)) + f; 
      
      if (tauMax > 0.) {
        tauL = intersect(u, dot, tauMax*0.001);
        tauR = intersect(u, dot, tauMax*1.5);
      } else {
        tauL = 0.;
        tauR = intersect(u, dot, 0.5);
      }
      tau = unif_(rnd)*(tauR-tauL)+tauL;
//      std::cout << tauL << " - " << tauMax << " - " << tauR 
//        << " tau= " << tau
//        << " : u " << u << " f(tau) " << f 
//        << " f(tau^star) " << propToConcentrationLogPdf(tauMax,dot)
//        << std::endl;
    }
    return tau;
  };

  /// Old backstepping slice sampler implementation
  T sampleConcentrationStepping(const T dot, std::mt19937& rnd, size_t maxIt, T tau0 = 0.3)
  {
//    std::cout << "start sampling concentration ---" << std::endl;
    // slice sampler for concentration paramter tau
    const T w = 0.1;  // width for expansions of search region
    T tau = tau0;      // arbitrary starting point
    for(size_t t=0; t<maxIt; ++t)
    {
      const T yMax = propToConcentrationLogPdf(tau,dot);
      const T y = log(unif_(rnd)) + yMax; 
      T tauMin = tau-w; 
      T tauMax = tau+w; 
//      std::cout << "before " << tauMin << " " << tauMax 
//        << ": " << propToConcentrationLogPdf(tauMin,dot)
//        << " " << propToConcentrationLogPdf(tauMax,dot) << std::endl;
      while (tauMin >=0. && propToConcentrationLogPdf(tauMin,dot) >= y) tauMin -= w;
      tauMin = std::max(static_cast<T>(0.),tauMin); 
      while (propToConcentrationLogPdf(tauMax,dot) >= y) tauMax += w;
//      std::cout << "after "  << tauMin << " " << tauMax 
//        << ": " << propToConcentrationLogPdf(tauMin,dot)
//        << " " << propToConcentrationLogPdf(tauMax,dot) << std::endl;
      while(42) {
        T tauNew = unif_(rnd)*(tauMax-tauMin)+tauMin;

//        std::cout << "@"<< t << ": " << tauMin << " " << tauMax << " " << tauNew << " " << tau 
//          << ": " << propToConcentrationLogPdf(tauNew,dot)
//          << " >=? " << y
//          << ",  " << propToConcentrationLogPdf(tauMin,dot)
//          << " "  << propToConcentrationLogPdf(tauMax,dot) << std::endl;

        if(propToConcentrationLogPdf(tauNew,dot) >= y)
        {
          tau = tauNew; break;
        }else{
          if (tauNew < tau) tauMin = tauNew; else tauMax = tauNew;
        }
      };
    }
    return tau;
  };
  std::uniform_real_distribution<T> unif_;
};
