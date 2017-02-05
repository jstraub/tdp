/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <random>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#include "vmf.hpp"
#include "timer.hpp"

template<typename T>
class vMFprior {
 public:
  vMFprior(const Eigen::Matrix<T,3,1>& m0, T a, T b)
    : m0_(m0), b_(b), a_(a), unif_(0.,1.)
  {}

  T logMarginal(const Eigen::Matrix<T,3,1>& x) const {
    const T bTilde = (x + b_*m0_).norm();
    // TODO make sure M_PI*0.5 is really what it is and not 2/M_PI
    const T bOverTan = b_ < 1e-9 ? M_PI*0.5 : b_/tan(M_PI*0.5*b_);  
    const T sinc = bTilde < 1e-9 ? 1. : sin(bTilde*M_PI)/(bTilde*M_PI);
    const T sinus = sin(bTilde*0.5*M_PI);
    return log(bOverTan*0.125*(1.-sinc)/(sinus*sinus));
  }

  vMF<T,3> sample(std::mt19937& rnd) {
//    std::cout << "sample from vMF prior " << a_ << " " << b_ 
//      << " " << m0_.transpose() << std::endl;
    Eigen::Matrix<T,3,1> mu;
    T tau = 1.;
    vMF<T,3> vmf(m0_, tau*b_);
//    std::cout << "sampling from base" << std::endl;
    for (size_t it=0; it<30; ++it) {
      vmf.tau_ = tau*b_;
      mu = vmf.sample(rnd);
//      std::cout << "mu " << mu.transpose() << std::endl;
      const T dot = mu.dot(m0_); 
      tau = sampleConcentration(dot, rnd, 10, tau);
//      std::cout <<"@" << it << "tau " << tau << " mu " << mu.transpose() << std::endl;
    }
    return vMF<T,3>(mu, tau);
  }

  vMFprior<T> posterior(const Eigen::Matrix<T,3,1>& xSum, const T count) const {
    T aN = a_+count;
    Eigen::Matrix<T,3,1> muN = xSum + b_*m0_;
    T bN = muN.norm();
    muN /= bN;
    return vMFprior<T>(muN, aN, bN);
  }

  Eigen::Matrix<T,3,1> m0_;
  T b_;
  T a_;
 private:


  T concentrationLogPdf(const T tau, const T dot) const
  {
    if (a_ == 1.) {
      // for a_ == 1 we have closed form pdf that is propperly
      // normalized
      return logxOverSinhX(tau) + b_*tau*dot + log(xOverTanPiHalfX(b_))
        - LOG_2 - 2.*LOG_PI;
    } else {
      return -1;
    }
  };

  T propToConcentrationLogPdf(const T tau, const T dot) const
  {
    if (a_ == 1.) {
      // for a_ == 1 we have closed form pdf that is propperly
      // normalized
      return logxOverSinhX(tau) + b_*tau*dot + log(xOverTanPiHalfX(b_));
    } else {
      // this is only for 3D case
      return a_*(0.5*LOG_PI-0.5*LOG_2 + logxOverSinhX(tau)) + tau*b_*dot; 
    }
  };


  T sampleConcentration(const T dot, std::mt19937& rnd, size_t maxIt, T tau0 = 0.3)
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
