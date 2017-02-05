/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <random>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

// g++ -Wall -std=c++1z -I /usr/include/eigen3/ main.cpp -o test 

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

template<typename T>
inline size_t sampleDisc(const Eigen::Matrix<T,Eigen::Dynamic,1>& pdf,
    std::mt19937& rnd) {
  std::uniform_real_distribution<T> unif(0.,1.);
  T u = unif(rnd);
  T cdf = pdf[0];
  size_t k;
  for (k=1; k< (size_t)pdf.rows(); ++k) {
    if (u <= cdf) {
      return k-1;
    }
    cdf += pdf[k];
  }
  return pdf.rows()-1;
}

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
    // implemented using rejection sampling and proposals from a gaussian
    Eigen::Matrix<T,D,1> x;
    T pdf_g = -LOG_4PI;
    // bound via maximum over vMF at mu
    T M = tau_; 
    while(42) {
      // sample from zero mean Gaussian 
      for (uint32_t d=0; d<D; d++) x[d] = gauss_(rnd);
      x.normalize();
      // rejection sampling (in log domain)
      T u = log(unif_(rnd));
      T pdf_f = tau_*x.dot(mu_); //this->logPdf(x);
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
    return 0.5*LOG_PI - 0.5*LOG_2 + tau_*mu_.dot(x) + logxOverSinhX(tau_);
  }
}

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
      tau = sampleConcentration(dot, rnd, 10);
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


  T sampleConcentration(const T dot, std::mt19937& rnd, size_t maxIt)
  {
    // slice sampler for concentration paramter tau
    const T w = 0.1;  // width for expansions of search region
    T tau = 0.3;      // arbitrary starting point
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

//        if (tau > 30)
//        std::cout << tauMin << " " << tauMax << " " << tauNew << " " << tau 
//          << ": " << propToConcentrationLogPdf(tauNew,dot)
//          << ": " << propToConcentrationLogPdf(tauMin,dot)
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


int main() {

  std::mt19937 rnd(1);

//  Eigen::VectorXf pdf = Eigen::VectorXf::Zero(10);
//  pdf[9] = 1.;
//  for (size_t it = 0; it<100; ++it) {
//    std::cout << sampleDisc(pdf,rnd) << " ";
//  }
//  std::cout << std::endl;

  vMF<float,3> vmfA(Eigen::Vector3f(0,0,1), 100);
  vMF<float,3> vmfB(Eigen::Vector3f(0,1,0), 100);
  vMF<float,3> vmfC(Eigen::Vector3f(1,0,0), 100);
  vMF<float,3> vmfD(Eigen::Vector3f(-1,0,0),100);

  std::vector<Eigen::Vector3f> x;
  for (size_t i=0; i<1000; ++i) {
    x.push_back(vmfA.sample(rnd));
    x.push_back(vmfB.sample(rnd));
    x.push_back(vmfC.sample(rnd));
    x.push_back(vmfD.sample(rnd));
  }
  std::cout << "have " << x.size() << " input data" << std::endl;

  std::vector<Eigen::Vector3f> xSum(1, Eigen::Vector3f::Zero());
  for (size_t i=0; i<x.size(); ++i) xSum[0] += x[i];
  std::vector<float> counts(1, x.size());
  std::vector<uint32_t> z(x.size(),0);
  std::vector<vMF<float,3>> vmfs;
  vMFprior<float> base(Eigen::Vector3f(0,0,1), 1., 0.);
  float logAlpha = log(10.);

  vmfs.push_back(base.sample(rnd));
  for (size_t it=0; it<10000; ++it) {
    // sample labels | parameters
    size_t K = vmfs.size();
    for (size_t i=0; i<x.size(); ++i) {
      Eigen::VectorXf logPdfs(K+1);
      Eigen::VectorXf pdfs(K+1);
      for (size_t k=0; k<K; ++k) {
        if (z[i] == k) {
          // TODO what if last in cluster
          logPdfs[k] = log(counts[k]-1)+vmfs[k].logPdf(x[i]);
        } else {
          logPdfs[k] = log(counts[k])+vmfs[k].logPdf(x[i]);
        }
      }
      logPdfs[K] = logAlpha + base.logMarginal(x[i]);
      logPdfs = logPdfs.array() - logSumExp<float>(logPdfs);
      pdfs = logPdfs.array().exp();
      size_t zPrev = z[i];
      z[i] = sampleDisc(pdfs, rnd);
//      std::cout << z[i] << " " << K << ": " << pdfs.transpose() << std::endl;
      if (z[i] == K) {
        vmfs.push_back(base.posterior(x[i],1).sample(rnd));
        counts.push_back(0);
        xSum.push_back(Eigen::Vector3f::Zero());
        K++;
      }
      counts[zPrev] --;
      counts[z[i]] ++;
      xSum[zPrev] -= x[i];
      xSum[z[i]] += x[i];
    }
//    std::cout << "sample parameters" << std::endl;
    // sample parameters | labels
//    for (size_t i=0; i<x.size(); ++i) {
//      xSum[z[i]] += x[i]; // TODO: can fold in above as well
//    }
    for (size_t k=0; k<K; ++k) {
      if (counts[k] > 0) {
        vmfs[k] = base.posterior(xSum[k],counts[k]).sample(rnd);
      }
    }
    std::cout << "counts " << K << ": ";
    for (size_t k=0; k<K; ++k) if (counts[k] > 0) std::cout << counts[k] << " ";
    std::cout << "\ttaus: " ;
    for (size_t k=0; k<K; ++k) if (counts[k] > 0) std::cout << vmfs[k].tau_ << " ";
    std::cout << std::endl;
  }
  return 0;
}
