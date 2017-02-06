/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 
#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <algorithm>
#include <random>

#include <iostream>

#define LOG_2PI 1.8378770664093453

template<typename T, int D>
class Normal
{
 public:

  Normal();
  Normal(const Eigen::Matrix<T,D,1>& mu, const Eigen::Matrix<T,D,D>& Sigma);
  Normal(const Eigen::Matrix<T,D,D>& Sigma);
  Normal(const Normal<T,D>& other);
  ~Normal();

  T logPdf(const Eigen::Matrix<T,D,D>& x) const;
  T logPdfSlower(const Eigen::Matrix<T,D,D>& x) const;
  T logPdf(const Eigen::Matrix<T,D,D>& scatter, 
      const Eigen::Matrix<T,D,1>& mean, T count) const;
  T logPdfSlower(const Eigen::Matrix<T,D,D>& scatter, 
      const Eigen::Matrix<T,D,1>& mean, T count) const;
  T logPdf(const Eigen::Matrix<T,D,D>& scatter, T count) const;

  Eigen::Matrix<T,D,1> sample(std::mt19937& rnd);

  void print() const;

  const Eigen::Matrix<T,D,D>& Sigma() const {return Sigma_;};
  void setSigma(const Eigen::Matrix<T,D,D>& Sigma)
  { Sigma_ = Sigma; SigmaLDLT_.compute(Sigma_); 
    logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();};
  T logDetSigma() const {return logDetSigma_;};
  const Eigen::LDLT<Eigen::Matrix<T,D,D> >& SigmaLDLT() const {return SigmaLDLT_;};

  Eigen::Matrix<T,D,1> mu_;
  Eigen::Matrix<T,D,D> Sigma_;
private:

  // helpers for fast computation
  T logDetSigma_;
  Eigen::LDLT<Eigen::Matrix<T,D,D> > SigmaLDLT_;

  std::normal_distribution<T> gauss_;
};

template<typename T, int D>
Normal<T,D>::Normal(const Eigen::Matrix<T,D,1>& mu, const Eigen::Matrix<T,D,D>& Sigma)
  : mu_(mu), Sigma_(Sigma), SigmaLDLT_(Sigma_), gauss_(0,1)
{
  // equivalent to log(det(Sigma)) but more stable for small values
  logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();
};

template<typename T, int D>
Normal<T,D>::Normal(const Eigen::Matrix<T,D,D>& Sigma)
  : mu_(Eigen::Matrix<T,D,1>::Zero()), Sigma_(Sigma), SigmaLDLT_(Sigma_) , gauss_(0,1)
{
  // equivalent to log(det(Sigma)) but more stable for small values
  logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();
};

template<typename T, int D>
Normal<T,D>::Normal()
  : mu_(Eigen::Matrix<T,D,1>::Zero()), Sigma_(Eigen::Matrix<T,D,D>::Identity()), SigmaLDLT_(Sigma_), gauss_(0,1)
{
  // equivalent to log(det(Sigma)) but more stable for small values
  logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();
};

template<typename T, int D>
Normal<T,D>::Normal(const Normal<T,D>& other)
  : mu_(other.mu_), Sigma_(other.Sigma_),
  logDetSigma_(other.logDetSigma_), SigmaLDLT_(Sigma_) , gauss_(0,1)
{};

template<typename T, int D>
Normal<T,D>::~Normal()
{}

template<typename T, int D>
T Normal<T,D>::logPdf(const Eigen::Matrix<T,D,D>& x) const
{
  return -0.5*(LOG_2PI*D + logDetSigma_ +((x-mu_).transpose()*SigmaLDLT_.solve(x-mu_)).sum() );
}

template<typename T, int D>
T Normal<T,D>::logPdfSlower(const Eigen::Matrix<T,D,D>& x) const
{
  return -0.5*(LOG_2PI*D + logDetSigma_
//      +((x-mu_).transpose()*Sigma_.inverse()*(x-mu_)).sum() );
  +((x-mu_).transpose()*Sigma_.fullPivHouseholderQr().solve(x-mu_)).sum() );
}

template<typename T, int D>
T Normal<T,D>::logPdf(const Eigen::Matrix<T,D,D>& scatter, 
      const Eigen::Matrix<T,D,1>& mean, T count) const
{
	Eigen::Matrix<T,1,D> meanT = mean.transpose();
	Eigen::Matrix<T,D,D> CovMu= SigmaLDLT_.solve(mu_);	
	return -0.5*((LOG_2PI*D + logDetSigma_)*count 
		+ count*(mu_.transpose()*CovMu).sum() 
		-2.*count*(meanT*CovMu).sum()
		+(SigmaLDLT_.solve(scatter + mean*meanT*count)).trace());

  //return -0.5*((LOG_2PI*D + logDetSigma_)*count
  //    + count*(mu_.transpose()*SigmaLDLT_.solve(mu_)).sum() 
  //    -2.*count*(mean.transpose()*SigmaLDLT_.solve(mu_)).sum()
  //    +(SigmaLDLT_.solve(scatter + mean*mean.transpose()*count )).trace());
}

template<typename T, int D>
T Normal<T,D>::logPdfSlower(const Eigen::Matrix<T,D,D>& scatter, 
      const Eigen::Matrix<T,D,1>& mean, T count) const
{
	Eigen::Matrix<T,1,D> meanT = mean.transpose();
	Eigen::Matrix<T,D,D> CovMu= 
    Sigma_.fullPivHouseholderQr().solve(mu_);
	return -0.5*((LOG_2PI*D + logDetSigma_)*count 
		+ count*(mu_.transpose()*CovMu).sum() 
		-2.*count*(meanT*CovMu).sum()
		+(Sigma_.fullPivHouseholderQr().solve(
      scatter + mean*meanT*count)).trace());
}

template<typename T, int D>
T Normal<T,D>::logPdf(const Eigen::Matrix<T,D,D>& scatter, T count) const
{
  assert(false);
//  cout<<count*(mu_.transpose()*SigmaLDLT_.solve(mu_)).sum()<<endl;
//  cout<<(SigmaLDLT_.solve(scatter)).trace()<<endl;
  return -0.5*((LOG_2PI*D + logDetSigma_)*count
      + count*(mu_.transpose()*SigmaLDLT_.solve(mu_)).sum() 
      +(SigmaLDLT_.solve(scatter)).trace());
}

template<typename T, int D>
Eigen::Matrix<T,D,1> Normal<T,D>::sample(std::mt19937& rnd)
{
  // populate the mean
  Eigen::Matrix<T,D,1> x(D);
  for (uint32_t d=0; d<D; d++)
    x[d] = gauss_(rnd); //gsl_ran_gaussian(r,1);
  Eigen::Matrix<T,D,D> sqrtD = Eigen::Matrix<T,D,D>::Zero(D,D); 
  sqrtD.diagonal() = SigmaLDLT_.vectorD();
  sqrtD = sqrtD.array().sqrt();
  return (SigmaLDLT_.matrixL()*sqrtD*SigmaLDLT_.matrixU())*x + mu_;
};

