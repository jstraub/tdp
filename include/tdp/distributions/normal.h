#pragma once

#include <stdint.h>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace tdp {

template<uint32_t D>
class Normal
{
public:
  Normal(const Eigen::Matrix<double,D,1>& mu, const
      Eigen::Matrix<double,D,D>& Sigma, double pi);
  Normal(const Normal<D>& other);
  ~Normal() = default;

  double pdf(const Eigen::Matrix<double,D,1>& x) const;
  double logPdf(const Eigen::Matrix<double,D,1>& x) const;
  double logPdfSlower(const Eigen::Matrix<double,D,1>& x) const;
  double logPdf(const Eigen::Matrix<double,D,D>& scatter, 
      const Eigen::Matrix<double,D,1>& mean, double count) const;

  void Print() const;

  double GetPi() const 
  {return pi_;};

  const Eigen::Matrix<double,D,D>& GetSigma() const 
  {return Sigma_;};

  const Eigen::Matrix<double,D,1>& GetMu() const 
  {return mu_;};
 
  /// Information Matrix
  const Eigen::Matrix<double,D,D>& GetOmega() const 
  {return Omega_;};

  /// Information Vector
  const Eigen::Matrix<double,D,1>& GetXi() const 
  {return xi_;};

  void SetSigma(const Eigen::Matrix<double,D,D>& Sigma)
  { Sigma_ = Sigma; SigmaLDLT_.compute(Sigma_); 
    logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();};

  double GetLogDetSigma() const 
  {return logDetSigma_;};

  const Eigen::LDLT<Eigen::Matrix<double,D,D> >& GetSigmaLDLT() const
  {return SigmaLDLT_;};

private:
  static constexpr double LOG_2PI = log(2.*M_PI);
  Eigen::Matrix<double,D,1> mu_;
  Eigen::Matrix<double,D,D> Sigma_;
  double pi_;
  // helpers for fast computation
  double logDetSigma_;
  Eigen::LDLT<Eigen::Matrix<double,D,D> > SigmaLDLT_;
  Eigen::Matrix<double,D,D> Omega_; // Information Matrix = Sigma^-1
  Eigen::Matrix<double,D,1> xi_; // Information Vector = Sigma^-1 mu
};
}
#include <tdp/distributions/normal_impl.h>
