#pragma once

#include <stdint.h>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace tdp {

template<typename T, uint32_t D>
class Normal
{
public:
  Normal(const Eigen::Matrix<T,D,1>& mu, const
      Eigen::Matrix<T,D,D>& Sigma, T pi);
  Normal(const Normal<T,D>& other);
  ~Normal() = default;

  T pdf(const Eigen::Matrix<T,D,1>& x) const;
  T logPdf(const Eigen::Matrix<T,D,1>& x) const;
  T logPdfSlower(const Eigen::Matrix<T,D,1>& x) const;
  T logPdf(const Eigen::Matrix<T,D,D>& scatter, 
      const Eigen::Matrix<T,D,1>& mean, T count) const;

  void Print() const;

  T GetPi() const 
  {return pi_;};

  const Eigen::Matrix<T,D,D>& GetSigma() const 
  {return Sigma_;};

  const Eigen::Matrix<T,D,1>& GetMu() const 
  {return mu_;};
 
  /// Information Matrix
  const Eigen::Matrix<T,D,D>& GetOmega() const 
  {return Omega_;};

  /// Information Vector
  const Eigen::Matrix<T,D,1>& GetXi() const 
  {return xi_;};

  void SetSigma(const Eigen::Matrix<T,D,D>& Sigma)
  { Sigma_ = Sigma; SigmaLDLT_.compute(Sigma_); 
    logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();};

  T GetLogDetSigma() const 
  {return logDetSigma_;};

  const Eigen::LDLT<Eigen::Matrix<T,D,D> >& GetSigmaLDLT() const
  {return SigmaLDLT_;};

private:
  static T LOG_2PI;
  Eigen::Matrix<T,D,1> mu_;
  Eigen::Matrix<T,D,D> Sigma_;
  T pi_;
  // helpers for fast computation
  T logDetSigma_;
  Eigen::LDLT<Eigen::Matrix<T,D,D> > SigmaLDLT_;
  Eigen::Matrix<T,D,D> Omega_; // Information Matrix = Sigma^-1
  Eigen::Matrix<T,D,1> xi_; // Information Vector = Sigma^-1 mu
};

typedef Normal<float,3> Normal3f;
typedef Normal<double,3> Normal3d;

}
#include <tdp/distributions/normal_impl.h>
