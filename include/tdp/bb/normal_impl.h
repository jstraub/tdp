namespace bb {

template<uint32_t D>
Normal<D>::Normal(const Eigen::Matrix<double,D,1>& mu, const
    Eigen::Matrix<double,D,D>& Sigma, double pi)
  : mu_(mu), Sigma_(Sigma), pi_(pi), 
    SigmaLDLT_(Sigma_), Omega_(Sigma.inverse()), xi_(SigmaLDLT_.solve(mu)) {
  // equivalent to log(det(Sigma)) but more stable for small values
  logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();
}

template<uint32_t D>
Normal<D>::Normal(const Normal<D>& other)
  : mu_(other.GetMu()), Sigma_(other.GetSigma()), pi_(other.GetPi()),
  logDetSigma_(other.logDetSigma_), SigmaLDLT_(Sigma_),
  Omega_(Sigma_.inverse()), xi_(SigmaLDLT_.solve(mu_)) {
}

template<uint32_t D>
double Normal<D>::pdf(const Eigen::Matrix<double,D,1>& x) const {
  return exp(-0.5*((x-mu_).transpose()*SigmaLDLT_.solve(x-mu_)).sum()) 
   / sqrt(exp(LOG_2PI*D + logDetSigma_));
}

template<uint32_t D>
double Normal<D>::logPdf(const Eigen::Matrix<double,D,1>& x) const {
  return -0.5*(LOG_2PI*D + logDetSigma_
  +((x-mu_).transpose()*SigmaLDLT_.solve(x-mu_)).sum() );
}

template<uint32_t D>
double Normal<D>::logPdfSlower(const Eigen::Matrix<double,D,1>& x) const {
  return -0.5*(LOG_2PI*D + logDetSigma_
  +((x-mu_).transpose()*Sigma_.fullPivHouseholderQr().solve(x-mu_)).sum() );
}

template<uint32_t D>
double Normal<D>::logPdf(const Eigen::Matrix<double,D,D>& scatter, 
      const Eigen::Matrix<double,D,1>& mean, double count) const {
  return -0.5*((LOG_2PI*D + logDetSigma_)*count
      + count*(mu_.transpose()*SigmaLDLT_.solve(mu_)).sum() 
      -2.*count*(mean.transpose()*SigmaLDLT_.solve(mu_)).sum()
      +(SigmaLDLT_.solve(scatter + mean*mean.transpose()*count )).trace());
}

template<uint32_t D>
void Normal<D>::Print() const {
  std::cout<< "Normal: pi=" << pi_ << " mu="<<mu_.transpose()<<std::endl;
  std::cout<<Sigma_<<std::endl;
}

}
