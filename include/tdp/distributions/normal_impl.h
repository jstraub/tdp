namespace tdp {

template<typename T, uint32_t D>
Normal<T,D>::Normal(const Eigen::Matrix<T,D,1>& mu, const
    Eigen::Matrix<T,D,D>& Sigma, T pi)
  : mu_(mu), Sigma_(Sigma), pi_(pi), 
    SigmaLDLT_(Sigma_), Omega_(Sigma.inverse()), xi_(SigmaLDLT_.solve(mu)) {
  // equivalent to log(det(Sigma)) but more stable for small values
  logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();
}

template<typename T, uint32_t D>
Normal<T,D>::Normal(const Normal<T,D>& other)
  : mu_(other.GetMu()), Sigma_(other.GetSigma()), pi_(other.GetPi()),
  logDetSigma_(other.logDetSigma_), SigmaLDLT_(Sigma_),
  Omega_(Sigma_.inverse()), xi_(SigmaLDLT_.solve(mu_)) {
}

template<typename T, uint32_t D>
T Normal<T,D>::pdf(const Eigen::Matrix<T,D,1>& x) const {
  return exp(-0.5*((x-mu_).transpose()*SigmaLDLT_.solve(x-mu_)).sum()) 
   / sqrt(exp(LOG_2PI*D + logDetSigma_));
}

template<typename T, uint32_t D>
T Normal<T,D>::logPdf(const Eigen::Matrix<T,D,1>& x) const {
  return -0.5*(LOG_2PI*D + logDetSigma_
  +((x-mu_).transpose()*SigmaLDLT_.solve(x-mu_)).sum() );
}

template<typename T, uint32_t D>
T Normal<T,D>::logPdfSlower(const Eigen::Matrix<T,D,1>& x) const {
  return -0.5*(LOG_2PI*D + logDetSigma_
  +((x-mu_).transpose()*Sigma_.fullPivHouseholderQr().solve(x-mu_)).sum() );
}

template<typename T, uint32_t D>
T Normal<T,D>::logPdf(const Eigen::Matrix<T,D,D>& scatter, 
      const Eigen::Matrix<T,D,1>& mean, T count) const {
  return -0.5*((LOG_2PI*D + logDetSigma_)*count
      + count*(mu_.transpose()*SigmaLDLT_.solve(mu_)).sum() 
      -2.*count*(mean.transpose()*SigmaLDLT_.solve(mu_)).sum()
      +(SigmaLDLT_.solve(scatter + mean*mean.transpose()*count )).trace());
}

template<typename T, uint32_t D>
void Normal<T,D>::Print() const {
  std::cout<< "Normal: pi=" << pi_ << " mu="<<mu_.transpose()<<std::endl;
  std::cout<<Sigma_<<std::endl;
}

}
