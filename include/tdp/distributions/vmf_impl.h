/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace tdp {

template<uint32_t D>
double ComputeLogvMFtovMFcost(const vMF<D>& vmf_A, const vMF<D>& vmf_B, 
  const Eigen::Matrix<double, D, 1>& mu_B_prime) {
  const double C = log(2.*M_PI) + log(vmf_A.GetPi()) +
    log(vmf_B.GetPi()) + vmf_A.GetLogZ() + vmf_B.GetLogZ();
  const double z_AB = (vmf_A.GetTau()*vmf_A.GetMu() +
      vmf_B.GetTau()*mu_B_prime).norm();
//  std::cout << mu_B_prime.transpose() << std::endl;
//  std::cout << (vmf_A.GetTau()*vmf_A.GetMu() +
//      vmf_B.GetTau()*mu_B_prime).transpose() << std::endl;
//  std::cout << C << " z_AB " << z_AB << " C " << (C + ComputeLog2SinhOverZ(z_AB))
//    << std::endl;
  return C + ComputeLog2SinhOverZ(z_AB);
};

template <uint32_t D>
vMF<D>::vMF(const Eigen::Matrix<double, D, 1>& mu, double tau, double
    pi) : mu_(mu), tau_(tau), pi_(pi)
{}

template <uint32_t D>
double vMF<D>::GetLogZ() const {
  return -ComputeLog2SinhOverZ(tau_) - log(2.*M_PI);
}

template <uint32_t D>
double vMF<D>::MLEstimateTau(const Eigen::Vector3d& xSum, const
    Eigen::Vector3d& mu, double count) {
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

}
